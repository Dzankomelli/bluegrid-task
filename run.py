"""Runs the whole pipeline from the raw CSV to the files in results/, one function per stage."""

import io
import json
import sys
from collections.abc import Callable
from functools import partial
from itertools import product
from pathlib import Path
from typing import NamedTuple

import matplotlib

# Figures are only saved to files, never shown.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from forecast.dataset import PREDICTORS, TARGET, load, matches_published, split_starts
from forecast.eda import run_eda
from forecast.features import build_features, correlated_features
from forecast.metrics import evaluate, mae, mae_by_year
from forecast.models import CALENDAR, Boosted, HourlyLinear, LinearPlusBoosted, naive_forecasts
from forecast.walk_forward import walk_forward

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "podatki.csv"
# Dataset and forecasts published with Lago et al. (2021); our data turned out to be their German dataset.
LAGO_DATA = ROOT / "data" / "lago_DE.csv"
LAGO_FORECASTS = ROOT / "data" / "lago_forecasts_DE.csv"
RESULTS = ROOT / "results"

CORRELATION_LIMIT = 0.95
# Candidates compared on the validation year. None = all history.
WINDOWS = [364, 728, None]
ALPHAS = [0.001, 0.003, 0.01, 0.03]
DEPTHS = [3, 6]
RETRAIN_DAYS = [7, 1]
TUNING_RETRAIN_DAYS = 7


class Candidate(NamedTuple):
    """One model setup: what to fit, on how much history, and how often to refit.

    Attributes:
        name: Model name used in the output tables.
        make_model: Returns a new unfitted model.
        window_days: Training window in days, None for all history.
        retrain_days: Days between refits.
        settings: Model settings, recorded with each trial.
    """

    name: str
    make_model: Callable
    window_days: int | None
    retrain_days: int
    settings: dict

    def forecast(self, X: pd.DataFrame, y: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Walk this setup forward, refitting on earlier days only.

        Args:
            X: Model inputs.
            y: Target on the same index.
            start: First forecast day.
            end: Last timestamp to forecast.
        """
        return walk_forward(X, y, self.make_model, start, end, self.window_days, self.retrain_days)

    def describe(self) -> dict:
        """The setup as a flat record for the results file."""
        return {"model": self.name, **self.settings, "window_days": self.window_days, "retrain_days": self.retrain_days}


class Trial(NamedTuple):
    """A candidate and its MAE on the validation year."""

    candidate: Candidate
    mae: float


def prepare_features(df: pd.DataFrame, val_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build all features, then drop the correlated ones, as decided on the training period only.

    Returns the full feature table, the filtered model inputs and the target on their index.

    Args:
        df: Hourly data.
        val_start: First hour of the validation period; earlier hours are training data.
    """
    # All 19 features for every hour with enough history; the first weeks are dropped.
    features = build_features(df)
    y = df.loc[features.index, TARGET]
    # The correlation filter only sees the training period, so validation and test cannot influence it.
    is_train = features.index < val_start
    dropped = correlated_features(features[is_train], y[is_train], CORRELATION_LIMIT)
    print(f"\ndropped as correlated: {dropped}")
    # X, the filtered table, is what all three models are trained on.
    return features, features.drop(columns=dropped), y


def validate(
    X: pd.DataFrame, y: pd.Series, period: tuple[pd.Timestamp, pd.Timestamp], candidates: list[Candidate]
) -> list[Trial]:
    """Walk every candidate through the validation period and score it by MAE.

    Args:
        X: Model inputs.
        y: Target on the same index.
        period: First and last timestamp of the validation period.
        candidates: Setups to compare.
    """
    trials = []
    for candidate in candidates:
        forecast = candidate.forecast(X, y, *period)
        trials.append(Trial(candidate, float(mae(y[forecast.index], forecast))))
        print({**candidate.describe(), "MAE": trials[-1].mae})
    return trials


def tune(
    X: pd.DataFrame, y: pd.Series, val_start: pd.Timestamp, test_start: pd.Timestamp
) -> tuple[dict[str, Trial], list[Trial]]:
    """Choose the settings of each model on the validation year, in four stages.

    1. linear: penalty and window, 2. boosted: depth and window, 3. linear+boosted:
    window, with the penalty and depth from 1 and 2, 4. every model: refit interval.
    Stages 1-3 refit weekly to keep the search fast.

    Returns the chosen trial of each model and every trial made.

    Args:
        X: Model inputs.
        y: Target on the same index.
        val_start: First hour of the validation period.
        test_start: First hour of the test period, which tuning never sees.
    """
    # Every candidate is walked through the validation year only, ending the hour before the test period.
    run = partial(validate, X, y, (val_start, test_start - pd.Timedelta(hours=1)))
    weekly = TUNING_RETRAIN_DAYS

    def best(trials: list[Trial]) -> Trial:
        # Lowest MAE wins; on a tie the earlier candidate in the grid is kept.
        return min(trials, key=lambda trial: trial.mae)

    # Stage 1, linear model: every LASSO penalty with every training window.
    linear_trials = run(
        [Candidate("linear", partial(HourlyLinear, a), w, weekly, {"alpha": a}) for a, w in product(ALPHAS, WINDOWS)]
    )
    # Stage 2, boosted trees: every tree depth with every training window.
    boosted_trials = run(
        [Candidate("boosted", partial(Boosted, d), w, weekly, {"max_depth": d}) for d, w in product(DEPTHS, WINDOWS)]
    )
    # Stage 3, linear + boosted: reuses the best penalty and depth from stages 1 and 2, only the window is searched.
    settings = {**best(linear_trials).candidate.settings, **best(boosted_trials).candidate.settings}
    make_combined = partial(LinearPlusBoosted, settings["alpha"], settings["max_depth"])
    combined_trials = run([Candidate("linear+boosted", make_combined, w, weekly, settings) for w in WINDOWS])
    trials = linear_trials + boosted_trials + combined_trials

    # Stage 4, every model: its best weekly setup is also tried with the other refit intervals (daily).
    chosen = {}
    for weekly_best in (best(linear_trials), best(boosted_trials), best(combined_trials)):
        # The weekly refit is already scored; only the other intervals are new.
        others = [r for r in RETRAIN_DAYS if r != weekly]
        refit_trials = run([weekly_best.candidate._replace(retrain_days=r) for r in others])
        trials += refit_trials
        chosen[weekly_best.candidate.name] = best([weekly_best, *refit_trials])
    return chosen, trials


def forecast_test(
    df: pd.DataFrame,
    features: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    chosen: dict[str, Candidate],
    test_start: pd.Timestamp,
) -> pd.DataFrame:
    """Forecast the test period once with the frozen settings, next to the naive and the N-only forecasts.

    Args:
        df: Hourly data.
        features: Full feature table, before the correlation filter.
        X: Model inputs.
        y: Target on the same index.
        chosen: Frozen setup of each model.
        test_start: First hour of the test period.
    """
    end = df.index[-1]
    print("\nrunning the test period, this is the longest step ...")
    # Our three models (linear, boosted, linear+boosted), each with its frozen settings and all features in X.
    forecasts = pd.DataFrame({name: c.forecast(X, y, test_start, end) for name, c in chosen.items()})
    # Reference run, not a fourth model: the linear model with its frozen settings, but given only the raw
    # N_1, N_2, hour and weekday. Comparing it with "linear" shows what the engineered features add.
    n_only = chosen["linear"].forecast(features[[*PREDICTORS, *CALENDAR]], y, test_start, end)
    forecasts.insert(0, "linear, N only", n_only)
    # Baseline, nothing fitted: yesterday's A for Tuesday to Friday, last week's A for Saturday to Monday.
    forecasts.insert(0, "naive", naive_forecasts(df[TARGET]).loc[forecasts.index, "mixed"])
    return forecasts


def score(forecasts: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the published forecasts, score everything, and save the predictions and the error-by-hour plot.

    Returns the metrics, one row per forecast, and the MAE of every forecast by year.

    Args:
        forecasts: One column per forecast over the test period, including "naive".
        y: Target, at least on the test period.
    """
    actual = y[forecasts.index]
    # The published forecasts are only comparable if they were made for exactly our test hours and values.
    lago = pd.read_csv(LAGO_FORECASTS, index_col=0, parse_dates=True)
    if not lago["Real price"].equals(actual.rename("Real price")):
        raise ValueError("Published forecasts do not cover the same values as our test period")
    # Their two ensembles are added as extra columns and scored by the same code as ours.
    forecasts = forecasts.assign(
        **{"Lago LEAR ensemble": lago["LEAR Ensemble"], "Lago DNN ensemble": lago["DNN Ensemble"]}
    )
    forecasts.assign(actual=actual).to_csv(RESULTS / "predictions.csv")
    # rMAE of every forecast is measured against the naive baseline.
    scores = evaluate(actual, forecasts, forecasts["naive"])
    print(f"test results\n{scores.round(3)}")

    # Mean absolute error of every forecast for each hour of the day.
    errors = forecasts.rsub(actual, axis=0).abs()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    errors.groupby(pd.DatetimeIndex(errors.index).hour).mean().plot(ax=ax)
    ax.set(xlabel="hour of day", ylabel="MAE", title="Test error by hour of day")
    fig.savefig(RESULTS / "error_by_hour.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    return scores, mae_by_year(actual, forecasts)


def main() -> None:
    """Run the stages of the README flow chart in order and save every number to results.json."""
    # Show progress line by line even when the output is redirected to a file.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    RESULTS.mkdir(exist_ok=True)

    # Load and check the data. It must be the published dataset row for row, or the comparison is meaningless.
    df = load(DATA)
    if not matches_published(df, LAGO_DATA):
        raise ValueError("Data is not the published dataset, so the published forecasts cannot be compared")
    # Train until val_start, validation year until test_start, the last 728 days are the test period.
    val_start, test_start = split_starts(df)
    print(f"validation from {val_start.date()}, test from {test_start.date()}\n")

    # Exploratory analysis. The test period is never looked at before the final evaluation.
    run_eda(df[df.index < test_start], RESULTS / "eda")
    # Build the features and drop the correlated ones.
    features, X, y = prepare_features(df, val_start)
    # Choose each model's settings on the validation year and freeze them.
    chosen, trials = tune(X, y, val_start, test_start)
    candidates = {name: trial.candidate for name, trial in chosen.items()}
    # Forecast the test period once, with the naive baseline and the N-only reference next to the models.
    forecasts = forecast_test(df, features, X, y, candidates, test_start)
    # Add the published forecasts and score everything the same way.
    scores, yearly = score(forecasts, y)

    # Every number of the run in one file; the report quotes these.
    results = {
        "data": {
            "rows": len(df),
            "first_hour": str(df.index[0]),
            "last_hour": str(df.index[-1]),
            "validation_start": str(val_start),
            "test_start": str(test_start),
            "identical_to_published": True,
        },
        "dropped_features": [c for c in features.columns if c not in X.columns],
        "validation": [{**t.candidate.describe(), "MAE": t.mae} for t in trials],
        "chosen_settings": {name: {**t.candidate.describe(), "validation_MAE": t.mae} for name, t in chosen.items()},
        "test_metrics": scores.to_dict(orient="index"),
        "test_mae_by_year": yearly.to_dict(orient="index"),
    }
    (RESULTS / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
