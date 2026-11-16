"""Exploratory analysis. Must only ever receive data from before the test period."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast.dataset import DAY, PREDICTORS, TARGET, WEEK
from forecast.models import RobustAsinh, naive_forecasts

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
REPORT_NAME = "EDA.md"
FIGURES = "figures"
TABLES_NAME = "eda.json"

# Report sections in reading order: table name -> (title, question it answers, figure file or None).
SECTIONS = {
    "distribution": ("Distribution", "How wide are the tails, and does A go to zero or below?", None),
    "transform": (
        "Effect of the asinh transform on A",
        "Does the target transform remove the skew and the heavy tails? Counts are on a log scale.",
        "distribution.png",
    ),
    "seasonality": (
        "Seasonality of A",
        "How strong are the daily and the weekly pattern? A is averaged per hour of the day and per weekday; "
        "the table gives the lowest and the highest of those averages and when they occur.",
        "seasonality.png",
    ),
    "weekday_means": ("Mean of A by weekday", "Which days differ from the rest?", None),
    "dependence": (
        "Dependence",
        "Rank correlation of A with the predictors and with its own past, and the shape of the relation.",
        "a_vs_predictors.png",
    ),
    "naive_benchmarks": (
        "Naive benchmarks",
        "Error of repeating yesterday, last week, or the mix of both. The autocorrelation shows why: "
        "peaks every 24 hours and the highest at one week.",
        "autocorrelation.png",
    ),
    "yearly_means": (
        "Yearly means (drift)",
        "Do the levels move over the years, i.e. will the future look like the past?",
        "drift.png",
    ),
}


def robust_asinh(x: pd.Series) -> pd.Series:
    """Apply the target transform the models use to a whole series.

    Args:
        x: Series to transform.
    """
    return RobustAsinh().fit(x).transform(x)


def tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute the numeric answers to the EDA questions, one table per section.

    Args:
        df: Hourly data with the columns A, N_1, N_2.
    """
    a = df[TARGET]
    index = pd.DatetimeIndex(df.index)
    spread = (df[PREDICTORS[0]] - df[PREDICTORS[1]]).rename("N_1-N_2")

    stats = df.describe(percentiles=[0.01, 0.5, 0.99]).T
    stats["skew"], stats["kurtosis"] = df.skew(), df.kurt()
    stats["share<=0"] = (df <= 0).mean()

    transformed = robust_asinh(a)
    transform = pd.DataFrame(
        {"skew": [a.skew(), transformed.skew()], "kurtosis": [a.kurt(), transformed.kurt()]},
        index=["raw", "robust asinh"],
    )

    naive = naive_forecasts(a)
    errors = naive.rsub(a.loc[naive.index], axis=0)
    benchmark = pd.DataFrame({"MAE": errors.abs().mean(), "RMSE": np.sqrt((errors**2).mean())})

    drivers = pd.concat(
        [df[PREDICTORS], spread, a.shift(DAY).rename("A lag 1d"), a.shift(WEEK).rename("A lag 7d")],
        axis=1,
    )
    dependence = drivers.corrwith(a, method="spearman").to_frame("spearman with A").T

    by_hour = a.groupby(index.hour).mean().rename("{:02d}:00".format)
    by_weekday = a.groupby(index.day_of_week).mean().set_axis(WEEKDAYS)
    # Size of the daily and of the weekly pattern: the lowest and highest mean of A and when they occur.
    seasonality = pd.DataFrame(
        [
            {
                "lowest mean": means.min(),
                "lowest at": means.idxmin(),
                "highest mean": means.max(),
                "highest at": means.idxmax(),
                "difference": np.ptp(means),
            }
            for means in (by_hour, by_weekday)
        ],
        index=["over the day (by hour)", "over the week (by weekday)"],
    )

    # A year is shown only if the data reaches its end; a few days of the next year would not be a yearly mean.
    complete_year = index.year < (index[-1] + pd.Timedelta(hours=1)).year
    yearly_means = pd.concat([df, spread], axis=1)[complete_year].groupby(index[complete_year].year).mean()

    return {
        "distribution": stats,
        "transform": transform,
        "seasonality": seasonality,
        "weekday_means": by_weekday.to_frame("mean A").T,
        "dependence": dependence,
        "naive_benchmarks": benchmark,
        "yearly_means": yearly_means,
    }


def figures(df: pd.DataFrame, out_dir: Path) -> None:
    """Save the plots that go with the tables.

    Args:
        df: Hourly data with the columns A, N_1, N_2.
        out_dir: Folder the PNG files are written to.
    """
    a = df[TARGET]
    index = pd.DatetimeIndex(df.index)
    spread = df[PREDICTORS[0]] - df[PREDICTORS[1]]

    fig, ax = plt.subplots(figsize=(8, 4))
    profile = a.groupby([index.day_of_week, index.hour]).mean().unstack(0)
    profile.set_axis(WEEKDAYS, axis=1).plot(ax=ax)
    ax.set(xlabel="hour of day", ylabel="mean A", title="Daily profile of A by weekday")
    fig.savefig(out_dir / "seasonality.png", dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax, col in zip(axes, df.columns):
        df[col].resample("MS").mean().plot(ax=ax)
        ax.set(ylabel=col)
    axes[0].set(title="Monthly means")
    fig.savefig(out_dir / "drift.png", dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, x in zip(axes, [df[PREDICTORS[0]], df[PREDICTORS[1]], spread.rename("N_1 - N_2")]):
        ax.scatter(x, a, s=1, alpha=0.1)
        ax.set(xlabel=x.name)
    axes[0].set(ylabel="A")
    fig.savefig(out_dir / "a_vs_predictors.png", dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    a.hist(bins=200, ax=axes[0])
    robust_asinh(a).hist(bins=200, ax=axes[1])
    axes[0].set(title="A", yscale="log")
    axes[1].set(title="A after robust asinh", yscale="log")
    fig.savefig(out_dir / "distribution.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 4))
    lags = range(1, 2 * WEEK + 1)
    ax.bar(lags, [a.autocorr(lag) for lag in lags], width=1)
    ax.set(xlabel="lag (hours)", ylabel="autocorrelation of A")
    fig.savefig(out_dir / "autocorrelation.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def run_eda(df: pd.DataFrame, out_dir: Path) -> None:
    """Write all EDA tables to one JSON, the figures as PNG, and a markdown report showing both.

    Args:
        df: Hourly data from before the test period.
        out_dir: Folder the report and the JSON are written to, with the figures in a subfolder.
    """
    (out_dir / FIGURES).mkdir(parents=True, exist_ok=True)
    figures(df, out_dir / FIGURES)
    results = tables(df)

    # One table per key, as {row: {column: value}}.
    data = {name: json.loads(table.to_json(orient="index")) for name, table in results.items()}
    (out_dir / TABLES_NAME).write_text(json.dumps(data, indent=1), encoding="utf-8")

    parts = [
        "# Exploratory analysis",
        f"Data from {df.index[0].date()} to {df.index[-1].date()} ({len(df):,} hourly rows). "
        "The test period is not part of this analysis. All tables are also in "
        f"[{TABLES_NAME}]({TABLES_NAME}).",
    ]
    for name, table in results.items():
        title, question, figure = SECTIONS[name]
        parts += [f"## {title}", question, table.round(3).to_markdown()]
        if figure:
            parts.append(f"![{title}]({FIGURES}/{figure})")
    (out_dir / REPORT_NAME).write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"EDA written to {out_dir / REPORT_NAME}")
