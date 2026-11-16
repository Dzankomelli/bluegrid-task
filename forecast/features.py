"""Feature table with one row per (day T, hour).

Information rule: when A for day T is forecast, A is known up to and including
day T-1 and N_1, N_2 up to and including day T. So every feature built from A
looks back at least 24 hours, while N features may use any hour of day T.
"""

from itertools import combinations

import pandas as pd

from forecast.dataset import DAY, PREDICTORS, TARGET, WEEK


def _mean_of_lags(x: pd.Series, step: int, count: int, start: int = 0) -> pd.Series:
    """Average of x over the same hour of several earlier days.

    Args:
        x: Hourly series.
        step: Hours between the days averaged (24 = consecutive days, 168 = same weekday).
        count: Number of earlier days averaged.
        start: Extra hours to look back before the first of them.
    """
    lags = [x.shift(start + step * k) for k in range(1, count + 1)]
    # skipna=False: an average over incomplete history stays missing instead of silently shrinking.
    return pd.concat(lags, axis=1).mean(axis=1, skipna=False)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the features for every hour that has enough history.

    Args:
        df: Hourly data with the columns A, N_1, N_2.
    """
    a, n1, n2 = df[TARGET], df[PREDICTORS[0]], df[PREDICTORS[1]]
    index = pd.DatetimeIndex(df.index)
    day = index.normalize()
    spread = n1 - n2
    # Shape of A over the whole previous day, repeated on each hour of day T.
    previous_day = a.groupby(day).agg(["mean", "min", "max"]).shift(1).reindex(day).set_axis(df.index)

    features = pd.DataFrame(
        {
            # Level of the predictors on day T. They are strictly positive, so a ratio is safe.
            "N_1": n1,
            "N_2": n2,
            "spread": spread,
            "ratio": n2 / n1,
            "spread_day_mean": spread.groupby(day).transform("mean"),
            # How day T differs from what A was last set against.
            "spread_chg_1d": spread - spread.shift(DAY),
            "spread_chg_7d": spread - spread.shift(WEEK),
            "N_1_rel_4w": n1 / _mean_of_lags(n1, WEEK, 4),
            "N_2_rel_4w": n2 / _mean_of_lags(n2, WEEK, 4),
            # Recent A. It crosses zero, so it enters as levels and differences, never ratios.
            "A_lag_1d": a.shift(DAY),
            "A_lag_2d": a.shift(2 * DAY),
            "A_lag_7d": a.shift(WEEK),
            "A_prev_mean": previous_day["mean"],
            "A_prev_min": previous_day["min"],
            "A_prev_max": previous_day["max"],
            # Was yesterday unusual for the last week / for its own weekday?
            "A_dev_7d": a.shift(DAY) - _mean_of_lags(a, DAY, 7),
            "A_dev_4w": a.shift(DAY) - _mean_of_lags(a, WEEK, 4, start=DAY),
            "hour": index.hour,
            "weekday": index.day_of_week,
        },
        index=df.index,
    )
    return features.dropna()


def correlated_features(features: pd.DataFrame, target: pd.Series, threshold: float) -> list[str]:
    """Pick the features to drop so that no remaining pair is correlated above threshold.

    Of two correlated features, the one less related to the target goes.
    Rank correlation is used because of the heavy tails.

    Args:
        features: Feature table of the training period.
        target: Target on the same index.
        threshold: Highest absolute rank correlation two kept features may have.
    """
    pairwise = features.corr(method="spearman").abs().to_numpy()
    relevance = features.corrwith(target, method="spearman").abs().to_numpy()
    # Feature pairs by position, most correlated first.
    pairs = sorted(combinations(range(features.shape[1]), 2), key=lambda pair: pairwise[pair], reverse=True)

    dropped = []
    for first, second in pairs:
        if pairwise[first, second] <= threshold:
            break
        if first not in dropped and second not in dropped:
            dropped.append(first if relevance[first] < relevance[second] else second)
    return [str(features.columns[position]) for position in dropped]
