"""The error measures used for every forecast."""

import numpy as np
import pandas as pd

TAIL = 0.05


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    """Mean absolute error: the typical size of a miss.

    Args:
        actual: Realised values.
        forecast: Forecast on the same index.
    """
    return (actual - forecast).abs().mean()


def evaluate(actual: pd.Series, forecasts: pd.DataFrame, naive: pd.Series) -> pd.DataFrame:
    """Score every forecast column against the actual values.

    MAE is the typical error and RMSE punishes large misses. rMAE below 1 means
    better than the naive forecast. Bias is the average of forecast minus actual,
    so a positive value means forecasting too high. Median AE is the error of an
    ordinary hour and max AE the single worst miss. MAE daily mean is the error
    on the average of a whole day. The tail columns are the MAE in the hours
    where the actual value was in its lowest / highest 5 %.

    Args:
        actual: Realised values.
        forecasts: One column per forecast, same index as actual.
        naive: Naive forecast on the same index, the rMAE reference.
    """
    error = forecasts.sub(actual, axis=0)
    low, high = actual <= actual.quantile(TAIL), actual >= actual.quantile(1 - TAIL)
    day = pd.DatetimeIndex(actual.index).normalize()
    return pd.DataFrame(
        {
            "MAE": error.abs().mean(),
            "RMSE": np.sqrt((error**2).mean()),
            "rMAE": error.abs().mean() / mae(actual, naive),
            "bias": error.mean(),
            "median AE": error.abs().median(),
            "max AE": error.abs().max(),
            "MAE daily mean": error.groupby(day).mean().abs().mean(),
            "MAE low tail": error[low].abs().mean(),
            "MAE high tail": error[high].abs().mean(),
        }
    )


def mae_by_year(actual: pd.Series, forecasts: pd.DataFrame) -> pd.DataFrame:
    """MAE of every forecast in each calendar year, to show whether accuracy holds over time.

    Args:
        actual: Realised values.
        forecasts: One column per forecast, same index as actual.
    """
    error = forecasts.sub(actual, axis=0)
    return error.abs().groupby(pd.DatetimeIndex(actual.index).year).mean()
