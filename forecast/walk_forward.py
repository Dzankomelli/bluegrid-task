"""Walk-forward backtest: the model is refitted as time moves on, as it would be live."""

from collections.abc import Callable

import pandas as pd

from forecast.models import RobustAsinh


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    make_model: Callable,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_days: int | None,
    retrain_days: int,
) -> pd.Series:
    """Forecast every day from start to end using only earlier days for fitting.

    A model fitted for day T sees targets up to day T-1. Between refits the same
    model is reused, but its inputs are still the up-to-date features of each day.

    Args:
        X: Feature table, one row per hour.
        y: Target on the same index.
        make_model: Returns a new unfitted model.
        start: First forecast day.
        end: Last timestamp to forecast.
        window_days: Length of the training window, None for all history.
        retrain_days: Days between refits.
    """
    forecasts = []
    for fit_day in pd.date_range(start, end, freq=f"{retrain_days}D"):
        train = X.index < fit_day
        if window_days:
            train &= X.index >= fit_day - pd.Timedelta(days=window_days)
        block = (X.index >= fit_day) & (X.index < fit_day + pd.Timedelta(days=retrain_days)) & (X.index <= end)

        scaler = RobustAsinh().fit(y[train])
        model = make_model().fit(X[train], scaler.transform(y[train]))
        forecasts.append(scaler.inverse(model.predict(X[block])))
    return pd.concat(forecasts)
