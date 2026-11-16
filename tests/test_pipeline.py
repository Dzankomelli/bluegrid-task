import numpy as np
import pandas as pd
import pytest

from forecast.dataset import PREDICTORS, TARGET
from forecast.features import build_features
from forecast.metrics import evaluate
from forecast.models import HourlyLinear, RobustAsinh, naive_forecasts
from forecast.walk_forward import walk_forward


@pytest.fixture
def create_data():
    """Factory for random hourly data with the columns A, N_1, N_2."""

    def _create_data(days: int = 60) -> pd.DataFrame:
        index = pd.date_range("2020-01-06", periods=days * 24, freq="h")
        rng = np.random.default_rng(0)
        values = rng.uniform(1, 100, size=(len(index), 3))
        return pd.DataFrame(values, index=index, columns=[TARGET, *PREDICTORS])

    return _create_data


def test_features_of_a_day_ignore_information_not_yet_known(create_data):
    """No look-ahead in the inputs.

    Everything not yet known when day T is forecast is changed: A from day T on,
    N_1 and N_2 from day T+1 on. The features of day T must stay the same.
    """
    df = create_data()
    forecast_day = df.index[0] + pd.Timedelta(days=45)
    next_day = forecast_day + pd.Timedelta(days=1)

    tampered = df.copy()
    tampered.loc[tampered.index >= forecast_day, TARGET] += 1000
    tampered.loc[tampered.index >= next_day, PREDICTORS] += 1000

    on_day = lambda f: f[(f.index >= forecast_day) & (f.index < next_day)]
    pd.testing.assert_frame_equal(on_day(build_features(df)), on_day(build_features(tampered)))


def test_forecast_of_a_day_ignores_targets_not_yet_known(create_data):
    """No look-ahead in the training.

    A is changed from day T on. The forecast of day T must stay the same, which
    it only does if the model and the target scaling are fitted on days before T.
    """
    df = create_data()
    X = build_features(df)
    y = df.loc[X.index, TARGET]
    forecast_day = df.index[0] + pd.Timedelta(days=45)
    next_day = forecast_day + pd.Timedelta(days=1)
    end = next_day + pd.Timedelta(hours=23)

    tampered = y.copy()
    tampered[tampered.index >= forecast_day] += 1000

    def forecast(target: pd.Series) -> pd.Series:
        made = walk_forward(X, target, lambda: HourlyLinear(0.1), forecast_day, end, None, 1)
        return made[made.index < next_day]

    pd.testing.assert_series_equal(forecast(y), forecast(tampered))


def test_target_transform_is_undone_by_its_inverse(create_data):
    """Forecasts come back in the original units: transforming A and inverting it returns A."""
    y = create_data()[TARGET]
    scaler = RobustAsinh().fit(y)

    pd.testing.assert_series_equal(scaler.inverse(scaler.transform(y)), y)


def test_naive_forecast_scores_rmae_of_one(create_data):
    """rMAE is measured against the naive forecast: scored against itself, it must be exactly 1."""
    actual = create_data()[TARGET]
    naive = naive_forecasts(actual)
    actual = actual.loc[naive.index]

    scores = evaluate(actual, naive[["mixed"]], naive["mixed"])

    assert scores.loc["mixed", "rMAE"] == pytest.approx(1.0)
