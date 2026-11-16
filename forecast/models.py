"""The naive benchmark and the three forecasting models, which are fitted on the transformed target."""

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation, norm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from forecast.dataset import DAY, WEEK

CALENDAR = ["hour", "weekday"]


def naive_forecasts(a: pd.Series) -> pd.DataFrame:
    """Build the persistence benchmarks: same hour yesterday, same hour last week,
    and the mix standard in the literature (yesterday for Tue-Fri, last week for Sat-Mon).

    Args:
        a: Hourly series of the target.
    """
    daily, weekly = a.shift(DAY), a.shift(WEEK)
    mixed = daily.where(pd.DatetimeIndex(a.index).day_of_week.isin([1, 2, 3, 4]), weekly)
    return pd.DataFrame({"daily": daily, "weekly": weekly, "mixed": mixed}).dropna()


class RobustAsinh:
    """Target transform: centre on the median, scale by MAD, then apply asinh.

    asinh is linear near zero and logarithmic in the tails, so it compresses
    spikes like a log does but is also defined for zero and negative values.
    Median and MAD are used because mean and standard deviation are themselves
    distorted by the spikes.
    """

    def fit(self, y: pd.Series) -> "RobustAsinh":
        """Learn the centre and scale, from training data only.

        Args:
            y: Target values of the training window.
        """
        self.center = y.median()
        # Dividing by the normal 75 % quantile puts MAD on the scale of a standard deviation.
        self.scale = median_abs_deviation(y, scale=float(norm.ppf(0.75)))
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        """Move values to the scale the models are fitted on.

        Args:
            y: Values in original units.
        """
        return np.arcsinh((y - self.center) / self.scale)

    def inverse(self, y: pd.Series) -> pd.Series:
        """Bring model output back to original units.

        Args:
            y: Values on the transformed scale.
        """
        return np.sinh(y) * self.scale + self.center


class HourlyLinear:
    """One LASSO regression per hour of the day.

    The effect of the predictors differs between night and peak hours, which a
    single linear model cannot express. Being linear, it keeps reacting to
    predictor values outside the range seen in training.

    Args:
        alpha: LASSO penalty; larger values switch off more features.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HourlyLinear":
        """Fit a separate regression on the rows of each hour.

        Args:
            X: Feature table including the hour and weekday columns.
            y: Transformed target on the same index.
        """
        self.models = {}
        for hour, rows in X.groupby("hour"):
            # Weekday is a category, not a quantity; hour is constant within one model.
            prepare = ColumnTransformer(
                [
                    ("weekday", OneHotEncoder(handle_unknown="ignore"), ["weekday"]),
                    ("numeric", StandardScaler(), [c for c in X.columns if c not in CALENDAR]),
                ]
            )
            self.models[hour] = make_pipeline(prepare, Lasso(alpha=self.alpha)).fit(rows, y[rows.index])
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Forecast each row with the regression of its hour.

        Args:
            X: Feature table with the same columns as in fit.
        """
        parts = [pd.Series(self.models[hour].predict(rows), index=rows.index) for hour, rows in X.groupby("hour")]
        return pd.concat(parts).reindex(X.index)


class Boosted:
    """One gradient-boosted tree model for all hours, with hour as a feature.

    Trees capture the non-linear shape and interactions, but predict a constant
    once a predictor leaves the range seen in training.

    Args:
        max_depth: Depth of each tree, i.e. how many features may interact.
    """

    def __init__(self, max_depth: int):
        self.model = XGBRegressor(
            max_depth=max_depth, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=0
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Boosted":
        """Fit the trees on all hours at once.

        Args:
            X: Feature table.
            y: Transformed target on the same index.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Forecast every row.

        Args:
            X: Feature table with the same columns as in fit.
        """
        return pd.Series(self.model.predict(X), index=X.index)


class LinearPlusBoosted:
    """Linear model first, then trees trained on what the linear model got wrong.

    The linear part carries the extrapolation, the trees only add a bounded
    non-linear correction on top of it.

    Args:
        alpha: LASSO penalty of the linear part.
        max_depth: Tree depth of the correction.
    """

    def __init__(self, alpha: float, max_depth: int):
        self.linear = HourlyLinear(alpha)
        self.correction = Boosted(max_depth)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearPlusBoosted":
        """Fit the linear part, then the trees on its training errors.

        Args:
            X: Feature table including the hour and weekday columns.
            y: Transformed target on the same index.
        """
        self.linear.fit(X, y)
        self.correction.fit(X, y - self.linear.predict(X))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Forecast as linear prediction plus tree correction.

        Args:
            X: Feature table with the same columns as in fit.
        """
        return self.linear.predict(X) + self.correction.predict(X)
