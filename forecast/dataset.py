"""Loading and chronological splitting of the hourly data."""

from pathlib import Path

import pandas as pd

TARGET = "A"
PREDICTORS = ["N_1", "N_2"]
# Hours in a day and in a week, the lags the data is shifted by.
DAY, WEEK = 24, 168

# Split of Lago et al. (2021): a "year" is 52 weeks (52x7 = 364 days) so every split starts
# on the same weekday and holds whole weeks, keeping the weekly price cycle balanced.
# The last two years are the test set, the year before is validation.
YEAR_DAYS = 364
TEST_DAYS = 2 * YEAR_DAYS
VAL_DAYS = YEAR_DAYS


def load(path: Path) -> pd.DataFrame:
    """Read the hourly data and refuse it if hours are missing, duplicated or empty.

    Args:
        path: CSV with a timestamp index and the columns A, N_1, N_2.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if not df.index.equals(pd.date_range(df.index[0], df.index[-1], freq="h")):
        raise ValueError("Index is not a gapless hourly range")
    if df.isna().any().any():
        raise ValueError("Data contains missing values")
    return df


def matches_published(df: pd.DataFrame, path: Path) -> bool:
    """Tell whether our data is, row for row, the published dataset of Lago et al. (2021).

    Timestamps and all three value columns must be equal; only the column names differ.

    Args:
        df: Hourly data as returned by load.
        path: CSV of the published dataset, columns in the order price, load, wind and solar.
    """
    published = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.index.equals(published.index) and bool((df.to_numpy() == published.to_numpy()).all())


def split_starts(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the first timestamp of the validation and of the test period.

    Both are counted back from the last day in the data.

    Args:
        df: Hourly data as returned by load.
    """
    test_start = df.index[-1].normalize() - pd.Timedelta(days=TEST_DAYS - 1)
    val_start = test_start - pd.Timedelta(days=VAL_DAYS)
    return val_start, test_start
