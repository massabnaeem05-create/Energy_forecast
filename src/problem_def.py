"""
Part 2: Forecasting problem definition — target, horizon, train/test split,
and evaluation metrics. Shared by every model in later parts.
"""
import numpy as np
import pandas as pd

TARGET = "Appliances"
HORIZON = 24          # hours ahead
TEST_DAYS = 14        # held-out test window, per assignment Part 6


def train_test_split(df: pd.DataFrame, test_days: int = TEST_DAYS):
    """Chronological split: last `test_days` days held out for testing."""
    test_start = df.index.max() - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
    train = df.loc[:test_start - pd.Timedelta(hours=1)]
    test = df.loc[test_start:]
    assert len(test) == test_days * 24, "Test window should be exactly test_days*24 hours"
    return train, test


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred, eps: float = 1e-6):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100)


def smape(y_true, y_pred, eps: float = 1e-6):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.clip((np.abs(y_true) + np.abs(y_pred)) / 2, eps, None)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def evaluate(y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


def rolling_windows(test: pd.DataFrame, horizon: int = HORIZON):
    """
    Yield (window_start_idx, y_true_window) for each non-overlapping
    24h block in the test period, used for the Part 8 robust comparison.
    """
    n = len(test)
    for start in range(0, n - horizon + 1, horizon):
        yield start, test[TARGET].iloc[start:start + horizon].values


if __name__ == "__main__":
    df = pd.read_csv("data/energy_hourly.csv", index_col="date", parse_dates=True)
    train, test = train_test_split(df)
    print(f"Train: {train.index.min()} -> {train.index.max()} ({len(train)} hrs)")
    print(f"Test:  {test.index.min()} -> {test.index.max()} ({len(test)} hrs)")
    print("Metric sanity check (perfect forecast):",
          evaluate(test[TARGET].values[:24], test[TARGET].values[:24]))
