"""
Part 3: Benchmark forecasting models — Mean, Naive, Daily/Weekly Seasonal
Naive, and Drift. All take a history Series and return a `horizon`-length
forecast array.
"""
import numpy as np
import pandas as pd

from problem_def import TARGET, HORIZON, TEST_DAYS, train_test_split, evaluate


def forecast_mean(history: pd.Series, horizon: int = HORIZON) -> np.ndarray:
    return np.full(horizon, history.mean())


def forecast_naive(history: pd.Series, horizon: int = HORIZON) -> np.ndarray:
    return np.full(horizon, history.iloc[-1])


def forecast_seasonal_naive(history: pd.Series, season: int, horizon: int = HORIZON) -> np.ndarray:
    """Repeat the last full season (season=24 for daily, 168 for weekly)."""
    last_season = history.iloc[-season:].values
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:horizon]


def forecast_drift(history: pd.Series, horizon: int = HORIZON) -> np.ndarray:
    """Naive + linear extrapolation of the average step-to-step change."""
    y0 = history.iloc[-1]
    n = len(history)
    avg_change = (history.iloc[-1] - history.iloc[0]) / (n - 1)
    steps = np.arange(1, horizon + 1)
    return y0 + avg_change * steps


BENCHMARKS = {
    "Mean": lambda h: forecast_mean(h),
    "Naive": lambda h: forecast_naive(h),
    "Daily Seasonal Naive": lambda h: forecast_seasonal_naive(h, season=24),
    "Weekly Seasonal Naive": lambda h: forecast_seasonal_naive(h, season=168),
    "Drift": lambda h: forecast_drift(h),
}


def rolling_evaluate(forecast_fn, full_series: pd.Series, test_index: pd.DatetimeIndex,
                      horizon: int = HORIZON) -> pd.DataFrame:
    """Roll a 24h forecast through the test period, one window per day."""
    scores = []
    n_windows = len(test_index) // horizon
    for w in range(n_windows):
        origin_time = test_index[w * horizon] - pd.Timedelta(hours=1)
        history = full_series.loc[:origin_time]
        y_true_w = full_series.loc[test_index[w * horizon: w * horizon + horizon]].values
        y_pred_w = forecast_fn(history)
        scores.append(evaluate(y_true_w, y_pred_w))
    return pd.DataFrame(scores)


def run_all_benchmarks(df: pd.DataFrame):
    train, test = train_test_split(df, TEST_DAYS)

    # First-24h evaluation (matches assignment's literal instruction)
    y_true = test[TARGET].iloc[:HORIZON].values
    first_window_results = {
        name: evaluate(y_true, fn(train[TARGET])) for name, fn in BENCHMARKS.items()
    }

    # Rolling evaluation across full test period (primary comparison score)
    rolling_results = {
        name: rolling_evaluate(fn, df[TARGET], test.index).mean()
        for name, fn in BENCHMARKS.items()
    }

    return (
        pd.DataFrame(first_window_results).T.sort_values("RMSE"),
        pd.DataFrame(rolling_results).T.sort_values("RMSE"),
    )


if __name__ == "__main__":
    df = pd.read_csv("data/energy_hourly.csv", index_col="date", parse_dates=True)
    first_window_df, rolling_df = run_all_benchmarks(df)

    print("=== First 24h window ===")
    print(first_window_df)
    print("\n=== Rolling 14-window average ===")
    print(rolling_df)

    rolling_df.to_csv("outputs/benchmark_rolling_results.csv")
    print("\nSaved rolling results to outputs/benchmark_rolling_results.csv")
