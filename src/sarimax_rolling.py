"""
Part 4 (continued): rolling 24h evaluation of the fitted SARIMAX model
across the full 14-day test period, using Kalman-filter state updates
(`append(..., refit=False)`) rather than refitting the whole model at
every origin -- this applies the same fixed, already-estimated parameters
to extend the state through new observations, which is standard practice
for rolling-origin evaluation and is far cheaper than refitting 147 grid
combinations x 14 windows.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from problem_def import TARGET, HORIZON, TEST_DAYS, train_test_split, evaluate

BEST_ORDER = (1, 0, 6)
SEASONAL_ORDER = (1, 1, 1, 24)

if __name__ == "__main__":
    df = pd.read_csv("data/energy_hourly.csv", index_col="date", parse_dates=True)
    train, test = train_test_split(df, TEST_DAYS)

    model = SARIMAX(train[TARGET], order=BEST_ORDER, seasonal_order=SEASONAL_ORDER,
                     enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False, maxiter=150, method="lbfgs")

    n_windows = len(test) // HORIZON
    scores = []
    current_res = res

    for w in range(n_windows):
        # Forecast the next 24h from wherever current_res's state currently ends
        fc = current_res.get_forecast(steps=HORIZON)
        y_pred = fc.predicted_mean.values
        y_true = test[TARGET].iloc[w * HORIZON: (w + 1) * HORIZON].values
        scores.append(evaluate(y_true, y_pred))

        # Update state with the true observations for this window (no refit)
        new_obs = test[TARGET].iloc[w * HORIZON: (w + 1) * HORIZON]
        current_res = current_res.append(new_obs, refit=False)
        print(f"Window {w+1}/{n_windows} done: RMSE={scores[-1]['RMSE']:.1f}")

    rolling_df = pd.DataFrame(scores)
    rolling_df.to_csv("outputs/sarimax_rolling_per_window.csv", index=False)
    print("\nMean rolling metrics:\n", rolling_df.mean())
    rolling_df.mean().to_csv("outputs/sarimax_rolling_mean.csv")
