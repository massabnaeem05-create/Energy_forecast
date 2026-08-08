"""
Part 4 (continued): fit the final SARIMAX model on the full training set,
run residual diagnostics, forecast 24h with confidence intervals, and
evaluate against the test set. Saves reusable artifacts to outputs/.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

from problem_def import TARGET, HORIZON, TEST_DAYS, train_test_split, evaluate

BEST_ORDER = (1, 0, 6)
SEASONAL_ORDER = (1, 1, 1, 24)


def fit_final(train: pd.Series):
    model = SARIMAX(
        train, order=BEST_ORDER, seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=150, method="lbfgs")
    print("Converged:", res.mle_retvals.get("converged"))
    return res


if __name__ == "__main__":
    df = pd.read_csv("data/energy_hourly.csv", index_col="date", parse_dates=True)
    train, test = train_test_split(df, TEST_DAYS)

    print("Fitting final SARIMAX(1,0,6)x(1,1,1,24) on full training data...")
    res = fit_final(train[TARGET])
    print(res.summary())

    # Residual diagnostics
    resid = res.resid
    lb = acorr_ljungbox(resid, lags=[24, 48], return_df=True)
    print("\nLjung-Box test on residuals:\n", lb)
    lb.to_csv("outputs/sarimax_ljungbox.csv")

    # Forecast next 24h (first test window) with confidence intervals
    fc = res.get_forecast(steps=HORIZON)
    forecast_mean = fc.predicted_mean
    conf_int = fc.conf_int(alpha=0.05)

    y_true = test[TARGET].iloc[:HORIZON].values
    metrics = evaluate(y_true, forecast_mean.values)
    print("\nFirst-24h test metrics:", metrics)

    forecast_df = pd.DataFrame({
        "actual": y_true,
        "forecast": forecast_mean.values,
        "lower_ci": conf_int.iloc[:, 0].values,
        "upper_ci": conf_int.iloc[:, 1].values,
    }, index=test.index[:HORIZON])
    forecast_df.to_csv("outputs/sarimax_forecast_24h.csv")

    pd.Series(metrics).to_csv("outputs/sarimax_first_window_metrics.csv")
    print("\nSaved: sarimax_final_model.pkl, sarimax_ljungbox.csv, "
          "sarimax_forecast_24h.csv, sarimax_first_window_metrics.csv")
