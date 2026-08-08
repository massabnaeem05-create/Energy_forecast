"""
Part 4: SARIMAX model.

- Seasonal order fixed at (1,1,1,24): Part 1's decomposition/ACF showed a
  dominant daily (24h) cycle, and a 24h-seasonal difference was confirmed
  stationary, so we set seasonal D=1, s=24, and P=Q=1 as a standard starting
  seasonal order (one seasonal AR and one seasonal MA term).
- Non-seasonal (p, d, q) is chosen by an exhaustive AIC grid search over
  p in [0,6], d in [0,2], q in [0,6], as required by the assignment.
- To keep the 147-combination grid search tractable, the search is run on a
  recent subset of the training data (last SEARCH_WINDOW hours). The winning
  order is then refit on the FULL training set for the final model.
"""
import time
import warnings
import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

SEASONAL_ORDER = (1, 1, 1, 24)
SEARCH_WINDOW = 720   # hours (30 days) used for the AIC grid search, for speed
P_RANGE = range(0, 7)  # 0..6
D_RANGE = range(0, 3)  # 0..2
Q_RANGE = range(0, 7)  # 0..6


def grid_search_orders(train: pd.Series, search_window: int = SEARCH_WINDOW,
                        seasonal_order=SEASONAL_ORDER, maxiter: int = 25):
    """Exhaustive AIC grid search over (p,d,q). Returns a results DataFrame."""
    subset = train.iloc[-search_window:]
    combos = list(itertools.product(P_RANGE, D_RANGE, Q_RANGE))
    records = []

    for i, (p, d, q) in enumerate(combos):
        t0 = time.time()
        try:
            model = SARIMAX(
                subset, order=(p, d, q), seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            res = model.fit(disp=False, maxiter=maxiter, method="lbfgs")
            aic = res.aic
            converged = res.mle_retvals.get("converged", None)
        except Exception:
            aic = np.inf
            converged = False
        records.append({
            "p": p, "d": d, "q": q, "AIC": aic,
            "converged": converged, "fit_seconds": round(time.time() - t0, 2),
        })
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(combos)} combinations tried")

    results = pd.DataFrame(records).sort_values("AIC").reset_index(drop=True)
    return results


def fit_final_model(train: pd.Series, order, seasonal_order=SEASONAL_ORDER):
    model = SARIMAX(
        train, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=100)


if __name__ == "__main__":
    df = pd.read_csv("data/energy_hourly.csv", index_col="date", parse_dates=True)
    y = df["Appliances"]
    train = y.iloc[:-336]

    print(f"Grid search over {len(list(P_RANGE))*len(list(D_RANGE))*len(list(Q_RANGE))} combinations "
          f"on last {SEARCH_WINDOW}h of training data...")
    results = grid_search_orders(train)
    results.to_csv("outputs/sarimax_grid_search.csv", index=False)

    best = results.iloc[0]
    best_order = (int(best.p), int(best.d), int(best.q))
    print(f"\nBest order by AIC: {best_order}, AIC={best.AIC:.1f}")

    print("Refitting on full training set...")
    final_res = fit_final_model(train, best_order)
    print(final_res.summary())
