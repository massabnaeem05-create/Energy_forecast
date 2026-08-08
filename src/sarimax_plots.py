"""
Part 4 (continued): plots -- residual ACF, residual distribution, and
24h forecast with confidence intervals.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf

from problem_def import TARGET, TEST_DAYS, train_test_split

BEST_ORDER = (1, 0, 6)
SEASONAL_ORDER = (1, 1, 1, 24)

df = pd.read_csv("data/energy_hourly.csv", index_col="date", parse_dates=True)
train, test = train_test_split(df, TEST_DAYS)
model = SARIMAX(train[TARGET], order=BEST_ORDER, seasonal_order=SEASONAL_ORDER,
                 enforce_stationarity=False, enforce_invertibility=False)
res = model.fit(disp=False, maxiter=150, method="lbfgs")

resid = res.resid

# 1. Residual ACF
fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(resid.dropna(), lags=48, ax=ax)
ax.set_title("ACF of SARIMAX Residuals")
fig.tight_layout()
fig.savefig("outputs/figures/06_sarimax_residual_acf.png", dpi=120)
plt.close(fig)

# 2. Residual distribution (histogram + normal overlay)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(resid, bins=50, density=True, alpha=0.7, color="steelblue")
mu, sigma = resid.mean(), resid.std()
x = np.linspace(resid.min(), resid.max(), 200)
from scipy.stats import norm
axes[0].plot(x, norm.pdf(x, mu, sigma), color="red", lw=2, label="Normal fit")
axes[0].set_title("Residual Distribution")
axes[0].legend()

from statsmodels.graphics.gofplots import qqplot
qqplot(resid.dropna(), line="s", ax=axes[1])
axes[1].set_title("Residual Q-Q Plot")
fig.tight_layout()
fig.savefig("outputs/figures/07_sarimax_residual_dist.png", dpi=120)
plt.close(fig)

# 3. Forecast plot with confidence intervals
forecast_df = pd.read_csv("outputs/sarimax_forecast_24h.csv", index_col=0, parse_dates=True)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(forecast_df.index, forecast_df["actual"], label="Actual", color="black", lw=2)
ax.plot(forecast_df.index, forecast_df["forecast"], label="SARIMAX Forecast", color="tab:blue", lw=1.5)
ax.fill_between(forecast_df.index, forecast_df["lower_ci"], forecast_df["upper_ci"],
                 color="tab:blue", alpha=0.2, label="95% CI")
ax.set_title("SARIMAX(1,0,6)x(1,1,1,24) — 24h Forecast with 95% CI")
ax.set_xlabel("Time"); ax.set_ylabel("Appliances (Wh)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("outputs/figures/08_sarimax_forecast_ci.png", dpi=120)
plt.close(fig)

print("Saved diagnostic and forecast plots to outputs/figures/")
