"""
Part 1 (continued): EDA plots, seasonal decomposition, and stationarity tests
for the hourly Appliances energy series.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

FIG_DIR = "outputs/figures"


def plot_full_series(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["Appliances"], lw=0.6)
    ax.set_title("Hourly Appliance Energy Use (Wh) — Full Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy (Wh)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_full_series.png", dpi=120)
    plt.close(fig)


def plot_daily_weekly_pattern(df: pd.DataFrame):
    hourly_avg = df.groupby(df.index.hour)["Appliances"].mean()
    dow_avg = df.groupby(df.index.dayofweek)["Appliances"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hourly_avg.index, hourly_avg.values, marker="o")
    axes[0].set_title("Mean Appliance Use by Hour of Day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Mean Energy (Wh)")

    axes[1].plot(dow_avg.index, dow_avg.values, marker="o", color="darkorange")
    axes[1].set_title("Mean Appliance Use by Day of Week (0=Mon)")
    axes[1].set_xlabel("Day of week")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_daily_weekly_pattern.png", dpi=120)
    plt.close(fig)


def plot_decomposition(df: pd.DataFrame, period: int = 24):
    """Additive decomposition using a daily period (24 hourly obs/day)."""
    result = seasonal_decompose(df["Appliances"], model="additive", period=period)
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_decomposition_daily.png", dpi=120)
    plt.close(fig)
    return result


def plot_acf_pacf(series: pd.Series, lags: int = 72, name: str = "levels"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF — {name}")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF — {name}")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/04_acf_pacf_{name}.png", dpi=120)
    plt.close(fig)


def run_adf(series: pd.Series, label: str):
    result = adfuller(series.dropna(), autolag="AIC")
    print(f"\nADF test — {label}")
    print(f"  ADF statistic: {result[0]:.4f}")
    print(f"  p-value:       {result[1]:.4f}")
    print(f"  # lags used:   {result[2]}")
    for k, v in result[4].items():
        print(f"  Critical value ({k}): {v:.4f}")
    conclusion = "STATIONARY (reject H0)" if result[1] < 0.05 else "NON-STATIONARY (fail to reject H0)"
    print(f"  => {conclusion}")
    return result


def run_kpss(series: pd.Series, label: str):
    result = kpss(series.dropna(), regression="c", nlags="auto")
    print(f"\nKPSS test — {label}")
    print(f"  KPSS statistic: {result[0]:.4f}")
    print(f"  p-value:        {result[1]:.4f}")
    for k, v in result[3].items():
        print(f"  Critical value ({k}): {v:.4f}")
    conclusion = "NON-STATIONARY (reject H0)" if result[1] < 0.05 else "STATIONARY (fail to reject H0)"
    print(f"  => {conclusion}")
    return result


if __name__ == "__main__":
    df = pd.read_csv("data/energy_hourly.csv", index_col="date", parse_dates=True)

    plot_full_series(df)
    plot_daily_weekly_pattern(df)
    decomp = plot_decomposition(df, period=24)
    plot_acf_pacf(df["Appliances"], lags=72, name="levels")

    print("=" * 60)
    print("STATIONARITY TESTS ON LEVELS")
    print("=" * 60)
    run_adf(df["Appliances"], "Appliances (levels)")
    run_kpss(df["Appliances"], "Appliances (levels)")

    # First difference
    diff1 = df["Appliances"].diff().dropna()
    print("\n" + "=" * 60)
    print("STATIONARITY TESTS ON FIRST DIFFERENCE")
    print("=" * 60)
    run_adf(diff1, "Appliances (1st diff)")
    run_kpss(diff1, "Appliances (1st diff)")
    plot_acf_pacf(diff1, lags=72, name="first_diff")

    # Seasonal (24h) difference
    sdiff = df["Appliances"].diff(24).dropna()
    print("\n" + "=" * 60)
    print("STATIONARITY TESTS ON SEASONAL (24h) DIFFERENCE")
    print("=" * 60)
    run_adf(sdiff, "Appliances (seasonal diff, 24h)")
    run_kpss(sdiff, "Appliances (seasonal diff, 24h)")
    plot_acf_pacf(sdiff, lags=72, name="seasonal_diff24")

    print("\nAll figures saved to outputs/figures/")
