"""
Part 1: Data loading, cleaning, and resampling to hourly frequency.
"""
import pandas as pd
import numpy as np

RAW_PATH = "data/energydata_complete.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw 10-minute Appliance Energy Prediction dataset."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def check_missing(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column and check for time gaps."""
    missing = df.isna().sum()
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    gaps = full_range.difference(df.index)
    print(f"Missing values per column (top 5):\n{missing.sort_values(ascending=False).head()}")
    print(f"\nExpected {len(full_range)} timestamps at 10-min freq, "
          f"got {len(df)} -> {len(gaps)} missing timestamps (gaps).")
    return missing


def resample_hourly(df: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    """
    Resample 10-minute data to hourly.
    Appliances/lights (energy, Wh per 10 min) are summed to get Wh per hour.
    Sensor/weather columns are averaged.
    """
    df = df.copy()
    energy_cols = ["Appliances", "lights"]
    sensor_cols = [c for c in df.columns if c not in energy_cols]

    hourly_energy = df[energy_cols].resample("1h").sum()
    hourly_sensors = df[sensor_cols].resample("1h").mean()

    hourly = pd.concat([hourly_energy, hourly_sensors], axis=1)
    hourly = hourly.dropna(how="all")
    return hourly


if __name__ == "__main__":
    df = load_raw()
    print(df.shape)
    print(df.head())
    check_missing(df)
    hourly = resample_hourly(df)
    print("\nHourly shape:", hourly.shape)
    hourly.to_csv("data/energy_hourly.csv")
    print("Saved hourly data to data/energy_hourly.csv")
