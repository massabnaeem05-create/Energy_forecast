"""
Part 5: Covariate correlation check + feature engineering (time encodings,
lag features, rolling statistics) producing the feature matrix used by the
Part 6 ML model.
"""
import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 24, 48, 168]
ROLL_WINDOWS = [24, 168]


def covariate_correlations(df: pd.DataFrame, target: str = "Appliances") -> pd.Series:
    sensor_cols = [c for c in df.columns if c not in [target, "lights", "rv1", "rv2"]]
    corrs = df[sensor_cols + [target]].corr()[target].drop(target)
    return corrs.sort_values(key=abs, ascending=False)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df.index.hour
    dow = df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "Appliances",
                      lags=LAGS, roll_windows=ROLL_WINDOWS) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)
    for w in roll_windows:
        df[f"roll_mean_{w}"] = df[target].shift(1).rolling(w).mean()
    df["roll_std_24"] = df[target].shift(1).rolling(24).std()
    return df


def build_feature_matrix(df: pd.DataFrame, target: str = "Appliances") -> pd.DataFrame:
    feat = add_time_features(df)
    feat = add_lag_features(feat, target)
    drop_cols = [c for c in ["rv1", "rv2"] if c in feat.columns]
    feat = feat.drop(columns=drop_cols)
    feat = feat.dropna()
    return feat


if __name__ == "__main__":
    df = pd.read_csv("data/energy_hourly.csv", index_col="date", parse_dates=True)

    print("Covariate correlations with Appliances:")
    print(covariate_correlations(df))

    feat = build_feature_matrix(df)
    print(f"\nFeature matrix shape: {feat.shape}")
    feat.to_csv("data/energy_features.csv")
    print("Saved to data/energy_features.csv")
