"""
Part 6: Feature-based ML model (XGBoost).

Trains on the Part 5 feature matrix and produces true 24h-ahead forecasts
via a recursive strategy: short lags (1,2,3h) and rolling stats are updated
step-by-step using the model's own predictions within each 24h window,
while seasonal lags (24,48,168h) always reference actual history (since the
horizon, 24h, never exceeds them). Weather/indoor-sensor covariates and time
features are taken as known in advance for the forecast window (a standard
simplifying assumption -- see notebook discussion) so results are
comparable with SARIMAX/benchmarks.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from problem_def import TARGET, HORIZON, TEST_DAYS, evaluate

FEATURE_EXCLUDE = ["Appliances", "lights"]


def load_features(path: str = "data/energy_features.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def split_features(feat: pd.DataFrame, test_days: int = TEST_DAYS):
    test_start = feat.index.max() - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
    train = feat.loc[:test_start - pd.Timedelta(hours=1)]
    test = feat.loc[test_start:]
    return train, test


def train_model(train: pd.DataFrame, feature_cols) -> XGBRegressor:
    X_train = train[feature_cols]
    y_train = train[TARGET]
    model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def recursive_forecast_window(model, feat: pd.DataFrame, feature_cols,
                               window_times: pd.DatetimeIndex,
                               actual_series: pd.Series) -> np.ndarray:
    """Forecast one 24h window recursively, updating short lags/rolling stats
    with the model's own predictions; seasonal lags use actual history."""
    temp_series = actual_series.astype(float).copy()
    preds = []
    for t in window_times:
        row = feat.loc[t, feature_cols].copy()
        row["lag_1"] = temp_series.loc[t - pd.Timedelta(hours=1)]
        row["lag_2"] = temp_series.loc[t - pd.Timedelta(hours=2)]
        row["lag_3"] = temp_series.loc[t - pd.Timedelta(hours=3)]
        row["lag_24"] = temp_series.loc[t - pd.Timedelta(hours=24)]
        row["lag_48"] = temp_series.loc[t - pd.Timedelta(hours=48)]
        row["lag_168"] = temp_series.loc[t - pd.Timedelta(hours=168)]

        window_24 = temp_series.loc[t - pd.Timedelta(hours=24): t - pd.Timedelta(hours=1)]
        row["roll_mean_24"] = window_24.mean()
        row["roll_std_24"] = window_24.std()
        window_168 = temp_series.loc[t - pd.Timedelta(hours=168): t - pd.Timedelta(hours=1)]
        row["roll_mean_168"] = window_168.mean()

        X_row = row.values.reshape(1, -1)
        pred = float(model.predict(X_row)[0])
        preds.append(pred)
        temp_series.loc[t] = pred
    return np.array(preds)


def rolling_evaluate_ml(model, feat: pd.DataFrame, feature_cols,
                         test: pd.DataFrame, horizon: int = HORIZON):
    actual_series = feat[TARGET]
    n_windows = len(test) // horizon
    scores, all_preds = [], []
    for w in range(n_windows):
        window_times = test.index[w * horizon:(w + 1) * horizon]
        y_true = test[TARGET].iloc[w * horizon:(w + 1) * horizon].values
        y_pred = recursive_forecast_window(model, feat, feature_cols, window_times, actual_series)
        scores.append(evaluate(y_true, y_pred))
        all_preds.append(y_pred)
    return pd.DataFrame(scores), np.concatenate(all_preds)


if __name__ == "__main__":
    feat = load_features()
    train, test = split_features(feat)
    feature_cols = [c for c in feat.columns if c not in FEATURE_EXCLUDE]

    print(f"Training XGBoost on {len(train)} rows, {len(feature_cols)} features...")
    model = train_model(train, feature_cols)

    print("Running rolling 24h recursive evaluation...")
    rolling_df, all_preds = rolling_evaluate_ml(model, feat, feature_cols, test)
    print("\nMean rolling metrics:\n", rolling_df.mean())

    rolling_df.to_csv("outputs/ml_rolling_per_window.csv", index=False)
    rolling_df.mean().to_csv("outputs/ml_rolling_mean.csv")

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 feature importances:\n", importances.head(10))
    importances.to_csv("outputs/ml_feature_importances.csv")
