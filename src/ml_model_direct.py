"""
Part 6 (alternative strategy): direct multi-horizon XGBoost.

Instead of recursively feeding predictions back in as lag features (which
compounds error over a 24-step horizon), we train 24 separate models, one
per horizon step h=1..24, each predicting y[t+h] directly from features
known at time t. This avoids recursive error accumulation entirely and is
a standard, often stronger, alternative for tree-based multi-step
forecasting.
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


def train_direct_models(feat: pd.DataFrame, train_end: pd.Timestamp, feature_cols,
                         horizon: int = HORIZON):
    """Train one model per horizon step, using only rows whose target
    (t+h) still falls within the training period."""
    models = {}
    for h in range(1, horizon + 1):
        y_shifted = feat[TARGET].shift(-h)
        X = feat[feature_cols]
        valid = X.index <= (train_end - pd.Timedelta(hours=h))
        X_h, y_h = X.loc[valid], y_shifted.loc[valid].dropna()
        X_h = X_h.loc[y_h.index]

        model = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        model.fit(X_h, y_h)
        models[h] = model
    return models


def rolling_evaluate_direct(models, feat: pd.DataFrame, feature_cols,
                             test: pd.DataFrame, horizon: int = HORIZON):
    n_windows = len(test) // horizon
    scores, all_preds = [], []
    for w in range(n_windows):
        origin_time = test.index[w * horizon] - pd.Timedelta(hours=1)
        X_origin = feat.loc[[origin_time], feature_cols]
        y_pred = np.array([models[h].predict(X_origin)[0] for h in range(1, horizon + 1)])
        y_true = test[TARGET].iloc[w * horizon:(w + 1) * horizon].values
        scores.append(evaluate(y_true, y_pred))
        all_preds.append(y_pred)
    return pd.DataFrame(scores), np.concatenate(all_preds)


if __name__ == "__main__":
    feat = load_features()
    train, test = split_features(feat)
    feature_cols = [c for c in feat.columns if c not in FEATURE_EXCLUDE]
    train_end = train.index.max()

    print(f"Training {HORIZON} direct-horizon XGBoost models...")
    models = train_direct_models(feat, train_end, feature_cols)

    print("Running rolling 24h direct-forecast evaluation...")
    rolling_df, all_preds = rolling_evaluate_direct(models, feat, feature_cols, test)
    print("\nMean rolling metrics (direct strategy):\n", rolling_df.mean())

    rolling_df.to_csv("outputs/ml_direct_rolling_per_window.csv", index=False)
    rolling_df.mean().to_csv("outputs/ml_direct_rolling_mean.csv")

    importances = pd.Series(models[1].feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 feature importances (h=1 model):\n", importances.head(10))
