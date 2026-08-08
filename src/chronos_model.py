"""
Part 7: Foundation model (Chronos) -- zero-shot 24h forecasting.

Uses Amazon's Chronos-Bolt-Small, a pretrained time-series foundation model,
with NO fine-tuning or training on this dataset (true zero-shot). Chronos-Bolt
is patch-based and fast enough for CPU inference, and outputs quantile
forecasts directly (no sampling needed), unlike the original Chronos-T5.

Note: loading the pretrained weights requires downloading from HuggingFace
Hub, so this script needs an unrestricted internet connection (works on
Colab; blocked in some sandboxed dev environments).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

from problem_def import TARGET, HORIZON, TEST_DAYS, train_test_split, evaluate

MODEL_NAME = "amazon/chronos-bolt-small"


def load_pipeline(model_name: str = MODEL_NAME):
    return BaseChronosPipeline.from_pretrained(
        model_name, device_map="cpu", torch_dtype=torch.float32,
    )


def forecast_window(pipeline, context: pd.Series, horizon: int = HORIZON):
    """Zero-shot median forecast for the next `horizon` steps."""
    context_tensor = torch.tensor(context.values, dtype=torch.float32)
    quantiles, mean = pipeline.predict_quantiles(
        context=context_tensor, prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    median = quantiles[0, :, 1].numpy()   # index 1 -> 0.5 quantile
    lower = quantiles[0, :, 0].numpy()
    upper = quantiles[0, :, 2].numpy()
    return median, lower, upper


def rolling_evaluate_chronos(pipeline, full_series: pd.Series, test: pd.DataFrame,
                              horizon: int = HORIZON):
    """Expanding-window zero-shot forecast: at each origin, the context is
    all real history up to that point (train + previously-revealed test
    windows) -- no model parameters are fit, this is pure zero-shot inference."""
    n_windows = len(test) // horizon
    scores = []
    for w in range(n_windows):
        origin_time = test.index[w * horizon] - pd.Timedelta(hours=1)
        context = full_series.loc[:origin_time]
        median, lower, upper = forecast_window(pipeline, context, horizon)
        y_true = test[TARGET].iloc[w * horizon:(w + 1) * horizon].values
        scores.append(evaluate(y_true, median))
    return pd.DataFrame(scores)


if __name__ == "__main__":
    df = pd.read_csv("data/energy_hourly.csv", index_col="date", parse_dates=True)
    train, test = train_test_split(df, TEST_DAYS)

    print(f"Loading {MODEL_NAME} (zero-shot, no fine-tuning)...")
    pipeline = load_pipeline()

    print("First-24h forecast...")
    median, lower, upper = forecast_window(pipeline, train[TARGET], HORIZON)
    y_true = test[TARGET].iloc[:HORIZON].values
    print("First-24h metrics:", evaluate(y_true, median))

    forecast_df = pd.DataFrame({
        "actual": y_true, "forecast": median, "lower": lower, "upper": upper,
    }, index=test.index[:HORIZON])
    forecast_df.to_csv("outputs/chronos_forecast_24h.csv")

    print("Rolling 14-window evaluation...")
    rolling_df = rolling_evaluate_chronos(pipeline, df[TARGET], test, HORIZON)
    rolling_df.to_csv("outputs/chronos_rolling_per_window.csv", index=False)
    rolling_df.mean().to_csv("outputs/chronos_rolling_mean.csv")
    print("Mean rolling metrics:\n", rolling_df.mean())
