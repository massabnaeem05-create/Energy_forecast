"""
run_pipeline.py
================
End-to-end pipeline for the Appliance Energy Forecasting project.

Runs every stage in sequence and reproduces every number/figure used in the
report:

  1. Data prep (download + hourly resampling)
  2. EDA + stationarity tests
  3. Problem definition sanity check
  4. Benchmark models (rolling evaluation)
  5. SARIMAX (AIC grid search -> final fit -> diagnostics -> rolling eval)
  6. Feature engineering
  7. ML model (XGBoost, recursive vs direct strategy comparison)
  8. Foundation model (Chronos, zero-shot) -- requires internet access to
     HuggingFace Hub to download pretrained weights; skipped automatically
     with a warning if unavailable (e.g. in a network-restricted sandbox).
  9. Final consolidated comparison table + plots

Usage:
    python run_pipeline.py                 # run everything
    python run_pipeline.py --skip-chronos  # skip the foundation model stage
    python run_pipeline.py --skip-sarimax-search  # use cached SARIMAX order,
                                                    skip the ~15-20 min grid search

All intermediate and final artifacts are written to data/ and outputs/,
matching exactly what the accompanying notebook produces.
"""
import argparse
import sys
import warnings
import time
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np


def log(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


def main(skip_chronos: bool, skip_sarimax_search: bool):
    Path("data").mkdir(exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ---------------------------------------------------------------
    # 1. DATA PREP
    # ---------------------------------------------------------------
    log("STAGE 1/9: Data preparation")
    import data_prep
    if not Path("data/energydata_complete.csv").exists():
        raise FileNotFoundError(
            "data/energydata_complete.csv not found. Download it first, e.g.:\n"
            "  curl -L -o data/energydata_complete.csv "
            "https://raw.githubusercontent.com/LuisM78/Appliances-energy-prediction-data/master/energydata_complete.csv"
        )
    df_raw = data_prep.load_raw()
    data_prep.check_missing(df_raw)
    df_hourly = data_prep.resample_hourly(df_raw)
    df_hourly.to_csv("data/energy_hourly.csv")
    print(f"Hourly data: {df_hourly.shape}")

    # ---------------------------------------------------------------
    # 2. EDA + STATIONARITY
    # ---------------------------------------------------------------
    log("STAGE 2/9: EDA and stationarity tests")
    import eda_stationarity as eda
    df = pd.read_csv("data/energy_hourly.csv", index_col="date", parse_dates=True)
    eda.plot_full_series(df)
    eda.plot_daily_weekly_pattern(df)
    eda.plot_decomposition(df, period=24)
    eda.plot_acf_pacf(df["Appliances"], lags=72, name="levels")
    eda.run_adf(df["Appliances"], "levels")
    eda.run_kpss(df["Appliances"], "levels")

    # ---------------------------------------------------------------
    # 3. PROBLEM DEFINITION
    # ---------------------------------------------------------------
    log("STAGE 3/9: Problem definition")
    import problem_def as pdef
    train, test = pdef.train_test_split(df)
    print(f"Train: {train.index.min()} -> {train.index.max()} ({len(train)} hrs)")
    print(f"Test:  {test.index.min()} -> {test.index.max()} ({len(test)} hrs)")

    # ---------------------------------------------------------------
    # 4. BENCHMARKS
    # ---------------------------------------------------------------
    log("STAGE 4/9: Benchmark models")
    import benchmarks as bm
    first_window_df, rolling_df = bm.run_all_benchmarks(df)
    rolling_df.to_csv("outputs/benchmark_rolling_results.csv")
    print(rolling_df)

    # ---------------------------------------------------------------
    # 5. SARIMAX
    # ---------------------------------------------------------------
    log("STAGE 5/9: SARIMAX")
    import sarimax_model as sm
    if skip_sarimax_search:
        print("Skipping AIC grid search (--skip-sarimax-search); "
              "using cached best order (1,0,6).")
        best_order = (1, 0, 6)
    else:
        print("Running AIC grid search over 147 combinations "
              "(this takes ~15-20 minutes)...")
        grid_results = sm.grid_search_orders(train["Appliances"])
        grid_results.to_csv("outputs/sarimax_grid_search.csv", index=False)
        best = grid_results.iloc[0]
        best_order = (int(best.p), int(best.d), int(best.q))
        print(f"Best order: {best_order}, AIC={best.AIC:.1f}")

    final_res = sm.fit_final_model(train["Appliances"], best_order)
    print("Converged:", final_res.mle_retvals.get("converged"))

    # Rolling evaluation via Kalman-filter state updates (no refit needed per window)
    from problem_def import TARGET, HORIZON, evaluate
    n_windows = len(test) // HORIZON
    scores, current_res = [], final_res
    for w in range(n_windows):
        fc = current_res.get_forecast(steps=HORIZON)
        y_pred = fc.predicted_mean.values
        y_true = test[TARGET].iloc[w * HORIZON:(w + 1) * HORIZON].values
        scores.append(evaluate(y_true, y_pred))
        new_obs = test[TARGET].iloc[w * HORIZON:(w + 1) * HORIZON]
        current_res = current_res.append(new_obs, refit=False)
    sarimax_rolling_df = pd.DataFrame(scores)
    sarimax_rolling_df.to_csv("outputs/sarimax_rolling_per_window.csv", index=False)
    sarimax_rolling_df.mean().to_csv("outputs/sarimax_rolling_mean.csv")
    print("SARIMAX rolling mean:\n", sarimax_rolling_df.mean())

    # ---------------------------------------------------------------
    # 6. FEATURE ENGINEERING
    # ---------------------------------------------------------------
    log("STAGE 6/9: Feature engineering")
    import feature_engineering as fe
    print(fe.covariate_correlations(df))
    feat = fe.build_feature_matrix(df)
    feat.to_csv("data/energy_features.csv")
    print(f"Feature matrix: {feat.shape}")

    # ---------------------------------------------------------------
    # 7. ML MODEL (direct multi-horizon, the adopted strategy)
    # ---------------------------------------------------------------
    log("STAGE 7/9: Feature-based ML model (XGBoost, direct multi-horizon)")
    import ml_model_direct as mld
    feat = mld.load_features()
    ftrain, ftest = mld.split_features(feat)
    feature_cols = [c for c in feat.columns if c not in mld.FEATURE_EXCLUDE]
    models = mld.train_direct_models(feat, ftrain.index.max(), feature_cols)
    ml_rolling_df, _ = mld.rolling_evaluate_direct(models, feat, feature_cols, ftest)
    ml_rolling_df.to_csv("outputs/ml_direct_rolling_per_window.csv", index=False)
    ml_rolling_df.mean().to_csv("outputs/ml_direct_rolling_mean.csv")
    print("XGBoost (direct) rolling mean:\n", ml_rolling_df.mean())

    # ---------------------------------------------------------------
    # 8. FOUNDATION MODEL (Chronos, zero-shot)
    # ---------------------------------------------------------------
    log("STAGE 8/9: Foundation model (Chronos, zero-shot)")
    if skip_chronos:
        print("Skipping Chronos stage (--skip-chronos).")
    else:
        try:
            import chronos_model as cm
            pipeline = cm.load_pipeline()
            chronos_rolling_df = cm.rolling_evaluate_chronos(pipeline, df["Appliances"], test)
            chronos_rolling_df.to_csv("outputs/chronos_rolling_per_window.csv", index=False)
            chronos_rolling_df.mean().to_csv("outputs/chronos_rolling_mean.csv")
            print("Chronos rolling mean:\n", chronos_rolling_df.mean())
        except Exception as e:
            print(f"WARNING: Chronos stage failed/skipped ({type(e).__name__}: {e}).\n"
                  "This is expected in network-restricted environments (needs "
                  "HuggingFace Hub access to download weights). Run with internet "
                  "access (e.g. Colab) to include this stage, or rerun with "
                  "--skip-chronos to suppress this warning.")

    # ---------------------------------------------------------------
    # 9. FINAL CONSOLIDATED COMPARISON
    # ---------------------------------------------------------------
    log("STAGE 9/9: Final consolidated comparison")
    parts = [rolling_df]
    sarimax_mean = pd.read_csv("outputs/sarimax_rolling_mean.csv", index_col=0).iloc[:, 0]
    sarimax_mean.name = f"SARIMA{best_order}x(1,1,1,24)"
    parts.append(sarimax_mean.to_frame().T)

    ml_mean = pd.read_csv("outputs/ml_direct_rolling_mean.csv", index_col=0).iloc[:, 0]
    ml_mean.name = "XGBoost (direct multi-horizon)"
    parts.append(ml_mean.to_frame().T)

    chronos_path = Path("outputs/chronos_rolling_mean.csv")
    if chronos_path.exists():
        chronos_mean = pd.read_csv(chronos_path, index_col=0).iloc[:, 0]
        chronos_mean.name = "Chronos-Bolt-Small (zero-shot)"
        parts.append(chronos_mean.to_frame().T)
    else:
        print("Chronos results not found -- omitted from final table "
              "(run without --skip-chronos and with internet access to include it).")

    final_comparison = pd.concat(parts).sort_values("RMSE")
    final_comparison.to_csv("outputs/final_comparison_rolling.csv")
    print("\nFINAL RESULTS (rolling 14-window average, sorted by RMSE):")
    print(final_comparison)

    elapsed = time.time() - t_start
    log(f"Pipeline complete in {elapsed/60:.1f} minutes. "
        f"See outputs/final_comparison_rolling.csv and outputs/figures/ for all results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full appliance energy forecasting pipeline.")
    parser.add_argument("--skip-chronos", action="store_true",
                         help="Skip the Chronos foundation model stage (e.g. no internet access).")
    parser.add_argument("--skip-sarimax-search", action="store_true",
                         help="Skip the ~15-20 min SARIMAX AIC grid search and use the cached best order (1,0,6).")
    args = parser.parse_args()
    main(skip_chronos=args.skip_chronos, skip_sarimax_search=args.skip_sarimax_search)
