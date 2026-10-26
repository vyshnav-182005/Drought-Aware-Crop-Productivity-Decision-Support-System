"""
Master Pipeline Runner for Crop Yield Prediction
Owner: Guhan K B

Runs the complete yield prediction workflow in order:
  01_prepare_crop_data.py          DES ingestion, name standardisation, long table
  02_build_features.py             Stage-wise weather features + SPI (weather data new, 2000-2017 baselines)
  03_train_yield_models.py         Baseline / ElasticNet / RF / XGBoost / HGB, chosen on validation, MLflow, SHAP
  04_in_season_forecast.py         LSTM 3-day forecast + analog years, backtest 2020-2022, impact handoff
  05_hurdle_yield_prediction.py    Leak-free hurdle forecast backtest + comparison with LSTM / no forecast / oracle
  06_train_hist_gradient_boosting_yield.py  HistGradientBoosting benchmark (optional)
  07_rolling_evaluation.py         Rolling-origin R² 2010-2022 (each year predicted from earlier years)

Usage: python run_yield_pipeline.py [--skip-optional]
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    ("Stage 1 - Crop data ingestion & standardisation", "01_prepare_crop_data.py", False),
    ("Stage 2 - Agro-meteorological features & SPI", "02_build_features.py", False),
    ("Stage 3 - Yield models (validation selection), MLflow & SHAP", "03_train_yield_models.py", False),
    ("Stage 4 - LSTM in-season forecasting, backtest & handoff", "04_in_season_forecast.py", False),
    ("Stage 5 - Hurdle forecast integration & forecaster comparison", "05_hurdle_yield_prediction.py", False),
    ("Stage 6 - HistGradientBoosting benchmark", "06_train_hist_gradient_boosting_yield.py", True),
    ("Stage 7 - Rolling-origin evaluation (R² 2010-2022)", "07_rolling_evaluation.py", False),
]


def run_pipeline(skip_optional: bool = False):
    print("=" * 70)
    print("DROUGHT-AWARE CROP PRODUCTIVITY DECISION SUPPORT SYSTEM")
    print("Crop Yield Prediction Pipeline Runner")
    print("=" * 70)

    for desc, script_name, optional in SCRIPTS:
        if optional and skip_optional:
            continue
        print(f"\n>>> Running: {desc}\n    File: {script_name}\n" + "-" * 70)
        # check=True raises CalledProcessError (non-zero exit) if a stage fails
        subprocess.run([sys.executable, str(BASE_DIR / script_name)], check=True, cwd=BASE_DIR)

    print("\n" + "=" * 70)
    print("SUCCESS: crop yield prediction pipeline finished.")
    print("Artifacts:")
    print("  - Data:    data/crop_production/cleaned_crop_yield.csv")
    print("  - Tables:  outputs/yield_prediction/model_table.csv, season_weather_features.csv")
    print("  - Models:  models/yield_models/*_best.joblib, models/hurdle_yield/*.joblib")
    print("  - Results: outputs/yield_prediction/crop_yield_model_benchmark_summary.csv, forecaster_comparison.csv, rolling_evaluation.csv")
    print("  - SHAP:    outputs/yield_prediction/shap_*")
    print("  - Plots:   outputs/yield_prediction/plots/")
    print("  - Handoff: outputs/yield_prediction/yield_impact_handoff.csv")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline(skip_optional="--skip-optional" in sys.argv)
