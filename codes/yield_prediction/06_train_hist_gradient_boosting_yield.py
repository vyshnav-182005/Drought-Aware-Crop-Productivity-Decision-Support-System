"""
Crop Yield Prediction benchmark focused on HistGradientBoostingRegressor
Owner: Guhan K B

- HistGradientBoosting (raw target) and HistGradientBoosting with a log1p target, benchmarked
  against Baseline, Random Forest and XGBoost per crop-season.
- Same protocol as 03 (modeling.select_and_evaluate): choose on validation RMSE (2018-2019),
  refit on 2000-2019, score test (2020-2022) once. Test scores are never used for choosing.
- Saves the HistGradientBoosting checkpoint per crop-season (models/yield_models/*_HistGBR.joblib),
  its SHAP importance, and outputs/yield_prediction/hist_gradient_boosting_yield_benchmark.csv.
"""

import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from baseline_model import BaselineModel
from common import MODEL_TABLE_FILE, OUTPUT_DIR, YIELD_MODELS_DIR, feature_matrix
from modeling import make_pipeline, select_and_evaluate, shap_for_tree

warnings.filterwarnings("ignore")


def hgb():
    return HistGradientBoostingRegressor(loss="squared_error", max_iter=50, max_leaf_nodes=8, min_samples_leaf=5,
                                         learning_rate=0.03, l2_regularization=3.0, random_state=42)


def candidates():
    return {
        "Baseline": (make_pipeline(BaselineModel()), None),
        "RandomForest": (make_pipeline(RandomForestRegressor(
            n_estimators=200, max_depth=3, min_samples_leaf=3, max_features="sqrt", random_state=42)), None),
        "XGBoost": (make_pipeline(xgb.XGBRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.04, subsample=0.75, colsample_bytree=0.75,
            reg_alpha=2.0, reg_lambda=5.0, random_state=42)), None),
        "HistGradientBoosting": (make_pipeline(hgb()), None),
        "HistGradientBoosting_log1p": (make_pipeline(hgb()), "log1p"),
    }


def main():
    df = pd.read_csv(MODEL_TABLE_FILE)
    all_results = []
    for (crop, season), sub in df.groupby(["crop", "season"]):
        out = select_and_evaluate(crop, season, sub, candidates())
        if out is None:
            continue
        results, bundle, df_test, _, refit = out
        all_results.append(results)
        tag = f"{crop.replace('/', '_')}_{season}"

        pipe, transform, _ = refit["HistGradientBoosting"]
        joblib.dump({"crop": crop, "season": season, "model_name": "HistGradientBoosting", "model": pipe,
                     "target_transform": transform, "feature_cols": bundle["feature_cols"],
                     "trained_on_years": bundle["trained_on_years"],
                     "metrics": results[results["model"] == "HistGradientBoosting"].iloc[0].to_dict()},
                    YIELD_MODELS_DIR / f"{tag}_HistGBR.joblib")
        try:
            sv, names = shap_for_tree(pipe, feature_matrix(df_test, bundle["feature_cols"]))
            (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)})
             .sort_values("mean_abs_shap", ascending=False)
             .to_csv(OUTPUT_DIR / f"shap_{tag}_HistGBR_importance.csv", index=False))
        except Exception as exc:
            print(f"Warning: SHAP failed for {crop} ({season}): {exc}")

    bench = pd.concat(all_results)
    bench.to_csv(OUTPUT_DIR / "hist_gradient_boosting_yield_benchmark.csv", index=False)
    wins = bench[bench["selected"]]["model"].value_counts()
    print("\nModels selected on validation across crop-seasons:\n" + wins.to_string())


if __name__ == "__main__":
    main()
