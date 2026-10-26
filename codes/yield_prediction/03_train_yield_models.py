"""
Stage 3: Crop Yield Modeling, MLflow Tracking, and SHAP Explainability
Tasks 7, 8, 9, 10:
- Temporal split: Train 2000-2017 | Val 2018-2019 | Test 2020-2022
- One model per crop-season, pooled across the 5 districts with district dummies.
- Candidates: district-mean/trend baseline, ElasticNet, Random Forest, XGBoost, HistGradientBoosting.
- Selection: lowest VALIDATION RMSE. The selected model is refit on 2000-2019 and the
  test years are scored once (RMSE, MAE, MAPE, and RMSE on drought test seasons, SPI < -1).
- Imputation/scaling live inside each model pipeline, so they are fit on training rows only.
- SHAP TreeExplainer on the selected model (or on the best tree model by validation
  RMSE if a linear/baseline model was selected).
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet

from baseline_model import BaselineModel
from common import MODEL_TABLE_FILE, OUTPUT_DIR, YIELD_MODELS_DIR
from modeling import make_pipeline, select_and_evaluate, shap_for_tree

warnings.filterwarnings("ignore")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

TREE_MODELS = {"RandomForest", "XGBoost", "HistGradientBoosting"}


def candidates():
    return {
        "Baseline": (make_pipeline(BaselineModel()), None),
        "ElasticNet": (make_pipeline(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42), scale=True), None),
        "RandomForest": (make_pipeline(RandomForestRegressor(
            n_estimators=200, max_depth=3, min_samples_leaf=3, max_features="sqrt", random_state=42)), None),
        "XGBoost": (make_pipeline(xgb.XGBRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.04, subsample=0.75, colsample_bytree=0.75,
            reg_alpha=2.0, reg_lambda=5.0, random_state=42)), None),
        "HistGradientBoosting": (make_pipeline(HistGradientBoostingRegressor(
            max_iter=50, max_leaf_nodes=8, min_samples_leaf=5, learning_rate=0.03,
            l2_regularization=3.0, random_state=42)), None),
    }


def get_mlflow_logger(crop, season):
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{(OUTPUT_DIR / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("Crop_Yield_Prediction")
    except Exception as exc:  # MLflow is optional for the pipeline to run
        print(f"MLflow unavailable ({exc}); continuing without tracking.")
        return None

    def log(name, row):
        with mlflow.start_run(run_name=f"{crop}_{season}_{name}"):
            mlflow.log_params({"crop": crop, "season": season, "model_name": name,
                               "selection": "val_rmse", "final_fit_years": "2000-2019"})
            mlflow.log_metrics({k: float(v) for k, v in row.items()
                                if isinstance(v, (int, float, np.floating)) and np.isfinite(v)})
    return log


def explain(crop, season, results, bundle, refit, df_test, test_pred):
    tag = f"{crop.replace('/', '_')}_{season}"
    tree_rows = results[results["model"].isin(TREE_MODELS)].sort_values("val_rmse")
    explain_name = bundle["model_name"] if bundle["model_name"] in TREE_MODELS else tree_rows.iloc[0]["model"]
    pipe = refit[explain_name][0]
    try:
        from common import feature_matrix
        X_test = feature_matrix(df_test, bundle["feature_cols"])
        shap_values, names = shap_for_tree(pipe, X_test)
        importance = (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
                      .sort_values("mean_abs_shap", ascending=False))
        importance.insert(0, "explained_model", explain_name)
        importance.to_csv(OUTPUT_DIR / f"shap_{tag}_importance.csv", index=False)

        breakdown = df_test[["district", "year", "yield"]].copy()
        breakdown["predicted_yield"] = refit[explain_name][2]
        for feat in importance["feature"].head(3):
            breakdown[f"shap_{feat}"] = shap_values[:, names.index(feat)]
        breakdown.to_csv(OUTPUT_DIR / f"shap_{tag}_test_breakdown.csv", index=False)
        note = "" if explain_name == bundle["model_name"] else f" (selected model is {bundle['model_name']}; explained best tree model)"
        print(f"SHAP saved for {explain_name}{note}")
    except Exception as exc:
        print(f"Warning: SHAP failed for {crop} ({season}): {exc}")


def main():
    df = pd.read_csv(MODEL_TABLE_FILE)
    print(f"Loaded {len(df)} rows from {MODEL_TABLE_FILE}")

    summary, all_results = [], []
    for (crop, season), sub in df.groupby(["crop", "season"]):
        out = select_and_evaluate(crop, season, sub, candidates(), log=get_mlflow_logger(crop, season))
        if out is None:
            continue
        results, bundle, df_test, test_pred, refit = out
        all_results.append(results)

        path = YIELD_MODELS_DIR / f"{crop.replace('/', '_')}_{season}_best.joblib"
        joblib.dump(bundle, path)
        explain(crop, season, results, bundle, refit, df_test, test_pred)

        m = bundle["metrics"]
        summary.append({"Crop": crop, "Season": season, "Selected Model": bundle["model_name"],
                        "Val RMSE": round(m["val_rmse"], 4), "Test RMSE": round(m["test_rmse"], 4),
                        "Test MAE": round(m["test_mae"], 4), "Test MAPE (%)": round(m["test_mape"], 2),
                        "Test R2": round(m["test_r2"], 3),
                        "Drought RMSE": round(m["drought_rmse"], 4)})

    pd.concat(all_results).to_csv(OUTPUT_DIR / "crop_yield_all_candidates.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "crop_yield_model_benchmark_summary.csv", index=False)
    print("\n" + "=" * 70 + "\nFINAL CROP YIELD SUMMARY (selected on validation, test scored once)\n" + "=" * 70)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
