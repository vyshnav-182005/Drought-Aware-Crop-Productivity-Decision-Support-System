"""
Model selection used by 03 and 06.

Protocol (no test-set peeking):
  1. Fit every candidate on TRAIN_YEARS.
  2. Score on VAL_YEARS and pick the candidate with the lowest validation RMSE.
  3. Refit candidates on TRAIN + VAL years.
  4. Score on TEST_YEARS once. Test scores are reported, never used for choosing.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from common import TEST_YEARS, TRAIN_YEARS, VAL_YEARS, feature_matrix, in_years


def make_pipeline(estimator, scale: bool = False) -> Pipeline:
    """Train-median imputation (+ optional scaling) bundled with the estimator, so nothing is fit outside train."""
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas"))]
    if scale:
        from sklearn.preprocessing import StandardScaler
        steps.append(("scale", StandardScaler().set_output(transform="pandas")))
    steps.append(("model", estimator))
    return Pipeline(steps)


def evaluate_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    valid = y_true > 1e-4
    mape = float(mean_absolute_percentage_error(y_true[valid], y_pred[valid]) * 100.0) if valid.any() else np.nan
    return rmse, mae, mape


def _fit_predict(pipe, transform, X_fit, y_fit, X_pred):
    if transform == "log1p":
        pipe.fit(X_fit, np.log1p(y_fit))
        return np.clip(np.expm1(pipe.predict(X_pred)), 0.0, None)
    pipe.fit(X_fit, y_fit)
    return np.clip(pipe.predict(X_pred), 0.0, None)


def split(df_sub: pd.DataFrame):
    return (df_sub[in_years(df_sub["year"], TRAIN_YEARS)],
            df_sub[in_years(df_sub["year"], VAL_YEARS)],
            df_sub[in_years(df_sub["year"], TEST_YEARS)])


def select_and_evaluate(crop: str, season: str, df_sub: pd.DataFrame, candidates: dict, log=None):
    """
    candidates: {name: (unfitted Pipeline, target_transform or None)}
    log: optional callback(name, metrics_dict) e.g. for MLflow.
    Returns (results DataFrame, bundle for the selected model, test frame, test predictions of selected).
    """
    df_train, df_val, df_test = split(df_sub)
    if len(df_train) == 0 or len(df_val) == 0 or len(df_test) == 0:
        print(f"Warning: insufficient data for {crop} ({season}); skipping.")
        return None

    feature_cols = list(feature_matrix(df_train).columns)
    X_tr, X_va, X_te = (feature_matrix(d, feature_cols) for d in (df_train, df_val, df_test))
    df_trval = pd.concat([df_train, df_val])
    X_trval = feature_matrix(df_trval, feature_cols)
    y_tr, y_va, y_te, y_trval = (d["yield"].to_numpy() for d in (df_train, df_val, df_test, df_trval))

    drought = (df_test["seasonal_spi"] < -1.0).to_numpy()
    drought_note = f"{drought.sum()} test rows with SPI < -1"
    if not drought.any():
        drought = (df_test["seasonal_spi"] == df_test["seasonal_spi"].min()).to_numpy()
        drought_note = f"no SPI < -1 in test; using driest test season (SPI {df_test['seasonal_spi'].min():.2f})"

    rows, refit = [], {}
    for name, (pipe, transform) in candidates.items():
        # Steps 1-2: train-only fit, validation score
        val_pred = _fit_predict(clone(pipe), transform, X_tr, y_tr, X_va)
        val_rmse, val_mae, val_mape = evaluate_metrics(y_va, val_pred)
        # Steps 3-4: refit on train+val, score test once
        final = clone(pipe)
        test_pred = _fit_predict(final, transform, X_trval, y_trval, X_te)
        test_rmse, test_mae, test_mape = evaluate_metrics(y_te, test_pred)
        dr_rmse, dr_mae, _ = evaluate_metrics(y_te[drought], test_pred[drought])
        refit[name] = (final, transform, test_pred)
        row = {"crop": crop, "season": season, "model": name,
               "val_rmse": val_rmse, "val_mae": val_mae, "val_mape": val_mape,
               "test_rmse": test_rmse, "test_mae": test_mae, "test_mape": test_mape,
               "test_r2": float(r2_score(y_te, test_pred)),
               "drought_rmse": dr_rmse, "drought_mae": dr_mae}
        rows.append(row)
        if log:
            log(name, row)

    results = pd.DataFrame(rows)
    best_name = results.loc[results["val_rmse"].idxmin(), "model"]
    results["selected"] = results["model"] == best_name
    final, transform, best_test_pred = refit[best_name]

    print(f"\n{crop} ({season}): train {len(df_train)} | val {len(df_val)} | test {len(df_test)} rows; {drought_note}")
    print(results[["model", "val_rmse", "test_rmse", "test_mae", "test_r2", "drought_rmse", "selected"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f">> Selected on validation RMSE: {best_name}")

    bundle = {
        "crop": crop, "season": season, "model_name": best_name, "model": final,
        "target_transform": transform, "feature_cols": feature_cols,
        "trained_on_years": [int(df_trval["year"].min()), int(df_trval["year"].max())],
        "selection": "lowest validation RMSE (2018-2019), refit on 2000-2019",
        "metrics": results[results["selected"]].iloc[0].to_dict(),
    }
    return results, bundle, df_test, best_test_pred, refit


def shap_for_tree(pipe: Pipeline, X: pd.DataFrame):
    """SHAP values for a tree model at the end of an impute(/scale) pipeline."""
    import shap
    X_t = pipe[:-1].transform(X)
    explainer = shap.TreeExplainer(pipe[-1])
    return np.asarray(explainer.shap_values(X_t)), list(X_t.columns)
