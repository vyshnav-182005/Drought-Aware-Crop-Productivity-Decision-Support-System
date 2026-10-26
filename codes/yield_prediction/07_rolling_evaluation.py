"""
Rolling-origin (expanding window) evaluation: the fairer way to quote R² for this data.

For every year Y from 2010 to 2022 and every crop-season, each candidate model is trained on all
years < Y and predicts year Y. Pooling the 13 out-of-sample years gives about 65 predictions per
crop-season instead of 15, covering drought and wet years, so R² is far less dependent on three
unusual test seasons (2020-2022 were record-wet, out-of-range years).

No year is ever predicted by a model that saw it. Baseline = district mean + trend.
Output: outputs/yield_prediction/rolling_evaluation.csv (+ per-year predictions)
"""

import importlib
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score, root_mean_squared_error

from common import MODEL_TABLE_FILE, OUTPUT_DIR, feature_matrix

warnings.filterwarnings("ignore")
candidates = importlib.import_module("03_train_yield_models").candidates

YEARS = range(2010, 2023)


def main():
    mt = pd.read_csv(MODEL_TABLE_FILE)
    preds, rows = [], []
    for (crop, season), sub in mt.groupby(["crop", "season"]):
        for name, (pipe, transform) in candidates().items():
            for year in YEARS:
                train, test = sub[sub["year"] < year], sub[sub["year"] == year]
                if test.empty:
                    continue
                cols = list(feature_matrix(train).columns)
                model = clone(pipe)
                y = np.log1p(train["yield"]) if transform == "log1p" else train["yield"]
                model.fit(feature_matrix(train, cols), y)
                p = model.predict(feature_matrix(test, cols))
                p = np.clip(np.expm1(p) if transform == "log1p" else p, 0, None)
                preds.append(test[["district", "crop", "season", "year", "yield"]].assign(model=name, predicted=p))
    preds = pd.concat(preds)
    for (crop, season, name), g in preds.groupby(["crop", "season", "model"]):
        rows.append({"crop": crop, "season": season, "model": name, "n": len(g),
                     "r2": r2_score(g["yield"], g["predicted"]),
                     "rmse": root_mean_squared_error(g["yield"], g["predicted"])})
    res = pd.DataFrame(rows)
    res.to_csv(OUTPUT_DIR / "rolling_evaluation.csv", index=False)
    preds.to_csv(OUTPUT_DIR / "rolling_evaluation_predictions.csv", index=False)

    table = res.pivot_table(index=["crop", "season"], columns="model", values="r2").round(3)
    print(f"Rolling-origin R² ({YEARS.start}-{YEARS.stop - 1}, each year predicted from earlier years only):")
    print(table.to_string())
    overall = preds.groupby("model").apply(lambda g: r2_score(g["yield"], g["predicted"]), include_groups=False)
    print("\nAll crop-seasons pooled R² by model:\n" + overall.round(3).to_string())


if __name__ == "__main__":
    main()
