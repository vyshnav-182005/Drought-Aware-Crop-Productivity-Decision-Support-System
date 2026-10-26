"""
In-season yield prediction shared by 04 (LSTM) and 05 (Hurdle comparison).

At a cutoff date the season's weather is assembled as:
  observed weather up to the cutoff
  + 3-day forecast from the chosen forecaster (only uses data up to the cutoff)
  + each historical analog year (2000-2017) for the rest of the season.
Each analog trajectory is turned into the same features used in training and scored by the
crop-season yield model; the spread across analog years gives P10 / median / P90.
"""

import numpy as np
import pandas as pd

from common import (TEST_YEARS, TRAIN_YEARS, composite_weather, feature_matrix, predict_bundle,
                    raw_season_features, season_bounds)

CUTOFFS = {
    "Kharif": ["07-01", "08-01", "09-01", "10-01"],
    "Rabi": ["11-15", "12-15", "01-15", "02-15"],
}


def cutoff_date(season: str, year: int, mmdd: str) -> pd.Timestamp:
    month = int(mmdd.split("-")[0])
    y = year + (1 if season == "Rabi" and month <= 2 else 0)
    return pd.Timestamp(f"{y}-{mmdd}")


def predict_in_season(district, season, target_year, cutoff, wdf, bundle, clim, meta, forecaster=None):
    """
    meta: {"district_mean_yield": float, "prev_year_area": float}
    Returns dict(median, p10, p90, n_analogs, origin, forecast) where forecast is the 3-day forecast used.
    """
    _, end = season_bounds(season, target_year)
    origin = min(pd.Timestamp(cutoff), end, wdf.index.max())

    fc = None
    if origin < end and forecaster is not None:
        fc = forecaster.forecast(district, origin)

    analog_years = [y for y in range(TRAIN_YEARS[0], TRAIN_YEARS[1] + 1) if y != target_year]
    if origin >= end:
        analog_years = analog_years[:1]  # fully observed season: one deterministic prediction

    rows = []
    for ay in analog_years:
        comp = composite_weather(wdf, season, target_year, origin, ay, forecast=fc)
        raw = raw_season_features(comp, season, target_year)
        if raw is None:
            continue
        row = clim.complete(raw, district, season)
        row.update({"district": district, "season": season, "year": target_year,
                    "trend": target_year, **meta})
        rows.append(row)
    if not rows:
        raise ValueError(f"No complete weather trajectory for {district} {season} {target_year} at {origin.date()}")

    X = feature_matrix(pd.DataFrame(rows), bundle["feature_cols"])
    preds = predict_bundle(bundle, X)
    return {"median": float(np.median(preds)), "p10": float(np.percentile(preds, 10)),
            "p90": float(np.percentile(preds, 90)), "n_analogs": len(preds), "origin": origin,
            "forecast": fc}


def backtest(forecaster, model_table, bundles, weather, clim, crop_seasons=None):
    """Runs predict_in_season at every cutoff for every test-year record (2020-2022)."""
    test = model_table[(model_table["year"] >= TEST_YEARS[0]) & (model_table["year"] <= TEST_YEARS[1])]
    records = []
    for (crop, season), grp in test.groupby(["crop", "season"]):
        if crop_seasons and (crop, season) not in crop_seasons:
            continue
        bundle = bundles.get((crop, season))
        if bundle is None:
            continue
        for _, r in grp.iterrows():
            wdf = weather[r["district"]]
            meta = {"district_mean_yield": r["district_mean_yield"], "prev_year_area": r["prev_year_area"]}
            for mmdd in CUTOFFS[season]:
                cdate = cutoff_date(season, int(r["year"]), mmdd)
                res = predict_in_season(r["district"], season, int(r["year"]), cdate, wdf, bundle, clim, meta, forecaster)
                fc = res["forecast"]
                fc_rain = float(fc["Rainfall_mm"].sum()) if fc is not None and len(fc) else np.nan
                act_rain = float(wdf.loc[cdate + pd.Timedelta(days=1):cdate + pd.Timedelta(days=3), "Rainfall_mm"].sum())
                records.append({
                    "forecaster": forecaster.name if forecaster else "No forecast",
                    "crop": crop, "season": season, "year": int(r["year"]), "district": r["district"],
                    "cutoff": mmdd, "cutoff_date": cdate.date(), "model": bundle["model_name"],
                    "actual_yield": r["yield"], "median_pred": res["median"], "p10": res["p10"], "p90": res["p90"],
                    "abs_error": abs(res["median"] - r["yield"]),
                    "in_p10_p90": res["p10"] <= r["yield"] <= res["p90"],
                    "forecast_rain_3d_mm": fc_rain, "actual_rain_3d_mm": act_rain,
                })
    return pd.DataFrame(records)


def load_bundles(models_dir, suffix="_best.joblib"):
    import joblib
    bundles = {}
    for path in models_dir.glob(f"*{suffix}"):
        b = joblib.load(path)
        bundles[(b["crop"], b["season"])] = b
    return bundles
