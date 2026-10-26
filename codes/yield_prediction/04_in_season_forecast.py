"""
Stage 4: In-Season Yield Prediction (LSTM weather forecast), Backtesting, and Handoff Export
Tasks 11, 12, 13:
- predict_in_season (inseason.py): observed weather up to the cutoff
  + 3-day forecast from the per-district LSTM weather model (uses data up to the cutoff only)
  + historical analog years (2000-2017) for the rest of the season
  -> median yield with P10-P90 bounds.
- Backtest every crop-season over test years 2020-2022 at
    Kharif: Jul 1, Aug 1, Sep 1, Oct 1   |   Rabi: Nov 15, Dec 15, Jan 15, Feb 15
  and plot mean absolute error vs cutoff.
- Task 13 handoff: predicted yield -> production (yield x sown area) and yield drop % vs
  the district's training-period mean, for
    * test seasons 2020-2022 (end-of-season prediction, DES area), and
    * seasons 2023-2025 with complete weather but no DES record yet
      (area = latest reported area, flagged as assumed).
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import (MODEL_TABLE_FILE, OUTPUT_DIR, PLOTS_DIR, SEASON_FEATURES_FILE, TEST_YEARS,
                    YIELD_MODELS_DIR, load_all_weather, load_climatology, season_bounds)
from forecasters import LSTMForecaster
from inseason import CUTOFFS, backtest, load_bundles, predict_in_season

warnings.filterwarnings("ignore")


def condition_flag(spi, drop_pct):
    if spi < -1.0 or drop_pct > 20.0:
        return "Drought-like"
    if drop_pct > 10.0:
        return "Deficit"
    return "Normal"


def plot_error_curves(bt: pd.DataFrame, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, season in zip(axes, ["Kharif", "Rabi"]):
        sub = bt[bt["season"] == season]
        for crop, grp in sub.groupby("crop"):
            err = grp.groupby("cutoff")["abs_error"].mean().reindex(CUTOFFS[season])
            ax.plot(err.index, err.values, marker="o", linewidth=2, label=crop)
        ax.set_title(f"{season}")
        ax.set_xlabel("Cutoff date (MM-DD)")
        ax.set_ylabel("Mean absolute error (t/ha)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_handoff(mt, bundles, weather, clim):
    records = []
    # 1) Test seasons with DES records: end-of-season prediction from fully observed weather
    test = mt[(mt["year"] >= TEST_YEARS[0]) & (mt["year"] <= TEST_YEARS[1])]
    for _, r in test.iterrows():
        b = bundles.get((r["crop"], r["season"]))
        if b is None:
            continue
        _, end = season_bounds(r["season"], int(r["year"]))
        meta = {"district_mean_yield": r["district_mean_yield"], "prev_year_area": r["prev_year_area"]}
        res = predict_in_season(r["district"], r["season"], int(r["year"]), end, weather[r["district"]], b, clim, meta)
        records.append({**r[["district", "crop", "season", "year"]].to_dict(), "record_type": "test (DES known)",
                        "sown_area_ha": r["area"], "area_assumed": False, "actual_yield": r["yield"],
                        "predicted_yield": res["median"], "p10_yield": res["p10"], "p90_yield": res["p90"],
                        "district_historical_mean_yield": r["district_mean_yield"], "seasonal_spi": r["seasonal_spi"]})

    # 2) Seasons after the DES data (2023+) with complete weather in "weather data new"
    season_feats = pd.read_csv(SEASON_FEATURES_FILE)
    last_des_year = int(mt["year"].max())
    latest = mt.sort_values("year").groupby(["district", "crop", "season"]).tail(1)
    for _, last in latest.iterrows():
        b = bundles.get((last["crop"], last["season"]))
        if b is None:
            continue
        future = season_feats[(season_feats["district"] == last["district"]) & (season_feats["season"] == last["season"])
                              & (season_feats["year"] > last_des_year)]
        for _, sf in future.iterrows():
            year = int(sf["year"])
            _, end = season_bounds(last["season"], year)
            meta = {"district_mean_yield": last["district_mean_yield"], "prev_year_area": last["area"]}
            res = predict_in_season(last["district"], last["season"], year, end, weather[last["district"]], b, clim, meta)
            records.append({"district": last["district"], "crop": last["crop"], "season": last["season"], "year": year,
                            "record_type": "post-DES (no actual yet)", "sown_area_ha": last["area"], "area_assumed": True,
                            "actual_yield": float("nan"), "predicted_yield": res["median"], "p10_yield": res["p10"],
                            "p90_yield": res["p90"], "district_historical_mean_yield": last["district_mean_yield"],
                            "seasonal_spi": sf["seasonal_spi"]})

    h = pd.DataFrame(records)
    h["predicted_production_tonnes"] = h["predicted_yield"] * h["sown_area_ha"]
    h["yield_drop_pct"] = ((h["district_historical_mean_yield"] - h["predicted_yield"])
                           / h["district_historical_mean_yield"].clip(lower=1e-4) * 100.0).clip(lower=0.0)
    h["condition_flag"] = [condition_flag(s, d) for s, d in zip(h["seasonal_spi"], h["yield_drop_pct"])]
    return h.round(4).sort_values(["record_type", "crop", "season", "district", "year"])


def main():
    mt = pd.read_csv(MODEL_TABLE_FILE)
    weather = load_all_weather()
    clim = load_climatology()
    bundles = load_bundles(YIELD_MODELS_DIR)
    print(f"Loaded {len(bundles)} crop-season yield models; loading per-district LSTM weather models...")
    lstm = LSTMForecaster(weather)

    print("Backtesting in-season predictions with the LSTM 3-day forecast (2020-2022)...")
    bt = backtest(lstm, mt, bundles, weather, clim)
    bt.to_csv(OUTPUT_DIR / "in_season_backtest_lstm.csv", index=False)
    print(f"Completed {len(bt)} backtest predictions.")
    summary = bt.groupby(["crop", "season", "cutoff"], sort=False).agg(
        mae=("abs_error", "mean"), p10_p90_coverage=("in_p10_p90", "mean")).round(4)
    print(summary.to_string())

    plot_error_curves(bt, PLOTS_DIR / "in_season_backtest_error_curves.png",
                      "In-season yield error vs cutoff (LSTM 3-day forecast + analog years), test 2020-2022")

    print("\nBuilding yield impact handoff table (Task 13)...")
    handoff = build_handoff(mt, bundles, weather, clim)
    handoff.to_csv(OUTPUT_DIR / "yield_impact_handoff.csv", index=False)
    print(handoff.groupby("record_type").size().to_string())
    print(handoff.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
