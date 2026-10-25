"""Score logged champion forecasts and gate rolling challenger promotion."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

import registry
from hurdle_pipeline import DATA_DIR, evaluate_bundle, fit_district, load_params

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "logs" / "forecasts_log.csv"


def score(log, actual, window_days):
    log = log.copy()
    log["target_date"] = pd.to_datetime(log["target_date"])
    log = log[log["target_date"] <= actual.index.max()]
    log = log[log["target_date"] > actual.index.max() - pd.Timedelta(days=window_days)]
    merged = log.merge(actual[["Rainfall_mm"]], left_on="target_date", right_index=True)
    if len(merged) < 2 or merged["Rainfall_mm"].var() == 0:
        return {"days": int(merged["target_date"].nunique()), "rmse": None, "r2": None}
    return {
        "days": int(merged["target_date"].nunique()),
        "rmse": float(np.sqrt(mean_squared_error(merged["Rainfall_mm"], merged["rainfall_mm"]))),
        "r2": float(r2_score(merged["Rainfall_mm"], merged["rainfall_mm"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    params = load_params()
    if not LOG_PATH.exists():
        raise SystemExit(f"{LOG_PATH} does not exist")
    logs = pd.read_csv(LOG_PATH)
    for district in params["districts"]:
        champion = registry.get_champion(f"{district}_hurdle")
        if champion is None:
            print(f"{district}: no champion")
            continue
        district_log = logs[
            (logs["district"] == district)
            & (logs["model_version"].astype(str) == str(champion.version))
        ]
        actual = pd.read_csv(
            DATA_DIR / f"{district.lower()}_weather.csv", parse_dates=["DATE"]
        ).set_index("DATE")
        result = score(district_log, actual, params["monitor"]["window_days"])
        print(f"{district}: {result}")
        baseline = registry.champion_metric(champion, "test_r2")
        if args.retrain and result["days"] >= params["monitor"]["min_days"] and baseline is not None:
            if result["r2"] is not None and result["r2"] < baseline - params["monitor"]["r2_drop_tolerance"]:
                last_date = actual.index.max()
                rolling_start = last_date - pd.Timedelta(days=364)
                challenger, challenger_metrics = fit_district(
                    district, params, save=False, test_start=rolling_start
                )
                champion_bundle = registry.download_champion(champion)
                champion_metrics = evaluate_bundle(district, champion_bundle, rolling_start)
                challenger_r2 = float(challenger_metrics["R2"].mean())
                champion_r2 = float(champion_metrics["R2"].mean())
                print(f"{district}: rolling challenger R2={challenger_r2:.4f}; "
                      f"champion R2={champion_r2:.4f}")
                if challenger_r2 > champion_r2 + params["monitor"]["min_improvement"]:
                    challenger["trained_through"] = str(last_date.date())
                    version = registry.log_and_register(
                        district, challenger, challenger_metrics, params
                    )
                    registry.set_champion(f"{district}_hurdle", version.version)
                    print(f"{district}: promoted hurdle version {version.version}")
                else:
                    print(f"{district}: challenger rejected")


if __name__ == "__main__":
    main()
