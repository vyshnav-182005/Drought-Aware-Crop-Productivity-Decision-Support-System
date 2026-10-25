"""Serve three-day hurdle rainfall forecasts from the MLflow champion."""
import csv
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes" / "forecasting"))
from hurdle_pipeline import DATA_DIR, legacy_forecast, load_weather
import registry

app = FastAPI(title="Drought-aware hurdle rainfall forecast")
LOG_PATH = Path(os.environ.get("FORECAST_LOG_PATH", str(ROOT / "logs" / "forecasts_log.csv")))
_cache = {}
_lock = threading.Lock()


def _log(district, version, origin, dates, values):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = ["logged_at", "district", "model_version", "origin_date", "target_date", "rainfall_mm"]
    with _lock:
        exists = LOG_PATH.exists()
        with LOG_PATH.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if not exists:
                writer.writeheader()
            for target, value in zip(dates, values):
                writer.writerow({
                    "logged_at": datetime.now(timezone.utc).isoformat(),
                    "district": district,
                    "model_version": str(version),
                    "origin_date": origin.date().isoformat(),
                    "target_date": target.date().isoformat(),
                    "rainfall_mm": float(value),
                })


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast/{district}")
def forecast_endpoint(district: str):
    normalized = next((item for item in ("Ahmednagar", "Beed", "Latur", "Osmanabad", "Solapur")
                       if item.lower() == district.lower()), None)
    if normalized is None:
        raise HTTPException(status_code=404, detail=f"unsupported district: {district}")
    try:
        if normalized not in _cache:
            version = registry.get_champion(f"{normalized}_hurdle")
            if version is None:
                raise RuntimeError(f"no champion registered for {normalized}")
            _cache[normalized] = (version, registry.download_champion(version))
        version, bundle = _cache[normalized]
        weather = pd.read_csv(
            DATA_DIR / f"{normalized.lower()}_weather.csv", parse_dates=["DATE"]
        ).set_index("DATE")
        origin, value = legacy_forecast(weather, bundle)
        dates, values = [origin], [value]
        _log(normalized, version.version, origin, dates, values)
        return {
            "district": normalized,
            "model_version": str(version.version),
            "origin_date": origin.date().isoformat(),
            "forecast_type": "original_checkpoint_same_day",
            "forecasts": [{
                "target_date": dates[0].date().isoformat(),
                "rainfall_mm": values[0],
            }],
        }
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
