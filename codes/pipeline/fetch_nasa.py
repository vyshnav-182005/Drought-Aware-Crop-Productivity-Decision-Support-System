"""Refresh the last 60 days of NASA POWER data into data/weather/live."""
import shutil
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting"))
from hurdle_pipeline import ROOT, load_params

LIVE = ROOT / "data" / "weather data new"
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
DISTRICTS = {
    "Ahmednagar": (19.0948, 74.7480), "Beed": (18.9891, 75.7601),
    "Latur": (18.4088, 76.5604), "Osmanabad": (18.1860, 76.0419),
    "Solapur": (17.6599, 75.9064),
}
PARAMETERS = {
    "PRECTOTCORR": "Rainfall_mm", "T2M": "Temperature_C", "T2M_MAX": "Temperature_Max_C",
    "T2M_MIN": "Temperature_Min_C", "RH2M": "Relative_Humidity_percent",
    "WS10M": "Wind_Speed_mps", "PS": "Surface_Pressure_kPa",
    "ALLSKY_SFC_SW_DWN": "Solar_Radiation", "T2MDEW": "Dew_Point_C",
    "CLOUD_AMT": "Cloud_Amount_percent", "WS50M": "Wind_Speed_50m_mps",
}


def fetch(latitude, longitude, start, end):
    response = requests.get(URL, params={
        "parameters": ",".join(PARAMETERS), "community": "AG",
        "latitude": latitude, "longitude": longitude,
        "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
        "format": "JSON", "time-standard": "LST",
    }, timeout=120)
    response.raise_for_status()
    frame = pd.DataFrame(response.json()["properties"]["parameter"])
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame.index.name = "DATE"
    return frame.reset_index().replace(-999, pd.NA).rename(columns=PARAMETERS)


def main():
    params = load_params()
    LIVE.mkdir(parents=True, exist_ok=True)
    floor = date.fromisoformat(params["nasa"]["start_date"])
    for district, (latitude, longitude) in DISTRICTS.items():
        path = LIVE / f"{district.lower()}_weather_2000_2025.csv"
        if not path.exists():
            seed = ROOT / "data" / "weather" / "v2" / f"{district.lower()}_weather.csv"
            shutil.copy2(seed, path)
        old = pd.read_csv(path, parse_dates=["DATE"])
        start = max(floor, old["DATE"].max().date() - timedelta(days=params["nasa"]["refresh_days"]))
        fresh = fetch(latitude, longitude, start, date.today())
        merged = pd.concat([old[old["DATE"] < pd.Timestamp(start)], fresh])
        merged = merged.drop_duplicates("DATE", keep="last").sort_values("DATE")[old.columns]
        merged.to_csv(path, index=False)
        print(f"{district}: refreshed through {merged['DATE'].max().date()}")
        time.sleep(2)


if __name__ == "__main__":
    main()
