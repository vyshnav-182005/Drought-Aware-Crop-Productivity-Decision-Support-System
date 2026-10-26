"""
Download daily soil wetness (2000 -> today) for the 5 districts from NASA POWER (no account needed).

Parameters (MERRA-2, 0 = completely dry, 1 = saturated):
  GWETTOP   surface soil wetness (0-5 cm)
  GWETROOT  root-zone soil wetness (0-100 cm)
  GWETPROF  whole-profile soil wetness

Output: data/remote_sensing/<district>_soil_moisture_nasa.csv  (DATE, GWETTOP, GWETROOT, GWETPROF)
Run:    python codes/pipeline/fetch_soil_moisture_nasa.py
"""
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "remote_sensing"
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
DISTRICTS = {  # same points as fetch_nasa.py
    "Ahmednagar": (19.0948, 74.7480), "Beed": (18.9891, 75.7601),
    "Latur": (18.4088, 76.5604), "Osmanabad": (18.1860, 76.0419),
    "Solapur": (17.6599, 75.9064),
}
PARAMETERS = ["GWETTOP", "GWETROOT", "GWETPROF"]


def fetch(lat, lon, start, end):
    r = requests.get(URL, params={
        "parameters": ",".join(PARAMETERS), "community": "AG", "latitude": lat, "longitude": lon,
        "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "format": "JSON",
        "time-standard": "LST",
    }, timeout=300)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "DATE"
    return df.replace(-999, pd.NA).reset_index()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for district, (lat, lon) in DISTRICTS.items():
        parts = []
        for y0 in range(2000, date.today().year + 1, 10):   # 10-year chunks keep requests small
            start = date(y0, 1, 1)
            end = min(date(y0 + 9, 12, 31), date.today())
            parts.append(fetch(lat, lon, start, end))
            time.sleep(2)
        df = pd.concat(parts).drop_duplicates("DATE").sort_values("DATE")
        path = OUT_DIR / f"{district.lower()}_soil_moisture_nasa.csv"
        df.to_csv(path, index=False)
        print(f"{district}: {len(df)} days ({df['DATE'].min().date()} -> {df['DATE'].max().date()}) -> {path.name}")


if __name__ == "__main__":
    main()
