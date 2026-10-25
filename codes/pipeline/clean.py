"""Clean a raw weather folder for hurdle inference and retraining."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting"))
from hurdle_pipeline import ROOT, load_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/weather data new")
    args = parser.parse_args()
    params = load_params()
    output = ROOT / "data" / "weather" / "clean"
    output.mkdir(parents=True, exist_ok=True)
    for district in params["districts"]:
        source_dir = ROOT / args.source if not Path(args.source).is_absolute() else Path(args.source)
        source = source_dir / f"{district.lower()}_weather_2000_2025.csv"
        frame = pd.read_csv(source, parse_dates=["DATE"]).sort_values("DATE")
        features = [column for column in frame.columns if column != "DATE"]
        frame[features] = frame[features].interpolate(method="linear", limit_direction="both")
        if frame[features].isna().any().any():
            raise ValueError(f"{district}: cleaning left missing values")
        frame.to_csv(output / source.name, index=False)
        print(f"{district}: {len(frame)} rows")


if __name__ == "__main__":
    main()
