"""Register newly trained local hurdle bundles and mark them as champions."""
import argparse
import joblib
import pandas as pd

from hurdle_pipeline import MODEL_DIR, OUTPUT_DIR, load_params, legacy_checkpoint
from registry import log_and_register, set_champion, setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--district")
    args = parser.parse_args()
    setup()
    params = load_params()
    districts = [args.district] if args.district else params["districts"]
    for district in districts:
        try:
            bundle = joblib.load(legacy_checkpoint(district))
            metrics = pd.read_csv(OUTPUT_DIR / "results" / "hurdle_per_district_metrics.csv")
            metrics = metrics[metrics["District"] == district]
            version = log_and_register(district, bundle, metrics, params)
            set_champion(f"{district}_hurdle", version.version)
            print(f"{district}: registered hurdle version {version.version} as champion")
        except Exception as error:
            raise RuntimeError(f"{district}: registration failed: {error}") from error


if __name__ == "__main__":
    main()
