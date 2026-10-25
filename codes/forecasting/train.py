"""Register or inspect the already-trained hurdle rainfall models."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hurdle_pipeline import load_params, load_legacy_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--district")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    params = load_params()
    districts = [args.district] if args.district else params["districts"]
    for district in districts:
        bundle = load_legacy_bundle(district)
        print(f"{district}: loaded existing checkpoint")
        if not args.no_mlflow:
            from registry import log_and_register
            import pandas as pd
            metrics = pd.read_csv(
                Path("outputs/hurdle/results/hurdle_per_district_metrics.csv")
            )
            metrics = metrics[metrics["District"] == district]
            print(log_and_register(district, bundle, metrics, params))


if __name__ == "__main__":
    main()
