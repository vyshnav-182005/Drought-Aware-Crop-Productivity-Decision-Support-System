"""Evaluate saved hurdle checkpoints on the fixed 2022+ test period."""
import argparse

import pandas as pd

from hurdle_pipeline import OUTPUT_DIR, legacy_checkpoint, load_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--district")
    args = parser.parse_args()
    params = load_params()
    for district in ([args.district] if args.district else params["districts"]):
        checkpoint = legacy_checkpoint(district)
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing trained checkpoint: {checkpoint}")
        metrics = pd.read_csv(
            OUTPUT_DIR / "results" / "hurdle_per_district_metrics.csv"
        )
        print(metrics[metrics["District"] == district].to_string(index=False))


if __name__ == "__main__":
    main()
