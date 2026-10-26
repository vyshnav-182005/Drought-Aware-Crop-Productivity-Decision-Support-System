"""
Integration: Hurdle rainfall forecaster -> Crop Yield Prediction, compared with the LSTM

Uses the leak-free direct 3-day hurdle model from codes/forecasting/hurdle_pipeline.py
(features at the cutoff day only; one classifier+regressor per horizon), retrained on days up
to 2018-12-31 so the 2020-2022 test seasons are unseen (see forecasters.HurdleForecaster).

The hurdle forecasts rainfall only, so for its 3 forecast days the other weather variables come
from the analog year, the same as the rest of the season.

Backtests the same in-season protocol as 04 (test years 2020-2022, all crop-seasons) with:
  - Hurdle 3-day forecast
  - LSTM 3-day forecast             (read from 04's output)
  - No forecast                     (observed to cutoff, analog years after)
  - Actual weather (oracle)         (upper bound for what a perfect 3-day forecast could add)

Outputs:
  outputs/yield_prediction/hurdle_yield_predictions.csv
  outputs/yield_prediction/forecaster_comparison.csv
  outputs/yield_prediction/plots/hurdle_vs_lstm_yield_comparison.png
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import MODEL_TABLE_FILE, OUTPUT_DIR, PLOTS_DIR, YIELD_MODELS_DIR, load_all_weather, load_climatology
from forecasters import ActualWeather, HurdleForecaster, LSTMForecaster, NoForecast
from inseason import backtest, load_bundles

warnings.filterwarnings("ignore")


def main():
    mt = pd.read_csv(MODEL_TABLE_FILE)
    weather = load_all_weather()
    clim = load_climatology()
    bundles = load_bundles(YIELD_MODELS_DIR)

    print("Preparing hurdle rainfall forecaster...")
    hurdle = HurdleForecaster(weather)

    runs = {}
    print("Backtesting with Hurdle 3-day forecast...")
    runs["Hurdle"] = backtest(hurdle, mt, bundles, weather, clim)
    runs["Hurdle"].to_csv(OUTPUT_DIR / "hurdle_yield_predictions.csv", index=False)

    lstm_path = OUTPUT_DIR / "in_season_backtest_lstm.csv"
    if lstm_path.exists():
        runs["LSTM"] = pd.read_csv(lstm_path)
    else:
        print("LSTM backtest not found; running it...")
        runs["LSTM"] = backtest(LSTMForecaster(weather), mt, bundles, weather, clim)

    print("Backtesting reference runs (no forecast, actual-weather oracle)...")
    runs["No forecast"] = backtest(NoForecast(), mt, bundles, weather, clim)
    runs["Actual (oracle)"] = backtest(ActualWeather(weather), mt, bundles, weather, clim)

    allbt = pd.concat([df.assign(forecaster=name) for name, df in runs.items()], ignore_index=True)
    allbt["rain_abs_error_3d"] = (allbt["forecast_rain_3d_mm"] - allbt["actual_rain_3d_mm"]).abs()

    comp = allbt.groupby("forecaster").agg(
        yield_mae=("abs_error", "mean"),
        p10_p90_coverage=("in_p10_p90", "mean"),
        rain_3d_mae_mm=("rain_abs_error_3d", "mean"),
        n=("abs_error", "size"),
    ).reindex(list(runs)).round(4)
    by_cutoff = allbt.pivot_table(index=["season", "cutoff"], columns="forecaster", values="abs_error",
                                  aggfunc="mean", sort=False)[list(runs)].round(4)
    comp.to_csv(OUTPUT_DIR / "forecaster_comparison.csv")
    by_cutoff.to_csv(OUTPUT_DIR / "forecaster_comparison_by_cutoff.csv")

    print("\nForecaster comparison over all test-year backtests (2020-2022):")
    print(comp.to_string())
    print("\nYield MAE (t/ha) by cutoff:")
    print(by_cutoff.to_string())

    # Unique 3-day rainfall forecasts (one per district/cutoff date) for a fair rain-skill number
    rain = (allbt.dropna(subset=["forecast_rain_3d_mm"])
            .drop_duplicates(["forecaster", "district", "cutoff_date"]))
    rain_skill = rain.groupby("forecaster").apply(
        lambda g: pd.Series({"rain_3d_mae_mm": (g["forecast_rain_3d_mm"] - g["actual_rain_3d_mm"]).abs().mean(),
                             "n_forecasts": len(g)}), include_groups=False).round(3)
    print("\n3-day rainfall forecast error at the cutoffs (unique forecasts):")
    print(rain_skill.to_string())

    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [f"{s}\n{c}" for s, c in by_cutoff.index]
    x = np.arange(len(labels))
    width = 0.8 / len(runs)
    for i, name in enumerate(runs):
        ax.bar(x + (i - (len(runs) - 1) / 2) * width, by_cutoff[name].values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Yield MAE (t/ha)")
    ax.set_title("In-season yield error by weather forecaster (test 2020-2022, all crops)", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "hurdle_vs_lstm_yield_comparison.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved comparison plot to {PLOTS_DIR / 'hurdle_vs_lstm_yield_comparison.png'}")


if __name__ == "__main__":
    main()
