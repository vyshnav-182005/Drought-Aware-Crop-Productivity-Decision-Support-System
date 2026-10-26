"""
Stage 2: Agro-Meteorological Feature Engineering
Tasks 4, 5, 6:
- Season windows & phenological stages (common.STAGE_WINDOWS):
    Kharif: Sowing (Jun), Vegetative (Jul), Flowering (Aug), Maturity (Sep-Oct)
    Rabi:   Sowing (Oct 15-Nov), Vegetative (Dec), Flowering (Jan), Maturity (Feb, incl. 29 Feb)
- Per stage: total rainfall, rainy days, longest dry spell, dry spells > 10d,
  mean Tmax, days > 35C, growing degree days (GDD), mean humidity,
  rainfall % departure from normal; plus seasonal rainfall and seasonal SPI.
- Weather source: data/weather_data/weather data new/<district>_weather_2000_2025.csv
- Every baseline (stage normals, SPI Gamma fit, district mean yield) uses TRAIN_YEARS
  (2000-2017) only, so validation/test years never leak into features.
- Non-weather features: trend (year), district mean yield (train years),
  previous season's sown area (same-year area is not used as an input).

Outputs:
- outputs/yield_prediction/season_weather_features.csv  (every complete season, incl. 2023-2025)
- outputs/yield_prediction/model_table.csv              (seasons with a DES yield record)
"""

import pandas as pd

from common import (CLEANED_CROP_FILE, MODEL_TABLE_FILE, SEASON_FEATURES_FILE, TRAIN_YEARS,
                    Climatology, build_season_feature_table, in_years, load_all_weather)


def add_crop_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["district", "crop", "season", "year"]).copy()
    df["trend"] = df["year"]

    # District mean yield from training years only.
    train = df[in_years(df["year"], TRAIN_YEARS)]
    dist_mean = train.groupby(["district", "crop", "season"])["yield"].mean().rename("district_mean_yield")
    # If a district has no training-year record for a crop, use that crop's
    # cross-district training mean (never the all-years mean).
    crop_mean = train.groupby(["crop", "season"])["yield"].mean().rename("crop_mean_yield")
    df = df.join(dist_mean, on=["district", "crop", "season"]).join(crop_mean, on=["crop", "season"])
    df["district_mean_yield"] = df["district_mean_yield"].fillna(df["crop_mean_yield"])
    df = df.drop(columns="crop_mean_yield")

    # Previous season's sown area (most recent earlier record) - known before the season starts.
    df["prev_year_area"] = df.groupby(["district", "crop", "season"])["area"].transform(lambda s: s.shift(1).ffill())
    return df


def main():
    print("Loading district daily weather (weather data new)...")
    weather = load_all_weather()
    for d, w in weather.items():
        print(f"  {d:<11s} {w.index.min().date()} -> {w.index.max().date()} ({len(w)} days)")

    print("Building stage-wise agro-meteorological features...")
    raw = build_season_feature_table(weather)
    clim = Climatology().fit(raw)
    season_feats = clim.complete_frame(raw)
    season_feats.to_csv(SEASON_FEATURES_FILE, index=False)
    print(f"Saved {len(season_feats)} complete district-season rows "
          f"({season_feats['year'].min()}-{season_feats['year'].max()}) to {SEASON_FEATURES_FILE}")

    crop_df = pd.read_csv(CLEANED_CROP_FILE)
    print(f"Loaded {len(crop_df)} crop yield observations.")

    merged = crop_df.merge(season_feats, on=["district", "season", "year"], how="inner")
    merged = add_crop_history_features(merged)
    merged.to_csv(MODEL_TABLE_FILE, index=False)
    print(f"Saved modeling table ({len(merged)} rows, {merged.shape[1]} columns) to {MODEL_TABLE_FILE}")
    print(merged[["district", "crop", "season", "year", "yield", "seasonal_rainfall",
                  "seasonal_spi", "district_mean_yield", "prev_year_area"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
