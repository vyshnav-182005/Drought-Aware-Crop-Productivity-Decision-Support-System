# Crop yield prediction

Predicts district-level yield (t/ha) for the 7 Kharif/Rabi crop-seasons in the 5 drought-prone
districts, during the season and at the end of it, and hands yield drop % / production to the
human-impact layer.

Run everything: `python codes/yield_prediction/run_yield_pipeline.py` (about 4 minutes on CPU).

| Script | What it does |
|---|---|
| `01_prepare_crop_data.py` | DES HTML-.xls → `data/crop_production/cleaned_crop_yield.csv` |
| `02_build_features.py` | Stage-wise weather features + SPI → `outputs/yield_prediction/model_table.csv` |
| `03_train_yield_models.py` | Baseline / ElasticNet / RF / XGBoost / HGB per crop-season, MLflow, SHAP |
| `04_in_season_forecast.py` | LSTM 3-day forecast + analog years, backtest 2020-22, impact handoff |
| `05_hurdle_yield_prediction.py` | Hurdle forecast backtest and comparison with LSTM / no forecast / oracle |
| `06_train_hist_gradient_boosting_yield.py` | HistGradientBoosting benchmark (optional) |
| `common.py`, `modeling.py`, `forecasters.py`, `inseason.py` | Shared code (one copy of season windows, features, selection, forecasters) |

Inputs: `data/weather_data/weather data new/*_weather_2000_2025.csv` (daily, 2000-01-01 to 2026-05-30;
blank trailing rows are dropped), `data/crop_production/horizontal_crop_vertical_year_report.xls`,
LSTM checkpoints `models/*_weather_lstm_pytorch.pt` + scalers `outputs/plots/lstm/<District>/`.

## Fixes in this branch

1. **Model chosen on validation, not test.** Candidates are fit on 2000-2017, the one with the lowest
   2018-2019 RMSE is chosen, refit on 2000-2019, and 2020-2022 is scored once.
   Imputation/scaling sit inside each model pipeline, so they are fit on training rows only.
2. **3-day forecasts only use data up to the cutoff.** The old `04` used the real weather for
   the 3 days after the cutoff; the old `05` hurdle fed each forecast day its own observed weather
   (and `rain_roll_3` / `rain_max_3` contained that day's rainfall). Now `05` uses the leak-free
   direct hurdle from `codes/forecasting/hurdle_pipeline.py`, retrained on days ≤ 2018-12-31.
   Checked: scrambling all weather after the cutoff leaves both forecasts unchanged.
3. **LSTM weather model connected.** `04` uses the per-district LSTM for all 11 weather variables.
4. **All baselines use 2000-2017 only** (rainfall normals, SPI Gamma fit, district mean yield).
   A district with no training-year record falls back to the crop's cross-district training mean.
5. **Analog-year dates fixed.** Same calendar day in the analog year (31st no longer copied from
   the 30th); 29 Feb maps to 28 Feb only when needed; Rabi season ends on 29 Feb in leap years.
6. **Same-year sown area is no longer an input.** It is replaced by the previous season's area
   (known before sowing). Same-year area is only used to convert yield into production.
7. One shared copy of the stage windows / feature code; dead return-code check removed from the runner.

Also new: the handoff table includes 2023-2025 seasons (complete weather in the new data, no DES
yield yet) with the latest reported area, flagged `area_assumed=True`.

## Results (test 2020-2022, chosen on validation)

| Crop | Season | Selected | Val RMSE | Test RMSE | Test MAE | Test MAPE % |
|---|---|---|---|---|---|---|
| Arhar/Tur | Kharif | ElasticNet | 0.204 | 0.336 | 0.234 | 38.3 |
| Jowar | Kharif | Baseline | 0.244 | 0.405 | 0.297 | 49.3 |
| Jowar | Rabi | RandomForest | 0.167 | 0.383 | 0.335 | 30.0 |
| Pearl millet | Kharif | HistGradientBoosting | 0.184 | 0.385 | 0.260 | 40.8 |
| Soyabean | Kharif | XGBoost | 0.410 | 0.447 | 0.341 | 21.3 |
| Sunflower | Kharif | RandomForest | 0.319 | 0.227 | 0.177 | 35.3 |
| Sunflower | Rabi | ElasticNet | 0.178 | 0.377 | 0.272 | 33.9 |

In-season backtest, all crops and cutoffs (412 predictions):

| 3-day forecast | Yield MAE (t/ha) | 3-day rain MAE (mm, 120 unique forecasts) |
|---|---|---|
| LSTM | 0.2774 | 5.91 |
| Hurdle | 0.2778 | 6.33 |
| No forecast | 0.2784 | – |
| Actual weather (oracle) | 0.2782 | 0 |

## Things to keep in mind when reporting

- **No drought seasons in the test years.** 2020-2022 were all wet (lowest test seasonal SPI ≈ +0.26),
  so "drought RMSE" is the error on the driest test season, not on a real drought.
- **A 3-day forecast hardly changes seasonal yield.** Even perfect weather for those 3 days
  (oracle) moves MAE by < 0.001 t/ha. A forecast only matters for yield at a much longer horizon.
- **P10-P90 bands are too narrow** (cover the actual yield ~11% of the time) because they only
  reflect weather uncertainty from analog years, not model error.
- Validation is 2 years × 5 districts (~10 rows per crop-season), so model choice is noisy;
  for Jowar Kharif the baseline (district mean + trend) wins.
