"""
Shared definitions for the crop yield prediction pipeline.

Everything that more than one stage needs lives here, so the season windows,
feature definitions and climatology baselines cannot drift apart between
training (02/03) and in-season prediction (04/05).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = next(p for p in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (p / "codes").is_dir() and (p / "data").is_dir())

WEATHER_DIR = ROOT_DIR / "data" / "weather_data" / "weather data new"
CROP_DIR = ROOT_DIR / "data" / "crop_production"
CLEANED_CROP_FILE = CROP_DIR / "cleaned_crop_yield.csv"

OUTPUT_DIR = ROOT_DIR / "outputs" / "yield_prediction"
PLOTS_DIR = OUTPUT_DIR / "plots"
SEASON_FEATURES_FILE = OUTPUT_DIR / "season_weather_features.csv"
MODEL_TABLE_FILE = OUTPUT_DIR / "model_table.csv"

YIELD_MODELS_DIR = ROOT_DIR / "models" / "yield_models"
LSTM_MODEL_DIR = ROOT_DIR / "models"                         # <district>_weather_lstm_pytorch.pt
LSTM_SCALER_DIR = ROOT_DIR / "outputs" / "plots" / "lstm"    # <District>/in_scaler.joblib, out_scaler.joblib
HURDLE_MODEL_DIR = ROOT_DIR / "models" / "hurdle_yield"     # leak-free hurdle models trained for this pipeline

for _d in (OUTPUT_DIR, PLOTS_DIR, YIELD_MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Scope and splits
# ---------------------------------------------------------------------------
DISTRICTS = ["Ahmednagar", "Beed", "Latur", "Osmanabad", "Solapur"]
SEASONS = ["Kharif", "Rabi"]

TRAIN_YEARS = (2000, 2017)   # model fitting + every baseline (normals, SPI, district mean yield)
VAL_YEARS = (2018, 2019)     # model selection only
TEST_YEARS = (2020, 2022)    # reported once, after selection

WEATHER_COLUMNS = [
    "Rainfall_mm", "Temperature_C", "Temperature_Max_C", "Temperature_Min_C",
    "Relative_Humidity_percent", "Wind_Speed_mps", "Surface_Pressure_kPa",
    "Solar_Radiation", "Dew_Point_C", "Cloud_Amount_percent", "Wind_Speed_50m_mps",
]


def in_years(years: pd.Series, span) -> pd.Series:
    return (years >= span[0]) & (years <= span[1])


# ---------------------------------------------------------------------------
# Season / phenological stage windows
# ---------------------------------------------------------------------------
# (stage, start (month, day), end (month, day), year offset of start, year offset of end)
# day = None means "last day of that month" (handles 29 Feb in leap years).
STAGE_WINDOWS = {
    "Kharif": [
        ("sowing",     (6, 1),   (6, 30),    0, 0),
        ("vegetative", (7, 1),   (7, 31),    0, 0),
        ("flowering",  (8, 1),   (8, 31),    0, 0),
        ("maturity",   (9, 1),   (10, 31),   0, 0),
    ],
    "Rabi": [
        ("sowing",     (10, 15), (11, 30),   0, 0),
        ("vegetative", (12, 1),  (12, 31),   0, 0),
        ("flowering",  (1, 1),   (1, 31),    1, 1),
        ("maturity",   (2, 1),   (2, None),  1, 1),
    ],
}
STAGE_NAMES = {s: [w[0] for w in STAGE_WINDOWS[s]] for s in SEASONS}


def _date(year: int, month: int, day):
    if day is None:
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return pd.Timestamp(year=year, month=month, day=day)


def stage_bounds(season: str, year: int):
    """[(stage, start_date, end_date)] for a season starting in `year` (DES crop year 'year-(year+1)')."""
    out = []
    for name, (sm, sd), (em, ed), so, eo in STAGE_WINDOWS[season]:
        out.append((name, _date(year + so, sm, sd), _date(year + eo, em, ed)))
    return out


def season_bounds(season: str, year: int):
    b = stage_bounds(season, year)
    return b[0][1], b[-1][2]


# ---------------------------------------------------------------------------
# Weather loading
# ---------------------------------------------------------------------------
def load_weather(district: str) -> pd.DataFrame:
    """Daily weather from data/weather_data/weather data new, blank trailing rows dropped."""
    path = WEATHER_DIR / f"{district.lower()}_weather_2000_2025.csv"
    df = pd.read_csv(path, parse_dates=["DATE"]).sort_values("DATE")
    df = df.dropna(subset=WEATHER_COLUMNS).drop_duplicates("DATE").set_index("DATE")
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    if len(full) != len(df):
        raise ValueError(f"{district}: weather has {len(full) - len(df)} missing days inside its range")
    return df[WEATHER_COLUMNS]


def load_all_weather() -> dict:
    return {d: load_weather(d) for d in DISTRICTS}


# ---------------------------------------------------------------------------
# Agro-meteorological features
# ---------------------------------------------------------------------------
STAGE_METRICS = ["rainfall", "rainy_days", "longest_dry_spell", "dry_spells_gt10",
                 "mean_tmax", "days_gt35", "gdd", "mean_humidity", "rainfall_pct_departure"]


def dry_spell_metrics(rain: np.ndarray, dry_thresh: float = 2.5):
    """Longest run of dry days (< dry_thresh mm) and number of dry runs longer than 10 days."""
    longest = gt10 = run = 0
    for is_dry in rain < dry_thresh:
        if is_dry:
            run += 1
            longest = max(longest, run)
        else:
            gt10 += run > 10
            run = 0
    gt10 += run > 10
    return int(longest), int(gt10)


def raw_season_features(wdf: pd.DataFrame, season: str, year: int):
    """
    Stage-wise metrics for one season from a daily weather frame (indexed by date).
    Returns None unless every day of the season is present.
    """
    row = {}
    total = 0.0
    for name, start, end in stage_bounds(season, year):
        sl = wdf.loc[start:end]
        if len(sl) != (end - start).days + 1:
            return None
        rain = sl["Rainfall_mm"].to_numpy()
        tmax = sl["Temperature_Max_C"].to_numpy()
        tmin = sl["Temperature_Min_C"].to_numpy()
        longest, gt10 = dry_spell_metrics(rain)
        row[f"{name}_rainfall"] = float(rain.sum())
        row[f"{name}_rainy_days"] = int((rain >= 2.5).sum())
        row[f"{name}_longest_dry_spell"] = longest
        row[f"{name}_dry_spells_gt10"] = gt10
        row[f"{name}_mean_tmax"] = float(tmax.mean())
        row[f"{name}_days_gt35"] = int((tmax > 35.0).sum())
        row[f"{name}_gdd"] = float(np.maximum(0.0, (tmax + tmin) / 2.0 - 10.0).sum())
        row[f"{name}_mean_humidity"] = float(sl["Relative_Humidity_percent"].mean())
        total += float(rain.sum())
    row["seasonal_rainfall"] = total
    return row


def build_season_feature_table(weather: dict) -> pd.DataFrame:
    rows = []
    for district, wdf in weather.items():
        last_year = wdf.index.max().year
        for season in SEASONS:
            for year in range(wdf.index.min().year, last_year + 1):
                feats = raw_season_features(wdf, season, year)
                if feats is not None:
                    rows.append({"district": district, "season": season, "year": year, **feats})
    return pd.DataFrame(rows)


class Climatology:
    """
    Rainfall normals (for % departure) and Gamma SPI parameters per district/season,
    fitted on TRAIN_YEARS only, then applied identically to history and in-season rows.
    """

    def __init__(self):
        self.normals = {}
        self.gamma = {}

    def fit(self, season_features: pd.DataFrame):
        base = season_features[in_years(season_features["year"], TRAIN_YEARS)]
        for (district, season), grp in base.groupby(["district", "season"]):
            for stage in STAGE_NAMES[season]:
                self.normals[(district, season, stage)] = float(grp[f"{stage}_rainfall"].mean())
            r = grp["seasonal_rainfall"].dropna().to_numpy()
            pos = r[r > 0]
            q = (len(r) - len(pos)) / len(r)
            if len(pos) >= 5:
                alpha, _, beta = stats.gamma.fit(pos, floc=0)
            else:  # method-of-moments fallback
                m, v = r.mean(), r.var() + 1e-6
                alpha, beta = m * m / v, v / max(m, 1e-6)
            self.gamma[(district, season)] = (q, float(alpha), float(beta))
        return self

    def spi(self, district: str, season: str, total_rain: float) -> float:
        q, alpha, beta = self.gamma[(district, season)]
        if total_rain <= 0:
            prob = max(q, 1e-6)
        else:
            prob = q + (1 - q) * stats.gamma.cdf(total_rain, alpha, scale=beta)
        return float(stats.norm.ppf(np.clip(prob, 1e-6, 1 - 1e-6)))

    def complete(self, row: dict, district: str, season: str) -> dict:
        """Adds % departure and seasonal SPI to a raw feature row (in place) and returns it."""
        for stage in STAGE_NAMES[season]:
            norm = self.normals[(district, season, stage)]
            row[f"{stage}_rainfall_pct_departure"] = (row[f"{stage}_rainfall"] - norm) / max(norm, 1.0) * 100.0
        row["seasonal_spi"] = self.spi(district, season, row["seasonal_rainfall"])
        return row

    def complete_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = [self.complete(r.to_dict(), r["district"], r["season"]) for _, r in df.iterrows()]
        return pd.DataFrame(rows, index=df.index)


def load_climatology() -> Climatology:
    return Climatology().fit(pd.read_csv(SEASON_FEATURES_FILE))


# ---------------------------------------------------------------------------
# Model feature matrix
# ---------------------------------------------------------------------------
# Never used as inputs: identifiers, the target, and same-year production/area
# (same-year area is only used afterwards to turn yield into production).
NON_FEATURE_COLS = ["district", "crop", "season", "year", "area", "production", "yield"]
DISTRICT_DUMMIES = [f"dist_{d}" for d in DISTRICTS]


def feature_matrix(df: pd.DataFrame, feature_cols=None) -> pd.DataFrame:
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns]).copy()
    for d in DISTRICTS:
        X[f"dist_{d}"] = (df["district"] == d).astype(float)
    if feature_cols is not None:
        X = X.reindex(columns=feature_cols)
    return X.astype(float)


# ---------------------------------------------------------------------------
# In-season weather trajectory
# ---------------------------------------------------------------------------
def analog_date(date: pd.Timestamp, target_year: int, analog_year: int) -> pd.Timestamp:
    """Same calendar day in the analog year. 29 Feb maps to 28 Feb when the analog year is not leap."""
    y = analog_year + (date.year - target_year)
    day = date.day
    if date.month == 2 and day == 29 and not pd.Timestamp(year=y, month=1, day=1).is_leap_year:
        day = 28
    return pd.Timestamp(year=y, month=date.month, day=day)


def composite_weather(wdf: pd.DataFrame, season: str, target_year: int, cutoff: pd.Timestamp,
                      analog_year: int, forecast: pd.DataFrame = None) -> pd.DataFrame:
    """
    Season weather as it would be known at `cutoff`:
      observed days up to cutoff
      + forecast days (columns the forecaster provides; any other column from the analog year)
      + the analog year's weather for every remaining day of the season.
    """
    start, end = season_bounds(season, target_year)
    cutoff = min(pd.Timestamp(cutoff), end)
    observed = wdf.loc[start:cutoff]
    if cutoff >= end:
        return observed

    future_dates = pd.date_range(cutoff + pd.Timedelta(days=1), end, freq="D")
    src = [analog_date(d, target_year, analog_year) for d in future_dates]
    future = wdf.reindex(src)
    future.index = future_dates

    if forecast is not None and len(forecast):
        fc = forecast.loc[forecast.index.isin(future_dates)]
        for col in fc.columns:
            if col in future.columns:
                future.loc[fc.index, col] = fc[col].to_numpy()

    return pd.concat([observed, future])


def predict_bundle(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Predict with a saved yield model bundle (handles log1p-target models)."""
    pred = bundle["model"].predict(X[bundle["feature_cols"]])
    if bundle.get("target_transform") == "log1p":
        pred = np.expm1(pred)
    return np.clip(pred, 0.0, None)
