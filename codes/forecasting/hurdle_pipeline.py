"""Leakage-safe three-day rainfall hurdle pipeline."""
import json
import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "weather data new"
MODEL_DIR = ROOT / "outputs" / "hurdle" / "checkpoints"
OUTPUT_DIR = ROOT / "outputs" / "hurdle"
FEATURES = [
    "sin_day_365", "cos_day_365", "rain_lag_1", "rain_lag_2", "rain_lag_3",
    "rain_roll_3", "rain_roll_7", "rain_roll_14", "rain_max_3",
    "humidity_lag_1", "humidity_diff_1", "pressure_lag_1", "pressure_diff_1",
    "Relative_Humidity_percent", "Surface_Pressure_kPa", "Temperature_C",
    "Temperature_Max_C", "Temperature_Min_C", "Wind_Speed_mps",
    "Solar_Radiation", "Dew_Point_C",
]


def load_params():
    return yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))


def load_weather(district):
    path = DATA_DIR / f"{district.lower()}_weather_2000_2025.csv"
    frame = pd.read_csv(path, parse_dates=["DATE"]).sort_values("DATE").set_index("DATE")
    required = {
        "Rainfall_mm", "Relative_Humidity_percent", "Surface_Pressure_kPa",
        "Temperature_C", "Temperature_Max_C", "Temperature_Min_C",
        "Wind_Speed_mps", "Solar_Radiation", "Dew_Point_C",
    }
    missing = required.difference(set(frame.columns))
    if missing:
        raise ValueError(f"{district}: missing weather columns: {sorted(missing)}")
    return frame


def legacy_checkpoint(district):
    return MODEL_DIR / f"hurdle_{district}.joblib"


def load_legacy_bundle(district):
    path = legacy_checkpoint(district)
    if not path.exists():
        raise FileNotFoundError(f"missing trained hurdle checkpoint: {path}")
    return joblib.load(path)


def legacy_features(frame):
    """Reproduce the feature contract used by the committed hurdle checkpoints."""
    result = frame.copy()
    result["day_of_year"] = result.index.day_of_year.astype(float)
    result["sin_day_365"] = np.sin(2 * np.pi * result["day_of_year"] / 365.25)
    result["cos_day_365"] = np.cos(2 * np.pi * result["day_of_year"] / 365.25)
    result["month_num"] = result.index.month
    result["temp_range"] = result["Temperature_Max_C"] - result["Temperature_Min_C"]
    result["dew_spread"] = result["Temperature_C"] - result["Dew_Point_C"]
    result["wind_shear"] = result["Wind_Speed_50m_mps"] - result["Wind_Speed_mps"]
    saturation = 0.61078 * np.exp(
        (17.27 * result["Temperature_C"]) / (result["Temperature_C"] + 237.3)
    )
    actual = saturation * result["Relative_Humidity_percent"] / 100.0
    result["vpd"] = saturation - actual
    result["moisture_flux"] = (
        result["Wind_Speed_mps"] * result["Relative_Humidity_percent"]
    ) / (result["vpd"] + 0.1)
    rain = result["Rainfall_mm"]
    result["rain_lag_1"] = rain.shift(1)
    result["rain_lag_2"] = rain.shift(2)
    result["rain_lag_3"] = rain.shift(3)
    result["rain_roll_3"] = rain.rolling(3, min_periods=1).mean()
    result["rain_roll_7"] = rain.rolling(7, min_periods=1).mean()
    result["rain_roll_14"] = rain.rolling(14, min_periods=1).mean()
    result["rain_max_3"] = rain.rolling(3, min_periods=1).max()
    result["humidity_lag_1"] = result["Relative_Humidity_percent"].shift(1)
    result["humidity_diff_1"] = (
        result["Relative_Humidity_percent"] - result["humidity_lag_1"]
    )
    result["pressure_lag_1"] = result["Surface_Pressure_kPa"].shift(1)
    result["pressure_diff_1"] = (
        result["Surface_Pressure_kPa"] - result["pressure_lag_1"]
    )
    return result.ffill().bfill()


def legacy_forecast(weather, bundle):
    """Predict the latest observed day using the original checkpoint contract."""
    features = legacy_features(weather).iloc[-1:]
    values = [
        "sin_day_365", "cos_day_365", "temp_range", "dew_spread", "wind_shear",
        "vpd", "moisture_flux", "rain_lag_1", "rain_lag_2", "rain_lag_3",
        "rain_roll_3", "rain_roll_7", "rain_roll_14", "rain_max_3",
        "Relative_Humidity_percent", "humidity_diff_1", "Surface_Pressure_kPa",
        "pressure_diff_1", "Temperature_C", "Temperature_Max_C",
        "Temperature_Min_C", "Wind_Speed_mps", "Solar_Radiation", "Dew_Point_C",
    ]
    probability = bundle["clf"].predict_proba(features[values])[:, 1][0]
    intensity = max(0.0, float(np.expm1(bundle["reg"].predict(features[values])[0])))
    prediction = intensity if probability > 0.35 else 0.0
    if features["Relative_Humidity_percent"].iloc[0] < 55 or features["dew_spread"].iloc[0] > 6:
        prediction = 0.0
    return features.index[0], float(prediction)


def engineer(frame):
    source = frame.copy()
    output = pd.DataFrame(index=source.index)
    day = source.index.day_of_year.to_numpy()
    output["sin_day_365"] = np.sin(2 * np.pi * day / 365.25)
    output["cos_day_365"] = np.cos(2 * np.pi * day / 365.25)
    rain = source["Rainfall_mm"]
    for lag in (1, 2, 3):
        output[f"rain_lag_{lag}"] = rain.shift(lag)
    output["rain_roll_3"] = rain.shift(1).rolling(3, min_periods=3).mean()
    output["rain_roll_7"] = rain.shift(1).rolling(7, min_periods=7).mean()
    output["rain_roll_14"] = rain.shift(1).rolling(14, min_periods=14).mean()
    output["rain_max_3"] = rain.shift(1).rolling(3, min_periods=3).max()
    humidity = source["Relative_Humidity_percent"]
    pressure = source["Surface_Pressure_kPa"]
    output["humidity_lag_1"] = humidity.shift(1)
    output["humidity_diff_1"] = humidity - humidity.shift(1)
    output["pressure_lag_1"] = pressure.shift(1)
    output["pressure_diff_1"] = pressure - pressure.shift(1)
    for column in (
        "Relative_Humidity_percent", "Surface_Pressure_kPa", "Temperature_C",
        "Temperature_Max_C", "Temperature_Min_C", "Wind_Speed_mps",
        "Solar_Radiation", "Dew_Point_C",
    ):
        output[column] = source[column]
    return output


def make_frame(weather, horizon):
    features = engineer(weather)
    target = weather["Rainfall_mm"].shift(-horizon).rename("target")
    frame = features.join(target).dropna()
    frame["target_date"] = frame.index + pd.to_timedelta(horizon, unit="D")
    return frame


def _models(params):
    model = params["model"]
    common = {"random_state": 42, "l2_regularization": model["l2_regularization"]}
    return (
        HistGradientBoostingClassifier(
            max_iter=model["max_iter_classifier"],
            learning_rate=model["learning_rate_classifier"], **common,
        ),
        HistGradientBoostingRegressor(
            max_iter=model["max_iter_regressor"],
            learning_rate=model["learning_rate_regressor"], **common,
        ),
    )


def fit_district(district, params, save=True, test_start=None):
    params = copy.deepcopy(params)
    if test_start is not None:
        params["split"]["test_start"] = str(pd.Timestamp(test_start).date())
    weather = load_weather(district)
    horizon = params["model"]["horizon"]
    test_start = pd.Timestamp(params["split"]["test_start"])
    results = []
    bundles = []
    for step in range(1, horizon + 1):
        frame = make_frame(weather, step)
        test = frame[frame["target_date"] >= test_start]
        trainval = frame[frame["target_date"] < test_start]
        val_start = trainval["target_date"].max() - pd.Timedelta(days=params["split"]["validation_days"] - 1)
        train = trainval[trainval["target_date"] < val_start]
        validation = trainval[trainval["target_date"] >= val_start]
        if train.empty or validation.empty or test.empty:
            raise ValueError(f"{district} horizon {step}: insufficient chronological data")
        classifier, regressor = _models(params)
        threshold = params["model"]["rain_threshold"]
        classifier.fit(train[FEATURES], train["target"] > threshold)
        rainy = train[train["target"] > 0]
        regressor.fit(rainy[FEATURES], np.log1p(rainy["target"]))
        probability = classifier.predict_proba(test[FEATURES])[:, 1]
        intensity = np.clip(np.expm1(regressor.predict(test[FEATURES])), 0, None)
        prediction = np.where(probability > params["model"]["probability_threshold"], intensity, 0.0)
        actual = test["target"].to_numpy()
        results.append({
            "horizon": step, "n": len(test),
            "MAE": mean_absolute_error(actual, prediction),
            "RMSE": np.sqrt(mean_squared_error(actual, prediction)),
            "R2": r2_score(actual, prediction),
            "validation_days": len(validation),
        })
        bundles.append({"classifier": classifier, "regressor": regressor, "horizon": step})
    bundle = {
        "district": district, "features": FEATURES, "horizon": horizon,
        "probability_threshold": params["model"]["probability_threshold"],
        "rain_threshold": params["model"]["rain_threshold"], "models": bundles,
        "trained_through": str(weather.index.max().date()),
    }
    metrics = pd.DataFrame(results)
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_DIR / f"{district.lower()}.joblib")
        metrics.to_csv(OUTPUT_DIR / f"{district.lower()}_metrics.csv", index=False)
        (OUTPUT_DIR / f"{district.lower()}_config.json").write_text(
            json.dumps({key: value for key, value in bundle.items() if key != "models"}, indent=2),
            encoding="utf-8",
        )
    return bundle, metrics


def forecast(district, weather, bundle):
    features = engineer(weather).dropna().tail(1)
    if features.empty:
        raise ValueError("at least 14 complete days are required for forecasting")
    origin = features.index[-1]
    predictions = []
    dates = []
    for model in bundle["models"]:
        probability = model["classifier"].predict_proba(features[bundle["features"]])[:, 1][0]
        intensity = max(0.0, float(np.expm1(model["regressor"].predict(features[bundle["features"]])[0])))
        predictions.append(intensity if probability > bundle["probability_threshold"] else 0.0)
        dates.append(origin + pd.Timedelta(days=model["horizon"]))
    return origin, dates, predictions


def evaluate_bundle(district, bundle, test_start=None):
    weather = load_weather(district)
    rows = []
    for model in bundle["models"]:
        frame = make_frame(weather, model["horizon"])
        start = test_start or load_params()["split"]["test_start"]
        test = frame[frame["target_date"] >= pd.Timestamp(start)]
        probability = model["classifier"].predict_proba(test[bundle["features"]])[:, 1]
        intensity = np.clip(np.expm1(model["regressor"].predict(test[bundle["features"]])), 0, None)
        prediction = np.where(probability > bundle["probability_threshold"], intensity, 0.0)
        rows.append({
            "horizon": model["horizon"], "n": len(test),
            "MAE": mean_absolute_error(test["target"], prediction),
            "RMSE": np.sqrt(mean_squared_error(test["target"], prediction)),
            "R2": r2_score(test["target"], prediction),
        })
    return pd.DataFrame(rows)
