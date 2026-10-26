"""
3-day weather forecasters used for in-season yield prediction.

Both only see weather up to and including the forecast origin (the cutoff date):

- LSTMForecaster: the per-district PyTorch LSTM (models/<district>_weather_lstm_pytorch.pt,
  scalers in outputs/plots/lstm/<District>/). Forecasts all 11 weather variables.
- HurdleForecaster: the leak-free direct multi-horizon hurdle rainfall model from
  codes/forecasting/hurdle_pipeline.py (engineer / FEATURES). Retrained here on days up to
  HURDLE_TRAIN_END so that the yield test seasons (2020-2022) are never in its training data.
  Forecasts rainfall only; other variables fall back to the analog year.
"""

import sys

import joblib
import numpy as np
import pandas as pd

from common import HURDLE_MODEL_DIR, LSTM_MODEL_DIR, LSTM_SCALER_DIR, ROOT_DIR, WEATHER_COLUMNS

sys.path.insert(0, str(ROOT_DIR / "codes" / "forecasting"))

HORIZON = 3
HURDLE_TRAIN_END = pd.Timestamp("2018-12-31")   # same training end as the LSTM


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
class LSTMForecaster:
    name = "LSTM"
    LOG_COLS = ["Rainfall_mm", "Wind_Speed_mps"]
    LOOKBACK = 45

    def __init__(self, weather: dict):
        import torch
        import torch.nn as nn

        class DirectLSTM(nn.Module):  # identical to codes/forecasting/lstm.ipynb
            def __init__(self, n_input, n_target, horizon, lstm_units, num_layers, dense_units, dropout):
                super().__init__()
                self.lstm = nn.LSTM(n_input, lstm_units, num_layers=num_layers, batch_first=True,
                                    dropout=dropout if num_layers > 1 else 0.0)
                self.horizon, self.n_target = horizon, n_target
                self.drop_lstm = nn.Dropout(dropout)
                self.fc1 = nn.Linear(lstm_units + horizon * 2, dense_units)
                self.drop1 = nn.Dropout(dropout)
                self.fc2 = nn.Linear(dense_units, dense_units // 2)
                self.drop2 = nn.Dropout(dropout)
                self.fc3 = nn.Linear(dense_units // 2, horizon * n_target)
                self.act = nn.ReLU()

            def forward(self, seq, cal):
                _, (h, _) = self.lstm(seq)
                x = torch.cat([self.drop_lstm(h[-1]), cal.reshape(cal.size(0), -1)], dim=1)
                x = self.drop1(self.act(self.fc1(x)))
                x = self.drop2(self.act(self.fc2(x)))
                return self.fc3(x).view(-1, self.horizon, self.n_target)

        self.torch = torch
        self.models, self.in_scalers, self.out_scalers, self.inputs = {}, {}, {}, {}
        for district, wdf in weather.items():
            ckpt = torch.load(LSTM_MODEL_DIR / f"{district.lower()}_weather_lstm_pytorch.pt",
                              map_location="cpu", weights_only=False)
            cfg = ckpt["model_config"]
            if cfg["horizon"] != HORIZON:
                raise ValueError(f"{district} LSTM horizon is {cfg['horizon']}, expected {HORIZON}")
            model = DirectLSTM(**cfg)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.models[district] = model
            self.in_scalers[district] = joblib.load(LSTM_SCALER_DIR / district / "in_scaler.joblib")
            self.out_scalers[district] = joblib.load(LSTM_SCALER_DIR / district / "out_scaler.joblib")
            # Model inputs for every day (rolling windows only look backwards, so a window
            # ending at the origin uses no later information).
            feats = self._model_inputs(wdf)
            cols = list(self.in_scalers[district].feature_names_in_)
            self.inputs[district] = pd.DataFrame(self.in_scalers[district].transform(feats[cols]),
                                                 index=feats.index, columns=cols)
        self.targets = WEATHER_COLUMNS

    def _model_inputs(self, df):
        m = df[WEATHER_COLUMNS].copy()
        for c in self.LOG_COLS:
            m[c] = np.log1p(m[c])
        m["Rain_roll7"] = df["Rainfall_mm"].rolling(7, min_periods=1).sum()
        m["Rain_roll14"] = df["Rainfall_mm"].rolling(14, min_periods=1).sum()
        m["Rain_roll30"] = df["Rainfall_mm"].rolling(30, min_periods=1).sum()
        m["Humid_roll7"] = df["Relative_Humidity_percent"].rolling(7, min_periods=1).mean()
        m["Solar_roll7"] = df["Solar_Radiation"].rolling(7, min_periods=1).mean()
        m["Wind_roll7"] = df["Wind_Speed_mps"].rolling(7, min_periods=1).mean()
        m["Wind50_roll7"] = df["Wind_Speed_50m_mps"].rolling(7, min_periods=1).mean()
        m["Cloud_roll7"] = df["Cloud_Amount_percent"].rolling(7, min_periods=1).mean()
        return m

    @staticmethod
    def _calendar(dates):
        doy = np.asarray(dates.dayofyear)
        frac = doy / np.where(np.asarray(dates.is_leap_year), 366, 365)
        return np.stack([np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)], axis=1)

    def forecast(self, district: str, origin: pd.Timestamp) -> pd.DataFrame:
        inp = self.inputs[district].loc[:origin].tail(self.LOOKBACK)
        if len(inp) < self.LOOKBACK or inp.index[-1] != origin:
            raise ValueError(f"{district}: need {self.LOOKBACK} observed days ending {origin.date()}")
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=HORIZON, freq="D")
        seq = self.torch.tensor(inp.to_numpy(np.float32)[None])
        cal = self.torch.tensor(self._calendar(dates).astype(np.float32)[None])
        with self.torch.no_grad():
            scaled = self.models[district](seq, cal).numpy()[0]
        out = self.out_scalers[district].inverse_transform(scaled)
        fc = pd.DataFrame(out, index=dates, columns=list(self.out_scalers[district].feature_names_in_))
        for c in self.LOG_COLS:
            fc[c] = np.expm1(fc[c])
        fc["Rainfall_mm"] = fc["Rainfall_mm"].clip(lower=0.0)
        return fc[self.targets]


# ---------------------------------------------------------------------------
# Hurdle (rainfall only)
# ---------------------------------------------------------------------------
class HurdleForecaster:
    name = "Hurdle"

    def __init__(self, weather: dict, params: dict = None):
        from hurdle_pipeline import FEATURES, engineer, load_params
        self.FEATURES = FEATURES
        self.params = params or load_params()
        self.bundles, self.features = {}, {}
        HURDLE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for district, wdf in weather.items():
            self.features[district] = engineer(wdf)
            self.bundles[district] = self._train_or_load(district, wdf)

    def _train_or_load(self, district, wdf):
        path = HURDLE_MODEL_DIR / f"hurdle_{district.lower()}_train_to_{HURDLE_TRAIN_END.date()}.joblib"
        if path.exists():
            return joblib.load(path)
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from hurdle_pipeline import make_frame
        p = self.params["model"]
        models = []
        for step in range(1, HORIZON + 1):
            frame = make_frame(wdf, step)                       # features at origin, target = rain at origin+step
            train = frame[frame["target_date"] <= HURDLE_TRAIN_END]
            common = {"random_state": 42, "l2_regularization": p["l2_regularization"]}
            clf = HistGradientBoostingClassifier(max_iter=p["max_iter_classifier"],
                                                 learning_rate=p["learning_rate_classifier"], **common)
            reg = HistGradientBoostingRegressor(max_iter=p["max_iter_regressor"],
                                                learning_rate=p["learning_rate_regressor"], **common)
            clf.fit(train[self.FEATURES], train["target"] > p["rain_threshold"])
            rainy = train[train["target"] > 0]
            reg.fit(rainy[self.FEATURES], np.log1p(rainy["target"]))
            models.append({"classifier": clf, "regressor": reg, "horizon": step})
        bundle = {"district": district, "models": models, "features": self.FEATURES,
                  "probability_threshold": p["probability_threshold"],
                  "trained_through": str(HURDLE_TRAIN_END.date())}
        joblib.dump(bundle, path)
        print(f"Trained leak-free hurdle model for {district} (targets <= {HURDLE_TRAIN_END.date()}) -> {path.name}")
        return bundle

    def forecast(self, district: str, origin: pd.Timestamp) -> pd.DataFrame:
        bundle = self.bundles[district]
        x = self.features[district].loc[[origin], self.FEATURES]
        if x.isna().any(axis=None):
            raise ValueError(f"{district}: incomplete hurdle features at {origin.date()}")
        dates, rain = [], []
        for m in bundle["models"]:
            prob = m["classifier"].predict_proba(x)[:, 1][0]
            intensity = max(0.0, float(np.expm1(m["regressor"].predict(x)[0])))
            rain.append(intensity if prob > bundle["probability_threshold"] else 0.0)
            dates.append(origin + pd.Timedelta(days=m["horizon"]))
        return pd.DataFrame({"Rainfall_mm": rain}, index=pd.DatetimeIndex(dates))


class ActualWeather:
    """Oracle reference only: the real weather for the 3 days after the cutoff (not a forecast)."""
    name = "Actual (oracle)"

    def __init__(self, weather: dict):
        self.weather = weather

    def forecast(self, district, origin):
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=HORIZON, freq="D")
        return self.weather[district].reindex(dates).dropna(subset=WEATHER_COLUMNS)


class NoForecast:
    """Observed weather up to the cutoff, analog years afterwards."""
    name = "No forecast"

    def __init__(self, weather: dict = None):
        pass

    def forecast(self, district, origin):
        return None
