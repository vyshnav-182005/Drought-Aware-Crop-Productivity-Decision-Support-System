# MLOps implementation guide: Drought-Aware Crop Productivity DSS

> **Production scope:** this implementation uses the hurdle rainfall ensemble
> only. The LSTM notebook is retained as an archived experiment and is not
> trained, registered, served, or monitored by the MLOps pipeline below.
> Sections in the original plan that refer to LSTM-specific artifacts should
> be interpreted as hurdle artifacts: `models/hurdle/`, `outputs/hurdle/`, and
> model names such as `<district>_hurdle`.

The production entry points are:

```bash
python codes/forecasting/train.py --district Latur --no-mlflow
python codes/forecasting/evaluate.py --district Latur
pytest -q
```

The current baseline is not retrained by these commands. It imports the
existing checkpoints from `outputs/hurdle/checkpoints/`, which were trained
from `data/weather data new/`, and reads the authoritative metrics from
`outputs/hurdle/results/hurdle_per_district_metrics.csv`. This preserves the
model that produced the project's reported results. Any future retraining must
first be explicitly approved because the original checkpoint contract is a
same-day validation model, not a verified three-day-ahead model.

Branch: `MLOps`. Shell: Ubuntu (WSL). All commands run from the repo root unless stated.

---

## 0. Read this first

**Status of the code in this file.** None of it has been run. It is written from the files you shared (LSTM notebook, preprocessing notebook, hurdle script, knowledge-graph scripts). Every task has a "Check" step. Run it, and if the result differs from what is stated, stop and report it before continuing.

**Things I could not verify (each can break a step):**

| # | Unverified point | Where it matters | Fallback |
|---|---|---|---|
| 1 | DagsHub's MLflow server supports model-registry **aliases** | Task 3 | `registry.py` also sets a `champion=true` version tag and reads it if the alias call fails |
| 2 | `mlflow.register_model("runs:/<id>/bundle", ...)` accepts a folder that is not an MLflow "model flavor" | Task 3 | Log the model with `mlflow.pytorch.log_model` instead (needs `DirectLSTM` importable at load time) |
| 3 | DagsHub menu names (Remote, Tokens) | Phase A | Use the DagsHub docs search |
| 4 | NASA POWER lag (how many days behind today) | Task 4 | Measured in Phase A step 7; the API reports `data_age_days` |
| 5 | Your WSL Python / scikit-learn versions | Tasks 4, 5 | Pin the same versions in Docker and CI |
| 6 | Repo has no branch protection on `main` | Task 5 (the weekly job pushes a `.dvc` file) | Use a pull request instead of a direct push |

**Decisions your team still has to make (I used the defaults shown; change them in `params.yaml`):**

| Decision | Default used here |
|---|---|
| Retrain trigger | Rolling R² < champion's own test R² minus `0.10` |
| Rolling split | test = last 365 days, validation = 1095 days before that, train = everything earlier |
| Where weekly NASA data goes | `data/weather/live/` (v2 stays untouched) |
| Where `forecasts_log.csv` lives for the weekly monitor | Open. See Task 6, "Open problem" |

**What MLOps does not change:** the model, its features and its accuracy.

---

## 1. Phase A: One-time setup

### A1. Tools

```bash
git --version
python3 --version      # note the minor version (e.g. 3.11); use the same in CI and Docker
sudo apt update && sudo apt install -y git python3-venv
```

### A2. GitHub SSH access

```bash
ssh-keygen -t ed25519 -C "your-github-email@example.com"     # press Enter 3 times
cat ~/.ssh/id_ed25519.pub
```
GitHub → Settings → SSH and GPG keys → New SSH key → paste. Then:
```bash
ssh -T git@github.com     # expect: "Hi <username>! You've successfully authenticated"
```

### A3. Clone and create your branch

```bash
cd ~
git clone git@github.com:vyshnav-182005/Drought-Aware-Crop-Productivity-Decision-Support-System.git
cd Drought-Aware-Crop-Productivity-Decision-Support-System
git config user.name "Your Name"
git config user.email "your-github-email@example.com"
git checkout -b vaish_mlops
git push -u origin vaish_mlops
git branch --show-current          # must print: vaish_mlops
```
Permission error on push → ask the repo owner to add you under Settings → Collaborators.

Clone inside WSL (`~/...`), not under `/mnt/c/...` (very slow with DVC). Open the folder in VS Code with `code .`.

### A4. Inspect the repo (send me this output if anything looks odd)

```bash
ls
cat .gitignore
ls outputs outputs/* | head -40
git ls-files | grep -E "\.(pt|joblib)$" | head -20
git ls-files data | head -20
du -sh data models outputs
```

### A5. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
cat requirements.txt
pip install -r requirements.txt
pip install dvc mlflow fastapi uvicorn pytest pyyaml requests
python -c "import torch, pandas, sklearn; print(torch.__version__, torch.cuda.is_available())"
```

Append to `.gitignore` (skip lines that already exist):
```bash
cat >> .gitignore <<'EOF'
.venv/
.env
__pycache__/
mlruns/
logs/
EOF
```
Pin the new packages:
```bash
pip freeze | grep -iE "^(dvc|mlflow|fastapi|uvicorn|pytest|pyyaml|requests)==" >> requirements.txt
```
Open `requirements.txt` afterwards and remove duplicate lines.

### A6. Baseline: prove the existing project reproduces

```bash
pip install jupyter nbconvert
cp -r outputs/lstm /tmp/lstm_baseline            # keep the original metrics for later comparison
cp -r data/weather/clean /tmp/clean_baseline
jupyter nbconvert --to notebook --execute codes/forecasting/lstm.ipynb --output /tmp/lstm_check.ipynb
```
With `RUN_TRAINING = False` this only loads the saved models. If `outputs/lstm` does not exist, the path in `ls outputs` (A4) tells you where the metrics live; use that path instead.

### A7. Measure the NASA lag

```bash
tail -n 2 data/weather/clean/latur_weather.csv | cut -d, -f1
date +%F
python - <<'EOF'
import requests, datetime as dt
end = dt.date.today(); start = end - dt.timedelta(days=30)
r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", params={
    "parameters":"PRECTOTCORR","community":"AG","latitude":18.4088,"longitude":76.5604,
    "start":start.strftime("%Y%m%d"),"end":end.strftime("%Y%m%d"),"format":"JSON"}, timeout=60)
vals = r.json()["properties"]["parameter"]["PRECTOTCORR"]
last = max(d for d, v in vals.items() if v != -999)
print("today:", end, "| last real NASA day:", last)
EOF
```
The gap between those two dates is the lag that Task 4 must live with.

### A8. DagsHub account and repo

1. dagshub.com → sign up with GitHub.
2. Create a new repository by connecting this GitHub repo.
3. Open the repo page → **Remote** → copy the **DVC** URL and the **MLflow** URL (`https://dagshub.com/<USER>/<REPO>.dvc` and `.mlflow`).
4. Account Settings → **Tokens** → create a token.
5. Confirm with your team that nobody else created a second DagsHub repo for this project. Two repos mean two different "champion" models.

Create a local secrets file (never commit it; `.env` is already in `.gitignore`):
```bash
cat > .env <<'EOF'
MLFLOW_TRACKING_URI=https://dagshub.com/<USER>/<REPO>.mlflow
MLFLOW_TRACKING_USERNAME=<USER>
MLFLOW_TRACKING_PASSWORD=<TOKEN>
EOF
```
Load it into your shell whenever you open a terminal:
```bash
set -a; source .env; set +a
```

### A9. Commit routine (use after every task)

```bash
git status
git add <specific files>        # never "git add ." (risk of committing .env or large files)
git commit -m "task N: <what>"
git push
```
Sync with `main` when teammates merge:
```bash
git fetch origin
git merge origin/main
git push
```

---

## 2. Task 0: Verify the hurdle R² against the LSTM R²

Verdict from reading `train_hurdle_per_district.py` (not run): **the 0.79 to 0.94 rainfall R² is not comparable to the LSTM's**.

| Check | Result | Evidence |
|---|---|---|
| Same daily scale (mm) | Pass | Raw `Rainfall_mm` |
| Same test years (2022+) | **Fail** | `val_days = 365` evaluates only the last year of the file |
| Same 3-day horizon | **Fail** | No shift anywhere. Day *t* features predict day *t* rain (a nowcast), though plot titles say "3-Day Ahead" |
| No future data in inputs | **Fail** | `rain_roll_3/7/14` and `rain_max_3` include day *t*'s own rain; same-day RH, dew point, pressure, temperature and wind are inputs; the Stage 3 gate uses same-day RH and dew spread |

Other risks: the `0.35` threshold and the gate cut-offs may have been tuned on the evaluation window (ask the author); `ffill().bfill()` before the split lets `bfill` pull values backwards.

### Corrected script

```bash
git checkout vaish_mlops
mkdir -p codes/forecasting outputs/hurdle
cp <path-to>/train_hurdle_per_district.py codes/forecasting/hurdle.py
```

In `hurdle.py`:
1. Delete `DISTRICT_FILES` and the plotting functions (not needed for the comparison).
2. In `load_and_engineer_district_data`, change the last fill line to forward-fill only:
   `df[numeric_cols] = df[numeric_cols].ffill()`
3. Replace `train_and_eval_district` and `main` with the code below. Keep `feature_cols` exactly as in the original script.

```python
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data" / "weather" / "clean"
DISTRICTS = ["Latur", "Beed", "Osmanabad", "Solapur", "Ahmednagar"]
TRAIN_END, VAL_END, H = pd.Timestamp("2018-12-31"), pd.Timestamp("2021-12-31"), 3

FEATURE_COLS = [
    'sin_day_365', 'cos_day_365', 'temp_range', 'dew_spread', 'wind_shear', 'vpd', 'moisture_flux',
    'rain_lag_1', 'rain_lag_2', 'rain_lag_3', 'rain_roll_3', 'rain_roll_7', 'rain_roll_14', 'rain_max_3',
    'Relative_Humidity_percent', 'humidity_diff_1', 'Surface_Pressure_kPa', 'pressure_diff_1',
    'Temperature_C', 'Temperature_Max_C', 'Temperature_Min_C', 'Wind_Speed_mps', 'Solar_Radiation', 'Dew_Point_C',
]


def make_frame(df, h):
    d = df.copy()
    d["target"] = d["Rainfall_mm"].shift(-h)                      # rain h days AFTER the origin day
    d["target_date"] = d["DATE"] + pd.Timedelta(days=h)
    frac = d["target_date"].dt.dayofyear / np.where(d["target_date"].dt.is_leap_year, 366, 365)
    d["sin_day_365"] = np.sin(2 * np.pi * frac)                   # calendar of the day being forecast
    d["cos_day_365"] = np.cos(2 * np.pi * frac)
    return d.dropna(subset=["target"])


def split_by_target_date(d):
    td = d["target_date"]
    return d[td <= TRAIN_END], d[(td > TRAIN_END) & (td <= VAL_END)], d[td > VAL_END]


def hurdle_predict(prob, intensity, thr):
    return np.where(prob > thr, intensity, 0.0)


def run_district(district):
    df = load_and_engineer_district_data(CLEAN_DIR / f"{district.lower()}_weather.csv", district)
    y_all, p_all = [], []
    for h in range(1, H + 1):
        tr, va, te = split_by_target_date(make_frame(df, h))
        clf = HistGradientBoostingClassifier(random_state=42, max_iter=250, learning_rate=0.04,
                                             l2_regularization=0.1).fit(tr[FEATURE_COLS], tr["target"] > 0.25)
        rainy = tr[tr["target"] > 0.0]
        reg = HistGradientBoostingRegressor(random_state=42, max_iter=350, learning_rate=0.03,
                                            l2_regularization=0.1).fit(rainy[FEATURE_COLS], np.log1p(rainy["target"]))
        # threshold is chosen on VALIDATION only
        pv = clf.predict_proba(va[FEATURE_COLS])[:, 1]
        iv = np.clip(np.expm1(reg.predict(va[FEATURE_COLS])), 0, None)
        thr = max(np.arange(0.20, 0.65, 0.05), key=lambda t: r2_score(va["target"], hurdle_predict(pv, iv, t)))
        # TEST: evaluated once
        pt = clf.predict_proba(te[FEATURE_COLS])[:, 1]
        it = np.clip(np.expm1(reg.predict(te[FEATURE_COLS])), 0, None)
        y_all.append(te["target"].values)
        p_all.append(hurdle_predict(pt, it, thr))
    y, p = np.concatenate(y_all), np.concatenate(p_all)
    return {"district": district, "R2": r2_score(y, p),
            "RMSE": root_mean_squared_error(y, p), "MAE": mean_absolute_error(y, p)}


if __name__ == "__main__":
    out = pd.DataFrame([run_district(d) for d in DISTRICTS]).round(4)
    out.to_csv(ROOT / "outputs" / "hurdle" / "hurdle_rainfall_metrics.csv", index=False)
    print(out.to_string(index=False))
```
The original Stage 3 physics gate is dropped on purpose: with a shifted target, origin-day RH and dew spread no longer describe the target day. If you want it back, add its cut-offs to the validation search.

### Compare with the LSTM

```bash
python codes/forecasting/hurdle.py
python - <<'EOF'
import pandas as pd
rows = []
for d in ["Latur","Beed","Osmanabad","Solapur","Ahmednagar"]:
    m = pd.read_csv(f"outputs/lstm/{d}/test_metrics_by_feature.csv", index_col="feature")
    rows.append({"district": d, "LSTM_rain_R2": m.loc["Rainfall_mm","R2"], "LSTM_rain_RMSE": m.loc["Rainfall_mm","RMSE"]})
h = pd.read_csv("outputs/hurdle/hurdle_rainfall_metrics.csv").set_index("district")
print(pd.DataFrame(rows).set_index("district").join(h, rsuffix="_hurdle"))
EOF
```
**Check:** the LSTM's rainfall R² is pooled over all 2022+ target days and all 3 horizons, which is how the script above pools the hurdle numbers. Report the new numbers, not 0.79 to 0.94. Do not present the "+500%" TFT comparison anywhere.

```bash
git add codes/forecasting/hurdle.py outputs/hurdle/hurdle_rainfall_metrics.csv
git commit -m "task 0: leakage-free 3-day hurdle comparison"
git push
```

---

## 3. Task 1: Notebook to scripts

Files to create under `codes/forecasting/`: `common.py`, `plots.py`, `train.py`, `evaluate.py`, `registry.py` (Task 3). Also `params.yaml` at the repo root.

### 3.1 `params.yaml`

```yaml
split: {train_end: "2018-12-31", val_end: "2021-12-31"}
model: {horizon: 3, lookback: 45, lstm_units: 128, num_lstm_layers: 1, dense_units: 384, dropout: 0.25}
train: {learning_rate: 0.0005, batch_size: 16, max_epochs: 80, patience: 12, seed: 42}
log_cols: [Rainfall_mm, Wind_Speed_mps]
districts: [Latur, Beed, Osmanabad, Solapur, Ahmednagar]
nasa: {start_date: "2000-01-01", refresh_days: 60}
monitor: {window_days: 60, min_days: 30, r2_drop_tolerance: 0.10, min_improvement: 0.0}
rolling: {test_days: 365, val_days: 1095}
```

### 3.2 `codes/forecasting/common.py`

Light on purpose (no matplotlib), so CI and Docker stay small.

```python
"""Shared code for training, evaluation, monitoring and serving."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data" / "weather" / "clean"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs" / "lstm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENGINEERED = ["Rain_roll7", "Rain_roll14", "Rain_roll30", "Humid_roll7",
              "Solar_roll7", "Wind_roll7", "Wind50_roll7", "Cloud_roll7"]


def load_params():
    return yaml.safe_load((ROOT / "params.yaml").read_text())


def model_path(district):
    return MODELS_DIR / f"{district.lower()}_weather_lstm_pytorch.pt"


class DirectLSTM(nn.Module):
    def __init__(self, n_input, n_target, horizon, lstm_units, num_layers, dense_units, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_input, lstm_units, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.horizon = horizon
        self.n_target = n_target
        self.drop_lstm = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_units + horizon * 2, dense_units)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_units, dense_units // 2)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(dense_units // 2, horizon * n_target)
        self.act = nn.ReLU()

    def forward(self, seq, cal):
        _, (h, _) = self.lstm(seq)
        enc = self.drop_lstm(h[-1])
        cal_flat = cal.reshape(cal.size(0), -1)
        x = torch.cat([enc, cal_flat], dim=1)
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.act(self.fc2(x)))
        x = self.fc3(x)
        return x.view(-1, self.horizon, self.n_target)


def calendar_feats(index):
    """sin/cos of day-of-year: always known in advance for a future date."""
    index = pd.DatetimeIndex(index)
    doy = np.asarray(index.dayofyear)
    leap = np.asarray(index.is_leap_year)
    frac = doy / np.where(leap, 366, 365)
    return np.stack([np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)], axis=1)


def make_windows(in_arr, out_arr, cal_arr, lookback, horizon):
    Xs, Ys, Cs = [], [], []
    for i in range(len(in_arr) - lookback - horizon + 1):
        Xs.append(in_arr[i:i + lookback])
        Ys.append(out_arr[i + lookback:i + lookback + horizon])
        Cs.append(cal_arr[i + lookback:i + lookback + horizon])
    return (np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32), np.array(Cs, dtype=np.float32))


def make_windows_with_context(in_ctx, out_ctx, cal_ctx, in_arr, out_arr, cal_arr, lookback, horizon):
    """Prepend the tail of the previous split so a split's first `lookback` days aren't lost."""
    full_in = np.concatenate([in_ctx[-lookback:], in_arr], axis=0)
    full_out = np.concatenate([out_ctx[-lookback:], out_arr], axis=0)
    full_cal = np.concatenate([cal_ctx[-lookback:], cal_arr], axis=0)
    return make_windows(full_in, full_out, full_cal, lookback, horizon)


def inverse_target(scaled_seq, scaler, log_cols, feature_names):
    n, h, f = scaled_seq.shape
    inv = scaler.inverse_transform(scaled_seq.reshape(-1, f)).reshape(n, h, f)
    for c in log_cols:
        j = feature_names.index(c)
        inv[:, :, j] = np.expm1(inv[:, :, j])
    return inv


def build_model_frame(df, log_cols):
    """log1p on skewed columns + causal rolling features (past values only). df is indexed by DATE."""
    m = df.copy()
    for c in log_cols:
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


def prepare_data(district, params, train_end=None, val_end=None):
    train_end = train_end or params["split"]["train_end"]
    val_end = val_end or params["split"]["val_end"]
    df = pd.read_csv(CLEAN_DIR / f"{district.lower()}_weather.csv", parse_dates=["DATE"]).set_index("DATE")
    targets = list(df.columns)
    m = build_model_frame(df, params["log_cols"])
    train_df = m.loc[:train_end]
    val_df = m.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1):val_end]
    test_df = m.loc[pd.Timestamp(val_end) + pd.Timedelta(days=1):]
    return {"df": df, "train": train_df, "val": val_df, "test": test_df,
            "targets": targets, "inputs": targets + ENGINEERED,
            "train_end": train_end, "val_end": val_end}


def rolling_dates(last_date, test_days, val_days):
    """Rolling split: test = last `test_days`, val = `val_days` before that, train = the rest."""
    last_date = pd.Timestamp(last_date)
    val_end = last_date - pd.Timedelta(days=test_days)
    train_end = val_end - pd.Timedelta(days=val_days)
    return str(train_end.date()), str(val_end.date())


def evaluate(data, model, in_scaler, out_scaler, log_cols, lookback, horizon):
    """Test-period metrics in original units, pooled over the forecast days."""
    targets, inputs = data["targets"], data["inputs"]
    trainval = pd.concat([data["train"], data["val"]])
    Xtest, Ytest, Ctest = make_windows_with_context(
        in_scaler.transform(trainval[inputs]), out_scaler.transform(trainval[targets]),
        calendar_feats(trainval.index),
        in_scaler.transform(data["test"][inputs]), out_scaler.transform(data["test"][targets]),
        calendar_feats(data["test"].index), lookback, horizon)
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(Xtest).to(dev), torch.from_numpy(Ctest).to(dev)).cpu().numpy()
    y_true = inverse_target(Ytest, out_scaler, log_cols, targets)
    y_pred = inverse_target(pred_scaled, out_scaler, log_cols, targets)
    rows = []
    for j, feat in enumerate(targets):
        a, p = y_true[:, :, j].reshape(-1), y_pred[:, :, j].reshape(-1)
        rows.append({"feature": feat, "RMSE": np.sqrt(mean_squared_error(a, p)),
                     "MAE": mean_absolute_error(a, p), "R2": r2_score(a, p)})
    metrics = pd.DataFrame(rows).set_index("feature").round(3)
    test_dates = trainval.index[-lookback:].append(data["test"].index)
    return metrics, float(metrics["R2"].mean()), y_true, y_pred, test_dates


def load_checkpoint(path):
    """Load a model checkpoint (dict with model_state_dict + model_config)."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    mc = ckpt["model_config"]
    model = DirectLSTM(mc["n_input"], mc["n_target"], mc["horizon"], mc["lstm_units"],
                       mc["num_layers"], mc["dense_units"], mc["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_bundle(bundle_dir):
    """A bundle = model.pt + in_scaler.joblib + out_scaler.joblib + config.json."""
    bundle_dir = Path(bundle_dir)
    cfg = json.loads((bundle_dir / "config.json").read_text())
    model = load_checkpoint(bundle_dir / "model.pt")
    return (model, joblib.load(bundle_dir / "in_scaler.joblib"),
            joblib.load(bundle_dir / "out_scaler.joblib"), cfg)


def forecast_next(df, bundle):
    """Forecast `horizon` days after the last date in df (df indexed by DATE, cleaned)."""
    model, in_scaler, out_scaler, cfg = bundle
    lookback, horizon = cfg["lookback"], cfg["horizon"]
    frame = build_model_frame(df, cfg["log_transformed_columns"])
    if len(frame) < lookback:
        raise ValueError(f"need at least {lookback} days of data, got {len(frame)}")
    X = in_scaler.transform(frame[cfg["input_features"]].tail(lookback))
    origin = frame.index[-1]
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon)
    cal = calendar_feats(dates)
    dev = next(model.parameters()).device
    with torch.no_grad():
        pred = model(torch.tensor(X[None], dtype=torch.float32).to(dev),
                     torch.tensor(cal[None], dtype=torch.float32).to(dev)).cpu().numpy()
    inv = inverse_target(pred, out_scaler, cfg["log_transformed_columns"], cfg["target_features"])[0]
    for c in cfg["log_transformed_columns"]:          # rainfall and wind cannot be negative
        j = cfg["target_features"].index(c)
        inv[:, j] = np.clip(inv[:, j], 0, None)
    return origin, dates, cfg["target_features"], inv
```

### 3.3 `codes/forecasting/plots.py`

Take section 6 of the notebook (`finish`, `plot_loss`, `plot_evaluation`) and paste the three functions **unchanged** below this header:

```python
import matplotlib
matplotlib.use("Agg")                      # no display in WSL / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import OUTPUT_DIR, load_params

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
SHOW_PLOTS = False
_P = load_params()
HORIZON = _P["model"]["horizon"]
LOOKBACK = _P["model"]["lookback"]

# --- paste finish(), plot_loss(), plot_evaluation() from the notebook here, unchanged ---
```

### 3.4 `codes/forecasting/train.py`

This version already includes MLflow (Task 3). `registry.py` comes in section 5. If you want to test Task 1 alone first, run it with `--no-mlflow`.

```python
"""Train district LSTMs. Logs to MLflow and registers a model version unless --no-mlflow."""
import argparse
import copy
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

import registry
from common import (DEVICE, MODELS_DIR, OUTPUT_DIR, DirectLSTM, calendar_feats, evaluate, load_params,
                    make_windows, make_windows_with_context, model_path, prepare_data)


def train_model(district, data, params):
    t, m = params["train"], params["model"]
    np.random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    targets, inputs = data["targets"], data["inputs"]
    train_df, val_df = data["train"], data["val"]

    in_scaler, out_scaler = MinMaxScaler(), MinMaxScaler()
    train_in = in_scaler.fit_transform(train_df[inputs])
    val_in = in_scaler.transform(val_df[inputs])
    train_out = out_scaler.fit_transform(train_df[targets])
    val_out = out_scaler.transform(val_df[targets])
    cal_train, cal_val = calendar_feats(train_df.index), calendar_feats(val_df.index)

    Xtr, Ytr, Ctr = make_windows(train_in, train_out, cal_train, m["lookback"], m["horizon"])
    Xval, Yval, Cval = make_windows_with_context(train_in, train_out, cal_train, val_in, val_out, cal_val,
                                                 m["lookback"], m["horizon"])
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ctr), torch.from_numpy(Ytr)),
                        batch_size=t["batch_size"], shuffle=True)
    Xval_t, Cval_t, Yval_t = (torch.from_numpy(a).to(DEVICE) for a in (Xval, Cval, Yval))

    model = DirectLSTM(len(inputs), len(targets), m["horizon"], m["lstm_units"], m["num_lstm_layers"],
                       m["dense_units"], m["dropout"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    loss_fn = nn.HuberLoss(delta=1.0)

    history = {"loss": [], "val_loss": []}
    best_val, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(t["max_epochs"]):
        model.train()
        running, n_batches = 0.0, 0
        for xb, cb, yb in loader:
            xb, cb, yb = xb.to(DEVICE), cb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, cb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_t, Cval_t), Yval_t).item()
        scheduler.step(val_loss)
        history["loss"].append(running / n_batches)
        history["val_loss"].append(val_loss)
        if val_loss < best_val - 1e-5:
            best_val, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= t["patience"]:
                break
    model.load_state_dict(best_state)
    print(f"[{district}] {len(history['loss'])} epochs, best val loss {best_val:.5f}")
    return model, in_scaler, out_scaler, history


def write_bundle(bundle_dir, district, data, params, model, in_scaler, out_scaler, metrics, blended):
    m, t = params["model"], params["train"]
    bundle_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "model_config": {"n_input": len(data["inputs"]), "n_target": len(data["targets"]),
                                 "horizon": m["horizon"], "lstm_units": m["lstm_units"],
                                 "num_layers": m["num_lstm_layers"], "dense_units": m["dense_units"],
                                 "dropout": m["dropout"]}}, bundle_dir / "model.pt")
    joblib.dump(in_scaler, bundle_dir / "in_scaler.joblib")
    joblib.dump(out_scaler, bundle_dir / "out_scaler.joblib")
    metrics.to_csv(bundle_dir / "test_metrics_by_feature.csv")
    config = {
        "district": district, "target_features": data["targets"], "input_features": data["inputs"],
        "log_transformed_columns": params["log_cols"], "lookback": m["lookback"], "horizon": m["horizon"],
        "lstm_units": m["lstm_units"], "num_layers": m["num_lstm_layers"], "dense_units": m["dense_units"],
        "dropout": m["dropout"], "learning_rate": t["learning_rate"], "batch_size": t["batch_size"],
        "train_end": data["train_end"], "val_end": data["val_end"], "blended_test_r2": blended,
        "trained_through_date": str(data["df"].index.max().date()),
    }
    (bundle_dir / "config.json").write_text(json.dumps(config, indent=2))


def log_run(district, params, data, history, metrics, blended, bundle_dir):
    mlflow.set_experiment("drought-weather-lstm")
    with mlflow.start_run(run_name=f"{district}_{datetime.utcnow():%Y%m%d_%H%M}") as run:
        flat = {f"model.{k}": v for k, v in params["model"].items()}
        flat.update({f"train.{k}": v for k, v in params["train"].items()})
        flat.update({"district": district, "train_end": data["train_end"], "val_end": data["val_end"]})
        mlflow.log_params(flat)
        for epoch, (tl, vl) in enumerate(zip(history["loss"], history["val_loss"])):
            mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)
        mlflow.log_metric("best_val_loss", min(history["val_loss"]))
        for feat, row in metrics.iterrows():
            mlflow.log_metrics({f"test_r2_{feat}": float(row["R2"]), f"test_rmse_{feat}": float(row["RMSE"])})
        mlflow.log_metric("blended_test_r2", blended)
        mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")
        return run.info.run_id


def run(district, params, train_end=None, val_end=None, promote=False, save_local=False, use_mlflow=True):
    data = prepare_data(district, params, train_end, val_end)
    model, in_scaler, out_scaler, history = train_model(district, data, params)
    m = params["model"]
    metrics, blended, *_ = evaluate(data, model, in_scaler, out_scaler, params["log_cols"],
                                    m["lookback"], m["horizon"])
    print(metrics, f"\n[{district}] blended test R2 = {blended:.4f} (test: {data['test'].index.min().date()} "
                    f"to {data['test'].index.max().date()})")
    result = {"district": district, "blended_r2": blended, "version": None, "run_id": None}
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        write_bundle(bundle, district, data, params, model, in_scaler, out_scaler, metrics, blended)
        if save_local:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            out_dir = OUTPUT_DIR / district
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(bundle / "model.pt", model_path(district))
            for f in ("in_scaler.joblib", "out_scaler.joblib", "config.json", "test_metrics_by_feature.csv"):
                shutil.copy(bundle / f, out_dir / f)
        if use_mlflow:
            registry.setup()
            result["run_id"] = log_run(district, params, data, history, metrics, blended, bundle)
    if use_mlflow:
        mv = registry.register(district, result["run_id"])
        result["version"] = mv.version
        name = f"{district}_lstm"
        if promote or registry.get_champion(name) is None:
            registry.set_champion(name, mv.version)
            print(f"[{district}] version {mv.version} is now champion")
        else:
            print(f"[{district}] version {mv.version} registered (not champion)")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--district", default=None, help="one district; default = all in params.yaml")
    p.add_argument("--train-end"); p.add_argument("--val-end")
    p.add_argument("--promote", action="store_true", help="force this version to become champion")
    p.add_argument("--save-local", action="store_true", help="also write models/ and outputs/lstm/<District>/")
    p.add_argument("--no-mlflow", action="store_true")
    a = p.parse_args()
    params = load_params()
    for d in ([a.district] if a.district else params["districts"]):
        run(d, params, a.train_end, a.val_end, a.promote, a.save_local, not a.no_mlflow)


if __name__ == "__main__":
    main()
```

### 3.5 `codes/forecasting/evaluate.py`

Re-evaluates the **saved local** models (same job as the notebook with `RUN_TRAINING = False`).

```python
"""Re-evaluate the locally saved models (models/*.pt + outputs/lstm/<District>/*)."""
import argparse

import joblib
import pandas as pd

from common import OUTPUT_DIR, evaluate, load_checkpoint, load_params, model_path, prepare_data
from plots import plot_evaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--district", default=None)
    a = p.parse_args()
    params = load_params()
    m = params["model"]
    summary = {}
    for d in ([a.district] if a.district else params["districts"]):
        data = prepare_data(d, params)
        out_dir = OUTPUT_DIR / d
        out_dir.mkdir(parents=True, exist_ok=True)
        model = load_checkpoint(model_path(d))
        in_s, out_s = joblib.load(out_dir / "in_scaler.joblib"), joblib.load(out_dir / "out_scaler.joblib")
        metrics, blended, y_true, y_pred, test_dates = evaluate(
            data, model, in_s, out_s, params["log_cols"], m["lookback"], m["horizon"])
        metrics.to_csv(out_dir / "test_metrics_by_feature.csv")
        plot_evaluation(d, data["targets"], metrics, blended, y_true, y_pred, test_dates)
        summary[d] = metrics["R2"].rename(d)
        print(f"[{d}] blended test R2 = {blended:.4f}")
    table = pd.DataFrame(summary)
    table.loc["BLENDED"] = table.mean()
    print(table.round(3))


if __name__ == "__main__":
    main()
```

### 3.6 Check, then commit

```bash
python codes/forecasting/evaluate.py
diff -r /tmp/lstm_baseline outputs/lstm -x "*.png" && echo "METRICS IDENTICAL"
```
Expected: `METRICS IDENTICAL` for the `.csv` files (same weights, same data, same code). If they differ, stop and send me the diff.

Optional training smoke test (about 10-15 min, no MLflow yet, nothing saved locally):
```bash
python codes/forecasting/train.py --district Latur --no-mlflow
```
(`train.py` imports `registry`, so create `registry.py` from section 5 first. The import needs `mlflow` installed, which A5 did.)

```bash
git add params.yaml codes/forecasting/common.py codes/forecasting/plots.py \
        codes/forecasting/train.py codes/forecasting/evaluate.py requirements.txt .gitignore
git commit -m "task 1: lstm notebook -> scripts, params.yaml"
git push
```
The notebook `lstm.ipynb` can stay in the repo as a reference.

---

## 4. Task 2: DVC with the DagsHub remote

Use the DVC URL and token from A8.

```bash
git checkout vaish_mlops
dvc init
dvc remote add -d origin https://dagshub.com/<USER>/<REPO>.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <USER>
dvc remote modify origin --local password <TOKEN>
```
The `--local` options write to `.dvc/config.local`, which DVC keeps out of git. Never put the token in `.dvc/config`.

Decide what to version (check A4 output for the real paths):

```bash
# stop git tracking the big files; --ignore-unmatch makes this safe if they were already untracked
git rm -r --cached --ignore-unmatch models data/weather outputs/lstm
dvc add data/weather models outputs/lstm
dvc push
git add .dvc .dvcignore data/weather.dvc models.dvc outputs/lstm.dvc data/.gitignore .gitignore
git status                       # check: no .pt / large csv / .env staged
git commit -m "task 2: version data/weather, models and lstm outputs with DVC"
git push
```
`dvc add` prints any `.gitignore` files it modified; add exactly those to git. If `outputs/lstm` does not exist (A4 shows the real folder), leave it out of both commands. `outputs/lstm` is included because the API needs the scalers and `config.json` that live there (my suggestion; it is not in your task list).

**Check (proves the remote works):**
```bash
rm -rf models
dvc pull models.dvc
ls models            # the 5 .pt files are back
dvc status           # "Data and pipelines are up to date."
```

---

## 5. Task 3: MLflow tracking and model registry

### 5.1 `codes/forecasting/registry.py`

```python
"""MLflow registry helpers: register, mark/read the champion, download a bundle."""
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ALIAS = "champion"


def setup():
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("Set MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD "
                           "(run: set -a; source .env; set +a)")
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


def register(district, run_id):
    return mlflow.register_model(f"runs:/{run_id}/bundle", f"{district}_lstm")


def set_champion(name, version):
    c = setup()
    version = str(version)
    try:
        c.set_registered_model_alias(name, ALIAS, version)
    except Exception as e:                                   # server without alias support
        print(f"alias not set ({e}); using version tags only")
    for mv in c.search_model_versions(f"name='{name}'"):
        c.set_model_version_tag(name, mv.version, ALIAS, "true" if str(mv.version) == version else "false")


def get_champion(name):
    c = setup()
    try:
        return c.get_model_version_by_alias(name, ALIAS)
    except Exception:
        pass
    try:
        versions = c.search_model_versions(f"name='{name}'")
    except Exception:
        return None
    tagged = [mv for mv in versions if mv.tags.get(ALIAS) == "true"]
    return max(tagged, key=lambda mv: int(mv.version)) if tagged else None


def download_bundle(mv, dst):
    setup()
    return Path(mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{mv.run_id}/bundle", dst_path=str(dst)))


def champion_metric(mv, key="blended_test_r2"):
    return setup().get_run(mv.run_id).data.metrics.get(key)
```

### 5.2 Register the models you already have as version 1 (no retraining)

Retraining all five districts would take over an hour and change the weights. Instead, log the existing files as version 1 and mark them champion.

`codes/forecasting/register_existing.py`:

```python
"""Log the already-trained models (models/*.pt + outputs/lstm/<District>/*) as version 1 and mark champion."""
import json
import shutil
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

import registry
from common import OUTPUT_DIR, load_params, model_path


def main():
    params = load_params()
    registry.setup()
    mlflow.set_experiment("drought-weather-lstm")
    for d in params["districts"]:
        src = OUTPUT_DIR / d
        cfg = json.loads((src / "config.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir()
            shutil.copy(model_path(d), bundle / "model.pt")
            for f in ("in_scaler.joblib", "out_scaler.joblib", "config.json", "test_metrics_by_feature.csv"):
                if (src / f).exists():
                    shutil.copy(src / f, bundle / f)
            with mlflow.start_run(run_name=f"{d}_imported_baseline") as run:
                mlflow.log_params({k: v for k, v in cfg.items()
                                   if k not in ("target_features", "input_features", "log_transformed_columns")})
                mlflow.set_tag("origin", "imported from notebook-trained model")
                metrics_csv = src / "test_metrics_by_feature.csv"
                if metrics_csv.exists():
                    for feat, row in pd.read_csv(metrics_csv, index_col="feature").iterrows():
                        mlflow.log_metrics({f"test_r2_{feat}": float(row["R2"]), f"test_rmse_{feat}": float(row["RMSE"])})
                mlflow.log_metric("blended_test_r2", float(cfg["blended_test_r2"]))
                mlflow.log_artifacts(str(bundle), artifact_path="bundle")
                run_id = run.info.run_id
        mv = registry.register(d, run_id)
        registry.set_champion(f"{d}_lstm", mv.version)
        print(f"{d}: registered {d}_lstm v{mv.version} as champion")


if __name__ == "__main__":
    main()
```

Run it:
```bash
set -a; source .env; set +a
python codes/forecasting/register_existing.py
```

**Check:**
- DagsHub repo → **Experiments** tab: 5 runs with `blended_test_r2` and per-variable `test_r2_*` metrics.
- MLflow UI (link in the repo's Remote panel) → **Models**: `Latur_lstm`, `Beed_lstm`, … each with version 1 and the `champion` alias (or tag `champion=true`).
- In Python:
```bash
python - <<'EOF'
import sys; sys.path.insert(0, "codes/forecasting")
import registry
mv = registry.get_champion("Latur_lstm")
print("champion version:", mv.version, "| run:", mv.run_id)
EOF
```

**If `register_model` fails** (unverified point #2): in `registry.register`, replace the body with a `mlflow.pytorch.log_model` flow and tell me the error text. I will adapt it.

### 5.3 Test the "register but don't promote" path

```bash
python codes/forecasting/train.py --district Latur            # ~10-15 min
```
Expected final line: `version 2 registered (not champion)`. Version 1 stays champion. This is the gate used by Task 6.

```bash
git add codes/forecasting/registry.py codes/forecasting/register_existing.py
git commit -m "task 3: MLflow tracking and champion registry"
git push
```

---

## 6. Task 4: FastAPI endpoint and Docker

### 6.1 `serve/__init__.py` (empty) and `serve/app.py`

```bash
mkdir -p serve logs && touch serve/__init__.py
```

```python
"""GET /forecast/{district}: loads the champion model and returns the next 3 days."""
import os
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes" / "forecasting"))

import common  # noqa: E402
import registry  # noqa: E402

LOG_PATH = Path(os.getenv("FORECAST_LOG", ROOT / "logs" / "forecasts_log.csv"))
DISTRICTS = {d.lower(): d for d in common.load_params()["districts"]}
_cache = {}                       # district -> (version, bundle)
_lock = threading.Lock()
app = FastAPI(title="Drought DSS weather forecast API")


def get_bundle(district):
    mv = registry.get_champion(f"{district}_lstm")
    if mv is None:
        raise HTTPException(503, f"no champion model registered for {district}")
    cached = _cache.get(district)
    if cached and cached[0] == mv.version:
        return mv, cached[1]
    tmp = tempfile.mkdtemp(prefix=f"{district}_")
    bundle = common.load_bundle(registry.download_bundle(mv, tmp))
    _cache[district] = (mv.version, bundle)
    return mv, bundle


def append_log(rows):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    with _lock:
        df.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast/{district}")
def forecast(district: str):
    name = DISTRICTS.get(district.lower())
    if name is None:
        raise HTTPException(404, f"unknown district; choose from {sorted(DISTRICTS.values())}")
    csv = common.CLEAN_DIR / f"{name.lower()}_weather.csv"
    if not csv.exists():
        raise HTTPException(503, f"missing data file {csv.name}")
    df = pd.read_csv(csv, parse_dates=["DATE"]).set_index("DATE")
    registry.setup()
    mv, bundle = get_bundle(name)
    origin, dates, variables, values = common.forecast_next(df, bundle)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, out = [], []
    for i, (d, vals) in enumerate(zip(dates, values), start=1):
        vals_d = {v: round(float(x), 3) for v, x in zip(variables, vals)}
        rows.append({"issued_at_utc": now, "district": name, "model_version": str(mv.version),
                     "origin_date": origin.date().isoformat(), "target_date": d.date().isoformat(),
                     "horizon": i, **vals_d})
        out.append({"date": d.date().isoformat(), "horizon": i, **vals_d})
    append_log(rows)
    return {"district": name, "model_version": str(mv.version), "origin_date": origin.date().isoformat(),
            "data_age_days": (datetime.now(timezone.utc).date() - origin.date()).days, "forecast": out}
```
`data_age_days` shows how stale the input data is (see the NASA lag from A7).

### 6.2 Run locally first

```bash
set -a; source .env; set +a
uvicorn serve.app:app --reload --port 8000
# second terminal:
curl http://localhost:8000/health
curl http://localhost:8000/forecast/Latur
cat logs/forecasts_log.csv | head -4
```
**Check:** the response has 3 forecast days starting the day after `origin_date`, and `logs/forecasts_log.csv` gained 3 rows.

### 6.3 Docker

Pin the same versions as your venv:
```bash
pip freeze | grep -iE "^(fastapi|uvicorn|pandas|numpy|scikit-learn|joblib|mlflow|pyyaml)==" > requirements-serve.txt
python3 --version
```

`Dockerfile` (change `3.11` to your Python minor version):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY params.yaml .
COPY codes/forecasting/common.py codes/forecasting/registry.py codes/forecasting/
COPY serve serve
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

`.dockerignore`:
```
.git
.venv
.env
data
models
outputs
logs
mlruns
__pycache__
```

Install Docker Desktop for Windows with "WSL integration" enabled for your Ubuntu distro (Settings → Resources → WSL integration), then:
```bash
docker build -t drought-api .
docker run --rm -p 8000:8000 --env-file .env \
  -v "$(pwd)/data/weather/clean:/app/data/weather/clean:ro" \
  -v "$(pwd)/logs:/app/logs" \
  drought-api
curl http://localhost:8000/forecast/Latur
```
The data and the log are mounted, not baked in: the image holds code only, the champion model comes from MLflow at request time, the data comes from DVC, and the log survives container restarts. Credentials are passed at run time and never stored in the image.

```bash
git add serve Dockerfile .dockerignore requirements-serve.txt
git commit -m "task 4: FastAPI forecast endpoint + Docker"
git push
```

---

## 7. Task 5: GitHub Actions (tests on push, weekly NASA fetch)

### 7.1 Tests

```bash
mkdir -p tests
```

`tests/conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes" / "forecasting"))
```

`tests/test_core.py` (synthetic data only, so CI needs no DVC):
```python
import numpy as np
import pandas as pd
import torch

from common import DirectLSTM, build_model_frame, calendar_feats, make_windows, rolling_dates


def test_model_output_shape():
    m = DirectLSTM(n_input=19, n_target=11, horizon=3, lstm_units=16, num_layers=1, dense_units=32, dropout=0.1)
    out = m(torch.randn(4, 45, 19), torch.randn(4, 3, 2))
    assert out.shape == (4, 3, 11)


def test_windows_target_is_after_inputs():
    arr = np.arange(20, dtype=np.float32).reshape(-1, 1)
    cal = np.zeros((20, 2), dtype=np.float32)
    X, Y, C = make_windows(arr, arr, cal, lookback=5, horizon=3)
    assert len(X) == 20 - 5 - 3 + 1
    assert (Y[:, 0, 0] > X[:, -1, 0]).all()            # first target day is after the last input day
    assert (Y[:, 0, 0] - X[:, -1, 0] == 1).all()       # and directly adjacent


def test_calendar_feats_are_unit_circle():
    c = calendar_feats(pd.date_range("2024-01-01", periods=400))
    assert c.shape == (400, 2)
    assert np.allclose(c[:, 0] ** 2 + c[:, 1] ** 2, 1.0)


def test_rolling_features_are_causal():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=100)
    cols = ["Rainfall_mm", "Relative_Humidity_percent", "Solar_Radiation",
            "Wind_Speed_mps", "Wind_Speed_50m_mps", "Cloud_Amount_percent"]
    df = pd.DataFrame(rng.uniform(0, 10, (100, len(cols))), index=idx, columns=cols)
    df2 = df.copy()
    df2.iloc[-1] = df2.iloc[-1] * 10                  # change only the LAST day
    f1 = build_model_frame(df, ["Rainfall_mm", "Wind_Speed_mps"])
    f2 = build_model_frame(df2, ["Rainfall_mm", "Wind_Speed_mps"])
    pd.testing.assert_frame_equal(f1.iloc[:-1], f2.iloc[:-1])    # earlier rows must not change


def test_rolling_split_dates():
    train_end, val_end = rolling_dates("2025-12-31", test_days=365, val_days=1095)
    assert pd.Timestamp(train_end) < pd.Timestamp(val_end) < pd.Timestamp("2025-12-31")
    assert (pd.Timestamp("2025-12-31") - pd.Timestamp(val_end)).days == 365
```
Run locally:
```bash
pytest -q
```
**Check:** 5 passed.

### 7.2 Data pipeline scripts

```bash
mkdir -p codes/pipeline
```

`codes/pipeline/fetch_nasa.py`: re-pulls the last `refresh_days` days (60) and merges them over the stored data. Same API, parameters and column names as `preprocessing.ipynb` step 2.

```python
"""Weekly NASA POWER refresh. Reads/writes data/weather/live/<district>_weather.csv."""
import shutil
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting"))
from common import ROOT, load_params  # noqa: E402

WEATHER = ROOT / "data" / "weather"
LIVE = WEATHER / "live"
SEED_VERSION = "v2"           # live/ starts as a copy of this folder, so v2 is never modified

DISTRICTS = {
    "Beed": (18.9891, 75.7601), "Latur": (18.4088, 76.5604), "Osmanabad": (18.1860, 76.0419),
    "Solapur": (17.6599, 75.9064), "Ahmednagar": (19.0948, 74.7480),
}
NASA_PARAMS = {
    "PRECTOTCORR": "Rainfall_mm", "T2M": "Temperature_C", "T2M_MAX": "Temperature_Max_C",
    "T2M_MIN": "Temperature_Min_C", "RH2M": "Relative_Humidity_percent", "WS10M": "Wind_Speed_mps",
    "PS": "Surface_Pressure_kPa", "ALLSKY_SFC_SW_DWN": "Solar_Radiation", "T2MDEW": "Dew_Point_C",
    "CLOUD_AMT": "Cloud_Amount_percent", "WS50M": "Wind_Speed_50m_mps",
}
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def get_with_retry(params, retries=4):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(URL, params=params, timeout=120)
            if r.status_code == 422:
                raise ValueError(f"request rejected (422): {r.text[:300]}")
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries:
                raise
            wait = 5 * 2 ** (attempt - 1)
            print(f"  attempt {attempt} failed ({e}); retrying in {wait}s")
            time.sleep(wait)


def fetch(lat, lon, start, end):
    r = get_with_retry({"parameters": ",".join(NASA_PARAMS), "community": "AG", "longitude": lon,
                        "latitude": lat, "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
                        "format": "JSON", "time-standard": "LST"})
    df = pd.DataFrame(r.json()["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "DATE"
    df = df.reset_index().replace(-999, pd.NA).rename(columns=NASA_PARAMS)
    values = [c for c in df.columns if c != "DATE"]
    return df.dropna(subset=values, how="all")          # drop days NASA has not processed yet


def main():
    refresh = load_params()["nasa"]["refresh_days"]
    start_floor = date.fromisoformat(load_params()["nasa"]["start_date"])
    LIVE.mkdir(parents=True, exist_ok=True)
    for name, (lat, lon) in DISTRICTS.items():
        path = LIVE / f"{name.lower()}_weather.csv"
        if not path.exists():
            shutil.copy(WEATHER / SEED_VERSION / path.name, path)
        old = pd.read_csv(path, parse_dates=["DATE"])
        start = max(start_floor, old["DATE"].max().date() - timedelta(days=refresh))
        new = fetch(lat, lon, start, date.today())
        combined = pd.concat([old[old["DATE"] < pd.Timestamp(start)], new], ignore_index=True)
        combined = combined.drop_duplicates("DATE", keep="last").sort_values("DATE")[list(old.columns)]
        combined.to_csv(path, index=False)
        print(f"{name}: {start} -> {combined['DATE'].max().date()}  rows={len(combined)}")
        time.sleep(2)


if __name__ == "__main__":
    main()
```

`codes/pipeline/clean.py`: the notebook's cleaning step, parameterised by source folder.

```python
"""Gap-fill raw weather -> data/weather/clean/. Same logic as preprocessing.ipynb step 5."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecasting"))
from common import ROOT, load_params  # noqa: E402

CLIM_COL = "Cloud_Amount_percent"


def load_raw(csv_path):
    raw = pd.read_csv(csv_path, parse_dates=["DATE"]).sort_values("DATE").reset_index(drop=True)
    features = [c for c in raw.columns if c != "DATE"]
    df = raw.copy()
    all_nan = df[features].isna().all(axis=1)
    if all_nan.any():
        df = df.loc[:df.index[~all_nan].max()].reset_index(drop=True)
    return df.set_index("DATE"), features


def fill_missing(df, features, train_end):
    df = df.copy()
    train_slice = df.loc[:train_end, CLIM_COL]
    clim = train_slice.groupby(train_slice.index.dayofyear).mean()
    clim_ext = pd.concat([clim, clim, clim])
    clim_smooth = clim_ext.rolling(7, center=True, min_periods=1).mean().iloc[len(clim):2 * len(clim)]
    clim_smooth.index = clim.index
    fill_values = df.index.dayofyear.map(clim_smooth)
    mask = df[CLIM_COL].isna()
    df.loc[mask, CLIM_COL] = fill_values[mask.values]
    other = [c for c in features if c != CLIM_COL]
    df[other] = df[other].interpolate(method="time", limit_direction="both")
    assert df.isna().sum().sum() == 0, "NaNs remain after cleaning"
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="v2", help="folder under data/weather/ (v2 or live)")
    a = p.parse_args()
    params = load_params()
    out = ROOT / "data" / "weather" / "clean"
    out.mkdir(parents=True, exist_ok=True)
    for name in params["districts"]:
        df, features = load_raw(ROOT / "data" / "weather" / a.source / f"{name.lower()}_weather.csv")
        n_missing = int(df.isna().sum().sum())
        df = fill_missing(df, features, params["split"]["train_end"])
        df.reset_index().to_csv(out / f"{name.lower()}_weather.csv", index=False)
        print(f"{name}: {len(df)} days, last={df.index.max().date()}, filled={n_missing}")


if __name__ == "__main__":
    main()
```

**Check (proves clean.py reproduces the notebook):**
```bash
python codes/pipeline/clean.py --source v2
diff -r /tmp/clean_baseline data/weather/clean && echo "CLEAN IDENTICAL"
```
**Check the fetch** (needs internet, takes about a minute):
```bash
python codes/pipeline/fetch_nasa.py
python codes/pipeline/clean.py --source live
tail -n 2 data/weather/clean/latur_weather.csv
```
The last date should match the NASA "last real day" from A7. Since this command changed `data/weather/clean`, run `dvc add data/weather` again afterwards if you want to keep the new data version; otherwise restore it with `dvc checkout data/weather.dvc`.

### 7.3 GitHub secrets

GitHub repo → Settings → Secrets and variables → Actions → New repository secret:
`DAGSHUB_USER`, `DAGSHUB_TOKEN`, `MLFLOW_TRACKING_URI`.

### 7.4 Workflows

`.github/workflows/ci.yml`:
```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"        # same minor version as your WSL Python
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install numpy pandas scikit-learn joblib pyyaml pytest
      - run: pytest -q
```

`.github/workflows/weekly-data.yml`:
```yaml
name: weekly-data
on:
  schedule:
    - cron: "0 3 * * 1"          # Mondays 03:00 UTC
  workflow_dispatch:
permissions:
  contents: write
jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install dvc pandas numpy pyyaml requests torch scikit-learn joblib --extra-index-url https://download.pytorch.org/whl/cpu
      - name: configure DVC remote credentials
        run: |
          dvc remote modify origin --local auth basic
          dvc remote modify origin --local user ${{ secrets.DAGSHUB_USER }}
          dvc remote modify origin --local password ${{ secrets.DAGSHUB_TOKEN }}
      - run: dvc pull data/weather.dvc
      - run: python codes/pipeline/fetch_nasa.py
      - run: python codes/pipeline/clean.py --source live
      - run: dvc add data/weather
      - run: dvc push
      - name: commit the updated DVC pointer
        run: |
          git config user.name "github-actions"
          git config user.email "github-actions@users.noreply.github.com"
          git add data/weather.dvc
          git diff --cached --quiet || (git commit -m "weekly data refresh $(date +%F)" && git push)
```
(`common.py` imports torch and sklearn, hence they are installed here although the fetch/clean scripts do not train anything.)

Important GitHub behaviours:
- Scheduled workflows run only from the **default branch** (`main`), and the "Run workflow" button appears only once the file is on `main`. Until you merge `vaish_mlops` (section 10), test the same steps by running the commands by hand.
- The `git push` at the end fails if `main` has branch protection. Then change that step to open a pull request instead.
- The `ci.yml` workflow runs on pushes to every branch, including `vaish_mlops`.

```bash
git add tests codes/pipeline .github
git commit -m "task 5: pytest CI, weekly NASA refresh"
git push
```
**Check:** GitHub → Actions tab → the `ci` run for your push is green.

---

## 8. Task 6: Weekly monitoring, retraining, gated promotion

### 8.1 `codes/forecasting/monitor.py`

```python
"""Weekly: rolling RMSE/R2 of logged forecasts vs actuals; retrain on trigger; promote only if better."""
import argparse
import os
import tempfile
from datetime import date

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

import registry
import train
from common import (CLEAN_DIR, ROOT, evaluate, load_bundle, load_params, prepare_data, rolling_dates)

LOG_PATH = ROOT / "logs" / "forecasts_log.csv"
METRICS_PATH = ROOT / "logs" / "monitor_metrics.csv"


def rolling_metrics(log, actuals, variables, window_days):
    """Compare logged forecasts with actuals over the last `window_days` target dates that have actuals."""
    log = log.copy()
    log["target_date"] = pd.to_datetime(log["target_date"])
    log = log.drop_duplicates(["origin_date", "target_date"], keep="last")
    last_actual = actuals.index.max()
    log = log[(log["target_date"] <= last_actual) &
              (log["target_date"] > last_actual - pd.Timedelta(days=window_days))]
    rows = []
    for v in variables:
        m = log[["target_date", v]].merge(
            actuals[[v]].rename(columns={v: "actual"}), left_on="target_date", right_index=True).dropna()
        if len(m) < 2 or np.var(m["actual"]) == 0:          # R2 is undefined without variance
            rows.append({"feature": v, "n": len(m), "RMSE": np.nan, "R2": np.nan})
            continue
        rows.append({"feature": v, "n": len(m),
                     "RMSE": float(np.sqrt(mean_squared_error(m["actual"], m[v]))),
                     "R2": float(r2_score(m["actual"], m[v]))})
    table = pd.DataFrame(rows).set_index("feature")
    return table, float(table["R2"].mean()), int(log["target_date"].nunique())


def challenge(district, params, dry_run=False):
    """Train on the rolling split, then compare with the champion on the SAME test period."""
    name = f"{district}_lstm"
    champ = registry.get_champion(name)
    last = pd.read_csv(CLEAN_DIR / f"{district.lower()}_weather.csv", parse_dates=["DATE"])["DATE"].max()
    train_end, val_end = rolling_dates(last, params["rolling"]["test_days"], params["rolling"]["val_days"])
    print(f"[{district}] rolling split: train<= {train_end}, val<= {val_end}, test = rest (to {last.date()})")
    if dry_run:
        return
    res = train.run(district, params, train_end, val_end, promote=False)
    data = prepare_data(district, params, train_end, val_end)
    m = params["model"]
    with tempfile.TemporaryDirectory() as tmp:
        model, in_s, out_s, cfg = load_bundle(registry.download_bundle(champ, tmp))
    if cfg["input_features"] != data["inputs"] or cfg["target_features"] != data["targets"]:
        raise RuntimeError("champion was trained on different features; cannot compare on the same test period")
    _, champ_r2, *_ = evaluate(data, model, in_s, out_s, params["log_cols"], m["lookback"], m["horizon"])
    new_r2 = res["blended_r2"]
    print(f"[{district}] same test period -> champion v{champ.version}: {champ_r2:.4f} | "
          f"challenger v{res['version']}: {new_r2:.4f}")
    if new_r2 > champ_r2 + params["monitor"]["min_improvement"]:
        registry.set_champion(name, res["version"])
        print(f"[{district}] PROMOTED v{res['version']}")
    else:
        print(f"[{district}] challenger rejected; champion v{champ.version} stays")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retrain", action="store_true", help="retrain districts whose rolling R2 is below the trigger")
    p.add_argument("--force", action="store_true", help="treat every district as triggered (testing)")
    p.add_argument("--dry-run", action="store_true", help="show the rolling split only, train nothing")
    a = p.parse_args()
    params, mon = load_params(), load_params()["monitor"]
    registry.setup()
    if not LOG_PATH.exists():
        raise SystemExit(f"{LOG_PATH} not found: nothing to monitor yet")
    log_all = pd.read_csv(LOG_PATH)
    mlflow.set_experiment("drought-monitoring")
    history = []
    for d in params["districts"]:
        champ = registry.get_champion(f"{d}_lstm")
        if champ is None:
            print(f"[{d}] no champion; skipped")
            continue
        log = log_all[(log_all["district"] == d) & (log_all["model_version"].astype(str) == str(champ.version))]
        actuals = pd.read_csv(CLEAN_DIR / f"{d.lower()}_weather.csv", parse_dates=["DATE"]).set_index("DATE")
        variables = [c for c in actuals.columns if c in log.columns]
        table, r2, n_days = rolling_metrics(log, actuals, variables, mon["window_days"])
        base = registry.champion_metric(champ)
        trigger = None if base is None else base - mon["r2_drop_tolerance"]
        print(f"[{d}] champion v{champ.version}: rolling R2={r2:.3f} over {n_days} days "
              f"(trigger below {trigger if trigger is None else round(trigger, 3)})")
        history.append({"date": date.today().isoformat(), "district": d, "champion_version": champ.version,
                        "days": n_days, "rolling_r2": r2, "trigger_below": trigger,
                        **{f"rmse_{v}": table.loc[v, "RMSE"] for v in table.index}})
        with mlflow.start_run(run_name=f"monitor_{d}_{date.today()}"):
            mlflow.log_params({"district": d, "champion_version": champ.version, "window_days": mon["window_days"]})
            if not np.isnan(r2):
                mlflow.log_metric("rolling_blended_r2", r2)
            mlflow.log_metric("days_scored", n_days)
        if n_days < mon["min_days"]:
            print(f"[{d}] only {n_days} scored days (< {mon['min_days']}); not enough evidence to act")
            continue
        if a.force or (trigger is not None and r2 < trigger):
            if a.retrain or a.dry_run:
                challenge(d, params, dry_run=a.dry_run)
            else:
                print(f"[{d}] TRIGGERED (run with --retrain to act)")
    if history:
        pd.DataFrame(history).to_csv(METRICS_PATH, mode="a", header=not METRICS_PATH.exists(), index=False)


if __name__ == "__main__":
    main()
```

**Notes on the logic (so you can defend it):**
- Only forecasts made by the **current champion version** are scored, so versions never mix.
- A forecast can be scored only after its target date's actual weather has arrived from NASA (the lag from A7). Newer forecasts are ignored until then.
- Rolling R² over 60 days is noisy and seasonal (a dry-season window has no rainfall variance, so rainfall is skipped). That is why a minimum of 30 scored days and a tolerance of 0.10 below the champion's own test R² are used, instead of a fixed absolute threshold. Tune both in `params.yaml` after you see real monitor numbers.
- The comparison uses one shared test period (the last 365 days), so the champion and the challenger are scored on identical data. The champion never trained on it because its training ended earlier.

### 8.2 Test it

```bash
set -a; source .env; set +a
python codes/forecasting/monitor.py                      # reports only
python codes/forecasting/monitor.py --force --dry-run    # prints the rolling split, trains nothing
```
**Check:** after you have called the API for a while (and NASA has caught up), the first command prints a rolling R² per district. Right after Task 4, expect "only N scored days; not enough evidence". That is correct behaviour.

To test promotion before real data exists, run `python codes/forecasting/monitor.py --force --retrain` once for a single district by temporarily editing `params["districts"]`. This trains for about 10-15 min and then prints the same-test-period comparison.

### 8.3 Weekly workflow

`.github/workflows/weekly-monitor.yml`:
```yaml
name: weekly-monitor
on:
  schedule:
    - cron: "0 6 * * 1"          # after weekly-data (03:00)
  workflow_dispatch:
jobs:
  monitor:
    runs-on: ubuntu-latest
    timeout-minutes: 360
    env:
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      MLFLOW_TRACKING_USERNAME: ${{ secrets.DAGSHUB_USER }}
      MLFLOW_TRACKING_PASSWORD: ${{ secrets.DAGSHUB_TOKEN }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install dvc mlflow pandas numpy scikit-learn joblib pyyaml seaborn torch --extra-index-url https://download.pytorch.org/whl/cpu
      - run: |
          dvc remote modify origin --local auth basic
          dvc remote modify origin --local user ${{ secrets.DAGSHUB_USER }}
          dvc remote modify origin --local password ${{ secrets.DAGSHUB_TOKEN }}
      - run: dvc pull data/weather.dvc
      - run: python codes/forecasting/monitor.py --retrain
```

### Open problem: where does `forecasts_log.csv` live?

The API writes the log on whatever machine runs it. A GitHub-hosted runner cannot see that file. Pick one (this is a team decision; I did not pick for you):

1. **Version the log with DVC:** wherever the API runs, periodically `dvc add logs/forecasts_log.csv && dvc push`, commit the `.dvc` file, and add `dvc pull logs/forecasts_log.csv.dvc` to the workflow. (Remove `logs/` from `.gitignore` first and add `/logs/forecasts_log.csv` to `.gitignore` only after DVC adds its own entry.)
2. **Have the weekly job generate and log the forecasts itself** using `common.forecast_next`, so the log lives in DVC and no always-on API is needed. This needs one extra script.
3. **Run `monitor.py` yourself** on the machine that runs the API. No workflow needed.

```bash
git add codes/forecasting/monitor.py .github/workflows/weekly-monitor.yml
git commit -m "task 6: weekly monitor, rolling-split retrain, gated promotion"
git push
```

---

## 9. Final checklist

| # | Check | Command / place |
|---|---|---|
| 0 | Hurdle vs LSTM compared on equal terms | `outputs/hurdle/hurdle_rainfall_metrics.csv` |
| 1 | Scripts reproduce notebook metrics | `diff -r /tmp/lstm_baseline outputs/lstm -x "*.png"` |
| 2 | Data and models restore from the remote | `rm -rf models && dvc pull models.dvc` |
| 3 | 5 baseline runs, 5 registered models, 5 champions | DagsHub → Experiments / Models |
| 4 | API returns 3 days and logs them | `curl localhost:8000/forecast/Latur` |
| 5 | CI green; weekly workflow runs after merge | GitHub → Actions |
| 6 | Monitor prints rolling R²; a forced retrain promotes only if better | `monitor.py --force --retrain` |

---

## 10. Merge to `main`

```bash
git checkout vaish_mlops
git fetch origin
git merge origin/main              # resolve conflicts if any, then:
git push
```
GitHub → **Compare & pull request** (`vaish_mlops` → `main`) → ask a teammate to review → merge. The two scheduled workflows start working only after this merge.

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Permission denied (publickey)` | SSH key not added | Redo A2 |
| `dvc push` → 401/403 | Wrong user/token in `.dvc/config.local` | Re-run the three `dvc remote modify --local` commands |
| `output ... is already tracked by SCM` on `dvc add` | File still in git | `git rm -r --cached <path>` then `dvc add <path>` |
| `RuntimeError: Set MLFLOW_TRACKING_URI` | `.env` not loaded in this terminal | `set -a; source .env; set +a` |
| `register_model` fails on the bundle folder | Unverified point #2 | Send me the error; I will switch to `mlflow.pytorch.log_model` |
| `get_champion` returns `None` after promotion | Server without alias support and tag call failed | Check the `champion` tag on the version in the Models UI |
| `torch.load ... weights_only` error | Old torch version | Upgrade torch, or change `weights_only=True` to `False` in `common.load_checkpoint` |
| API: `no champion model registered` | Task 3 `register_existing.py` not run | Run it |
| API forecast looks like the past | NASA lag | Read `data_age_days` in the response; see A7 |
| Docker build is very slow | CPU torch download | Normal for the first build |
| Weekly job cannot push | Branch protection on `main` | Change the last step to open a pull request |
