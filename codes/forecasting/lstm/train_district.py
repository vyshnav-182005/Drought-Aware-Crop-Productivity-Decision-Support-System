"""Train the single-district PyTorch direct-multistep LSTM (same architecture/config
that was finalized for Latur) for one district's CSV, saving model + all evaluation
artifacts into a given output directory.

Usage: python3 train_district.py <district_name> <csv_path> <output_dir>
"""
import sys
import json
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_END = "2018-12-31"
VAL_END = "2021-12-31"
HORIZON = 3
LOOKBACK = 45
LSTM_UNITS = 128
NUM_LSTM_LAYERS = 1
DENSE_UNITS = 384
DROPOUT = 0.25
LEARNING_RATE = 5e-4
BATCH_SIZE = 16
MAX_EPOCHS = 80
PATIENCE = 12
LOG_COLS = ["Rainfall_mm", "Wind_Speed_mps"]


def calendar_feats(index):
    doy = np.asarray(index.dayofyear)
    leap = np.asarray(index.is_leap_year)
    frac = doy / np.where(leap, 366, 365)
    return np.stack([np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)], axis=1)


def make_windows(in_arr, out_arr, cal_arr, lookback, horizon):
    Xs, Ys, Cs = [], [], []
    n = len(in_arr)
    for i in range(n - lookback - horizon + 1):
        Xs.append(in_arr[i:i + lookback])
        Ys.append(out_arr[i + lookback:i + lookback + horizon])
        Cs.append(cal_arr[i + lookback:i + lookback + horizon])
    return (np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32), np.array(Cs, dtype=np.float32))


def make_windows_with_context(in_ctx, out_ctx, cal_ctx, in_arr, out_arr, cal_arr, lookback, horizon):
    full_in = np.concatenate([in_ctx[-lookback:], in_arr], axis=0)
    full_out = np.concatenate([out_ctx[-lookback:], out_arr], axis=0)
    full_cal = np.concatenate([cal_ctx[-lookback:], cal_arr], axis=0)
    return make_windows(full_in, full_out, full_cal, lookback, horizon)


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


def inverse_target(scaled_seq, scaler, log_cols, feature_names):
    n, h, f = scaled_seq.shape
    flat = scaled_seq.reshape(-1, f)
    inv = scaler.inverse_transform(flat).reshape(n, h, f)
    for c in log_cols:
        j = feature_names.index(c)
        inv[:, :, j] = np.expm1(inv[:, :, j])
    return inv


def main(district, csv_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(csv_path, parse_dates=["DATE"]).sort_values("DATE").reset_index(drop=True)
    TARGET_FEATURES = [c for c in raw.columns if c != "DATE"]
    N_TARGET = len(TARGET_FEATURES)
    print(f"[{district}] rows={len(raw)}  range={raw.DATE.min().date()}..{raw.DATE.max().date()}")

    df = raw.copy()
    all_nan_mask = df[TARGET_FEATURES].isna().all(axis=1)
    if all_nan_mask.any():
        last_valid_idx = df.index[~all_nan_mask].max()
        df = df.loc[:last_valid_idx].reset_index(drop=True)
    df = df.set_index("DATE")

    clim_col = "Cloud_Amount_percent"
    train_slice = df.loc[:TRAIN_END, clim_col]
    clim = train_slice.groupby(train_slice.index.dayofyear).mean()
    clim_ext = pd.concat([clim, clim, clim])
    clim_smooth = clim_ext.rolling(7, center=True, min_periods=1).mean().iloc[len(clim):2 * len(clim)]
    clim_smooth.index = clim.index
    doy_all = df.index.dayofyear
    fill_values = doy_all.map(clim_smooth)
    mask = df[clim_col].isna()
    df.loc[mask, clim_col] = fill_values[mask.values]
    other_cols = [c for c in TARGET_FEATURES if c != clim_col]
    df[other_cols] = df[other_cols].interpolate(method="time", limit_direction="both")
    assert df.isna().sum().sum() == 0

    df_model = df.copy()
    for c in LOG_COLS:
        df_model[c] = np.log1p(df_model[c])

    df_model["Rain_roll7"] = df["Rainfall_mm"].rolling(7, min_periods=1).sum()
    df_model["Rain_roll14"] = df["Rainfall_mm"].rolling(14, min_periods=1).sum()
    df_model["Rain_roll30"] = df["Rainfall_mm"].rolling(30, min_periods=1).sum()
    df_model["Humid_roll7"] = df["Relative_Humidity_percent"].rolling(7, min_periods=1).mean()
    df_model["Solar_roll7"] = df["Solar_Radiation"].rolling(7, min_periods=1).mean()
    df_model["Wind_roll7"] = df["Wind_Speed_mps"].rolling(7, min_periods=1).mean()
    df_model["Wind50_roll7"] = df["Wind_Speed_50m_mps"].rolling(7, min_periods=1).mean()
    df_model["Cloud_roll7"] = df["Cloud_Amount_percent"].rolling(7, min_periods=1).mean()
    ENGINEERED = ["Rain_roll7", "Rain_roll14", "Rain_roll30", "Humid_roll7",
                  "Solar_roll7", "Wind_roll7", "Wind50_roll7", "Cloud_roll7"]
    INPUT_FEATURES = TARGET_FEATURES + ENGINEERED
    N_INPUT = len(INPUT_FEATURES)

    train_df = df_model.loc[:TRAIN_END]
    val_df = df_model.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1):VAL_END]
    test_df = df_model.loc[pd.Timestamp(VAL_END) + pd.Timedelta(days=1):]

    in_scaler = MinMaxScaler()
    out_scaler = MinMaxScaler()
    train_in = in_scaler.fit_transform(train_df[INPUT_FEATURES])
    val_in = in_scaler.transform(val_df[INPUT_FEATURES])
    test_in = in_scaler.transform(test_df[INPUT_FEATURES])
    train_out = out_scaler.fit_transform(train_df[TARGET_FEATURES])
    val_out = out_scaler.transform(val_df[TARGET_FEATURES])
    test_out = out_scaler.transform(test_df[TARGET_FEATURES])
    joblib.dump(in_scaler, out_dir / "in_scaler.joblib")
    joblib.dump(out_scaler, out_dir / "out_scaler.joblib")

    cal_train = calendar_feats(train_df.index)
    cal_val = calendar_feats(val_df.index)
    cal_test = calendar_feats(test_df.index)

    Xtr, Ytr, Ctr = make_windows(train_in, train_out, cal_train, LOOKBACK, HORIZON)
    Xval, Yval, Cval = make_windows_with_context(train_in, train_out, cal_train, val_in, val_out, cal_val, LOOKBACK, HORIZON)

    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ctr), torch.from_numpy(Ytr))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    Xval_t = torch.from_numpy(Xval).to(DEVICE)
    Cval_t = torch.from_numpy(Cval).to(DEVICE)
    Yval_t = torch.from_numpy(Yval).to(DEVICE)

    model = DirectLSTM(N_INPUT, N_TARGET, HORIZON, LSTM_UNITS, NUM_LSTM_LAYERS, DENSE_UNITS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    loss_fn = nn.HuberLoss(delta=1.0)

    history = {"loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss, n_batches = 0.0, 0
        for xb, cb, yb in train_loader:
            xb, cb, yb = xb.to(DEVICE), cb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb, cb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_t, Cval_t), Yval_t).item()
        scheduler.step(val_loss)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-5:
            best_val, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"[{district}] early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    print(f"[{district}] best_val={best_val:.5f} epochs={len(history['loss'])}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["loss"], label="train loss")
    ax.plot(history["val_loss"], label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Huber loss (scaled)")
    ax.set_title(f"{district} — training curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=110)
    plt.close(fig)

    trainval_in = np.concatenate([train_in, val_in], axis=0)
    trainval_out = np.concatenate([train_out, val_out], axis=0)
    trainval_cal = np.concatenate([cal_train, cal_val], axis=0)
    Xtest, Ytest, Ctest = make_windows_with_context(trainval_in, trainval_out, trainval_cal,
                                                     test_in, test_out, cal_test, LOOKBACK, HORIZON)
    Xtest_t = torch.from_numpy(Xtest).to(DEVICE)
    Ctest_t = torch.from_numpy(Ctest).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(Xtest_t, Ctest_t).cpu().numpy()

    y_true = inverse_target(Ytest, out_scaler, LOG_COLS, TARGET_FEATURES)
    y_pred = inverse_target(pred_scaled, out_scaler, LOG_COLS, TARGET_FEATURES)

    rows = []
    for j, feat in enumerate(TARGET_FEATURES):
        a, p = y_true[:, :, j].reshape(-1), y_pred[:, :, j].reshape(-1)
        rows.append({"feature": feat, "RMSE": np.sqrt(mean_squared_error(a, p)),
                     "MAE": mean_absolute_error(a, p), "R2": r2_score(a, p)})
    metrics_df = pd.DataFrame(rows).set_index("feature").round(3)
    metrics_df.to_csv(out_dir / "test_metrics_by_feature.csv")
    blended_r2 = float(metrics_df["R2"].mean())
    print(f"[{district}] BLENDED R2 = {blended_r2:.4f}")
    for feat, r2 in metrics_df["R2"].items():
        print(f"    {feat}: {r2:.3f}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#4c72b0" if v >= 0.8 else ("#dd8452" if v >= 0.5 else "#c44e52") for v in metrics_df["R2"]]
    ax.bar(metrics_df.index, metrics_df["R2"], color=colors)
    ax.axhline(0.8, color="black", linestyle="--", lw=1, label="R² = 0.8 reference")
    ax.axhline(blended_r2, color="#4c72b0", linestyle=":", lw=1.5, label=f"blended R² = {blended_r2:.3f}")
    ax.set_ylabel("R² (test set, pooled over all forecast days)")
    ax.set_title(f"{district} — per-feature R² (3-day-ahead forecast)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "r2_by_feature.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for ax, feat in zip(axes.flat, TARGET_FEATURES):
        j = TARGET_FEATURES.index(feat)
        a, p = y_true[:, :, j].reshape(-1), y_pred[:, :, j].reshape(-1)
        ax.scatter(a, p, s=4, alpha=0.15, color="#4c72b0")
        lims = [min(a.min(), p.min()), max(a.max(), p.max())]
        ax.plot(lims, lims, color="black", lw=1, linestyle="--")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"{feat}\nR² = {metrics_df.loc[feat, 'R2']:.3f}", fontsize=10)
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
    axes.flat[-1].axis("off")
    fig.suptitle(f"{district} — parity plots", y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "parity_plots.png", dpi=110)
    plt.close(fig)

    sample_positions = [len(Xtest) // 5, len(Xtest) // 2, 4 * len(Xtest) // 5]
    plot_features = ["Rainfall_mm", "Temperature_Max_C", "Relative_Humidity_percent"]
    test_full_dates = pd.concat([train_df, val_df]).index[-LOOKBACK:].append(test_df.index)

    fig, axes = plt.subplots(len(plot_features), len(sample_positions), figsize=(14, 9), sharey="row")
    for col, pos in enumerate(sample_positions):
        start_date = test_full_dates[pos + LOOKBACK]
        dates = pd.date_range(start_date, periods=HORIZON)
        for row, feat in enumerate(plot_features):
            j = TARGET_FEATURES.index(feat)
            ax = axes[row, col]
            ax.plot(dates, y_true[pos, :, j], marker="o", color="black", label="Actual")
            ax.plot(dates, y_pred[pos, :, j], marker="o", color="#c44e52", label="LSTM forecast")
            ax.set_xticks(dates)
            ax.set_xticklabels([d.strftime("%b %d") for d in dates])
            if row == 0:
                ax.set_title(f"Forecast from {start_date.date()}")
            if col == 0:
                ax.set_ylabel(feat)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"{district} — sample forecasts", y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "sample_forecasts.png", dpi=110)
    plt.close(fig)

    day1_true = y_true[:, 0, :]
    day1_pred = y_pred[:, 0, :]
    day1_dates = test_full_dates[LOOKBACK: LOOKBACK + len(day1_true)]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, feat in zip(axes, ["Rainfall_mm", "Temperature_Max_C"]):
        j = TARGET_FEATURES.index(feat)
        ax.plot(day1_dates, day1_true[:, j], color="black", lw=1, label="Actual")
        ax.plot(day1_dates, day1_pred[:, j], color="#c44e52", lw=1, alpha=0.85, label="LSTM (1-day-ahead component)")
        ax.set_title(f"{district} — {feat} rolling 1-day-ahead forecast")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "rolling_day1.png", dpi=110)
    plt.close(fig)

    model_path = out_dir / f"{district.lower()}_weather_lstm_pytorch.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {"n_input": N_INPUT, "n_target": N_TARGET, "horizon": HORIZON,
                          "lstm_units": LSTM_UNITS, "num_layers": NUM_LSTM_LAYERS,
                          "dense_units": DENSE_UNITS, "dropout": DROPOUT},
    }, model_path)

    config = {
        "district": district, "target_features": TARGET_FEATURES, "input_features": INPUT_FEATURES,
        "log_transformed_columns": LOG_COLS, "lookback": LOOKBACK, "horizon": HORIZON,
        "lstm_units": LSTM_UNITS, "num_layers": NUM_LSTM_LAYERS, "dense_units": DENSE_UNITS,
        "dropout": DROPOUT, "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "train_end": TRAIN_END, "val_end": VAL_END, "blended_test_r2": blended_r2,
        "trained_through_date": str(df.index.max().date()),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[{district}] saved artifacts to {out_dir}")
    return blended_r2, metrics_df


if __name__ == "__main__":
    district, csv_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    main(district, csv_path, out_dir)
