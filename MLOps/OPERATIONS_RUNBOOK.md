# MLOps Operations Runbook

This project uses the existing hurdle rainfall models for production. The
LSTM notebook is archived and is not part of serving, registration, or
monitoring.

## Production assets

| Asset | Location |
|---|---|
| Weather data | `data/weather data new/` |
| Checkpoints | `outputs/hurdle/checkpoints/` |
| Baseline metrics | `outputs/hurdle/results/hurdle_per_district_metrics.csv` |
| API | `serve/app.py` |
| MLflow models | `<district>_hurdle` |

The registered checkpoints reproduce the original same-day validation model.
The API labels responses `original_checkpoint_same_day`; do not describe them
as validated three-day forecasts.

## One-time setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set MLflow variables in each new PowerShell session:

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/<username>/DACPDSS.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<username>"
$env:MLFLOW_TRACKING_PASSWORD="<DAGSHUB_TOKEN>"
```

Configure DVC once:

```powershell
dvc remote add -d origin "https://dagshub.com/<username>/DACPDSS.dvc"
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "<username>"
dvc remote modify origin --local password "<DAGSHUB_TOKEN>"
dvc status
```

Never commit or share the token. Do not commit `.dvc/config.local`.

## Register the existing models

This imports the existing checkpoints; it does not retrain them:

```powershell
python codes\forecasting\register_existing.py
```

Register one district:

```powershell
python codes\forecasting\register_existing.py --district Latur
```

## Run the API

```powershell
uvicorn serve.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/forecast/Latur
```

Available districts: `Ahmednagar`, `Beed`, `Latur`, `Osmanabad`, and
`Solapur`. Each forecast resolves the MLflow champion, loads weather data,
returns a prediction, and appends to `logs/forecasts_log.csv`.

## GitHub Actions

Configure these repository **secrets** under **Settings → Secrets and
variables → Actions**:

```text
DAGSHUB_USER
DAGSHUB_TOKEN
MLFLOW_TRACKING_URI
```

| Workflow | Schedule | Purpose |
|---|---|---|
| `ci` | Push/PR | Installs dependencies and runs `pytest -q` |
| `weekly-data` | Monday 03:00 UTC (08:30 IST) | Refreshes NASA data and pushes the DVC update |
| `weekly-monitor` | Monday 06:00 UTC (11:30 IST) | Scores persisted forecasts and reports degradation |

To run one manually: **Actions → workflow → Run workflow → select `main`**.

`weekly-data` refreshes `data/weather data new/`, cleans it, pushes DVC
objects, and commits the updated `.dvc` pointer. `weekly-monitor` needs a
forecast log in DVC; otherwise it safely reports that no log is available.

## Persist forecast logs

GitHub Actions cannot access a log on a local machine. After generating
forecasts:

```powershell
dvc add logs/forecasts_log.csv
dvc push
git add logs/forecasts_log.csv.dvc
git commit -m "Persist forecast monitoring log"
git push
```

## Monitoring and promotion

The monitor calculates rolling RMSE/R² and checks the configured thresholds in
`params.yaml`:

```yaml
monitor:
  window_days: 60
  min_days: 30
  r2_drop_tolerance: 0.10
  min_improvement: 0.0
```

The intended sequence is: detect degradation, train a rolling challenger,
compare both models on the same period, and promote only a better challenger.
Confirm legacy-checkpoint compatibility before enabling automatic promotion.

## Useful commands

```powershell
dvc status
dvc pull "data/weather data new.dvc"
dvc push
pytest -q
```

If workflows are missing from Actions, merge the MLOps branch into `main`.
