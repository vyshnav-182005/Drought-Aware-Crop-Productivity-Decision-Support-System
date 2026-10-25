# Hurdle Model MLOps Operations Runbook

This document explains how to operate the production hurdle rainfall model
pipeline in this repository.

## Production scope

The production MLOps path uses the existing hurdle ensemble checkpoints:

- Data source: `data/weather data new/`
- Checkpoints: `outputs/hurdle/checkpoints/`
- Authoritative metrics: `outputs/hurdle/results/hurdle_per_district_metrics.csv`
- MLflow model names:
  - `Ahmednagar_hurdle`
  - `Beed_hurdle`
  - `Latur_hurdle`
  - `Osmanabad_hurdle`
  - `Solapur_hurdle`

The LSTM notebook is archived and is not part of the production MLOps path.

> **Important model limitation:** the registered checkpoints preserve the
> original hurdle implementation and metrics. The original implementation
> evaluates a same-day rainfall prediction using a one-year holdout. The API
> therefore reports `original_checkpoint_same_day`; it must not be described
> as a rigorously validated three-day-ahead forecast without retraining and
> re-evaluating the hurdle model with future-shifted targets.

## Required GitHub secrets

Configure these under:

```text
Repository → Settings → Secrets and variables → Actions → Secrets
```

Create three separate repository secrets:

```text
DAGSHUB_USER
DAGSHUB_TOKEN
MLFLOW_TRACKING_URI
```

Values:

```text
DAGSHUB_USER=your DagsHub username
DAGSHUB_TOKEN=your DagsHub access token
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/DACPDSS.mlflow
```

Never put tokens in Actions variables, source files, `.env` files committed
to Git, workflow YAML, screenshots, or chat messages.

## Local environment setup

From the repository root, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Configure MLflow for the current PowerShell session:

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/<username>/DACPDSS.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<username>"
$env:MLFLOW_TRACKING_PASSWORD="<DAGSHUB_TOKEN>"
```

Configure the DagsHub DVC remote once:

```powershell
dvc remote add -d origin "https://dagshub.com/<username>/DACPDSS.dvc"
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "<username>"
dvc remote modify origin --local password "<DAGSHUB_TOKEN>"
```

Check the configuration:

```powershell
dvc remote list
dvc status
```

## Registering the existing models

The original checkpoints are imported without retraining:

```powershell
python codes\forecasting\register_existing.py
```

This uploads the existing checkpoint and the corresponding row from
`hurdle_per_district_metrics.csv`, then marks the new version as `champion`.

To register one district only:

```powershell
python codes\forecasting\register_existing.py --district Latur
```

Verify all champions:

```powershell
python -c "import sys; sys.path.insert(0, 'codes\forecasting'); import registry; districts=['Ahmednagar','Beed','Latur','Osmanabad','Solapur']; [print(d, registry.get_champion(d+'_hurdle').version) for d in districts]"
```

## GitHub Actions workflows

### `ci`

Runs automatically on every push and pull request.

It:

1. Checks out the repository.
2. Installs Python dependencies needed by the tests.
3. Runs `pytest -q`.

Run it manually from **Actions → ci → Run workflow** when needed.

Expected result:

```text
2 passed
```

### `weekly-data`

Schedule: Mondays at 03:00 UTC, which is 08:30 IST.

Run manually from:

```text
Actions → weekly-data → Run workflow → select main → Run workflow
```

It:

1. Pulls the current DVC version of `data/weather data new/`.
2. Re-pulls the latest NASA POWER window.
3. Cleans the refreshed data.
4. Creates a new DVC version.
5. Pushes the data to DagsHub.
6. Commits the updated `data/weather data new.dvc` pointer to GitHub.

Run this once for testing, or let the Monday schedule operate it.

### `weekly-monitor`

Schedule: Mondays at 06:00 UTC, which is 11:30 IST.

Run manually from:

```text
Actions → weekly-monitor → Run workflow → select main → Run workflow
```

It:

1. Pulls the current weather data from DVC.
2. Pulls `logs/forecasts_log.csv` from DVC if its pointer exists.
3. Scores logged champion forecasts against actual rainfall.
4. Calculates rolling RMSE and R².
5. Detects degradation.
6. Trains a rolling challenger when the configured trigger is reached.
7. Promotes the challenger only if it beats the current champion on the same
   evaluation period.

If no forecast log has been persisted yet, it safely reports:

```text
No forecast log available; skipping monitoring.
```

## Running the API locally

Configure the MLflow environment variables first, then start the API:

```powershell
uvicorn serve.app:app --host 127.0.0.1 --port 8000
```

Keep that terminal running. From a second PowerShell terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected response:

```text
status
------
ok
```

Request forecasts:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/forecast/Ahmednagar
Invoke-RestMethod http://127.0.0.1:8000/forecast/Beed
Invoke-RestMethod http://127.0.0.1:8000/forecast/Latur
Invoke-RestMethod http://127.0.0.1:8000/forecast/Osmanabad
Invoke-RestMethod http://127.0.0.1:8000/forecast/Solapur
```

Each request:

1. Resolves the district's MLflow `champion`.
2. Downloads the registered hurdle checkpoint.
3. Loads the matching weather data.
4. Produces a rainfall prediction.
5. Appends a row to `logs/forecasts_log.csv`.

## Persisting forecast logs for monitoring

The API log is local runtime state. GitHub Actions cannot see a log stored on
your computer or on another API server until it is versioned with DVC.

After generating forecasts:

```powershell
dvc add logs/forecasts_log.csv
dvc push
git add logs/forecasts_log.csv.dvc
git commit -m "Persist forecast monitoring log"
git push
```

The weekly monitor can then pull and score the log.

## Promotion logic

The monitor uses this sequence:

```text
Current champion
    ↓
Rolling forecast R² is calculated
    ↓
R² falls below champion baseline - configured tolerance
    ↓
Train a rolling challenger
    ↓
Evaluate champion and challenger on the same period
    ↓
Promote only if challenger R² is better
```

The relevant settings are in `params.yaml`:

```yaml
monitor:
  window_days: 60
  min_days: 30
  r2_drop_tolerance: 0.10
  min_improvement: 0.0
```

## DVC operation

Check status:

```powershell
dvc status
```

Pull the authoritative weather data:

```powershell
dvc pull "data/weather data new.dvc"
```

Push local DVC objects:

```powershell
dvc push
```

Do not commit `.dvc/config.local`; it contains local credentials and is
ignored by Git.

## Troubleshooting

### `ModuleNotFoundError: No module named 'mlflow'`

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

### MLflow says `MLFLOW_TRACKING_URI is required`

Set the MLflow variables again in the current PowerShell window. Environment
variables do not automatically carry over to a newly opened terminal.

### A workflow is not visible

Scheduled workflows and their manual dispatch controls must be present on the
default branch. Merge the MLOps pull request into `main`, then reload the
Actions page.

### Weekly monitor skips

Check whether the forecast log exists:

```powershell
Test-Path logs\forecasts_log.csv
Test-Path logs\forecasts_log.csv.dvc
```

If the log exists locally but no `.dvc` pointer exists, persist it using the
commands above.

### DVC authentication fails

Reconfigure the local DVC credentials with a valid DagsHub token:

```powershell
dvc remote modify origin --local user "<username>"
dvc remote modify origin --local password "<DAGSHUB_TOKEN>"
```

Never print or commit the token.
