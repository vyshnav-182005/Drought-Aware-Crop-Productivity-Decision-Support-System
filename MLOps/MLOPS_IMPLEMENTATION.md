# MLOps Implementation

## Scope

The production pipeline uses the existing hurdle rainfall ensemble. The LSTM
notebook remains an archived experiment and is not retrained, registered,
served, or monitored.

The implementation preserves the original model and reported results:

- Data: `data/weather data new/`
- Checkpoints: `outputs/hurdle/checkpoints/hurdle_<district>.joblib`
- Metrics: `outputs/hurdle/results/hurdle_per_district_metrics.csv`
- Registry names: `<district>_hurdle`

The original checkpoints use the `{clf, reg}` format and perform same-day
validation. They are not proven three-day-ahead forecasts.

## Components

| Component | File | Responsibility |
|---|---|---|
| Checkpoint/data utilities | `codes/forecasting/hurdle_pipeline.py` | Load weather data, features, checkpoints, and predictions |
| Baseline loading | `codes/forecasting/train.py` | Verify/load existing checkpoints without retraining |
| Evaluation | `codes/forecasting/evaluate.py` | Read the authoritative district metrics |
| Registry | `codes/forecasting/registry.py` | Log bundles, create versions, and manage the `champion` |
| Import command | `codes/forecasting/register_existing.py` | Register the five existing models |
| API | `serve/app.py` | Serve forecasts and write the forecast log |
| Monitoring | `codes/forecasting/monitor.py` | Score logs and detect degradation |
| Data refresh | `codes/pipeline/fetch_nasa.py` | Refresh NASA POWER data |
| Data cleaning | `codes/pipeline/clean.py` | Clean the authoritative weather directory |

## Data and model flow

```text
NASA POWER → data/weather data new/ → DVC/DagsHub
                                      ↓
                           existing hurdle checkpoints
                                      ↓
                              MLflow champion
                                      ↓
                           FastAPI forecast endpoint
                                      ↓
                           forecasts_log.csv → monitor
```

## Commands

Install and test:

```powershell
python -m pip install -r requirements.txt
pytest -q
```

Register the existing checkpoints:

```powershell
python codes\forecasting\register_existing.py
```

Run the API:

```powershell
uvicorn serve.app:app --host 127.0.0.1 --port 8000
```

Run the baseline checks:

```powershell
python codes\forecasting\train.py --district Latur --no-mlflow
python codes\forecasting\evaluate.py --district Latur
```

## Automation

- `ci.yml`: tests pushes and pull requests.
- `weekly-data.yml`: refreshes NASA data, updates DVC, and commits the data
  pointer every Monday at 03:00 UTC.
- `weekly-monitor.yml`: scores persisted forecast logs every Monday at
  06:00 UTC.

The workflows require `DAGSHUB_USER`, `DAGSHUB_TOKEN`, and
`MLFLOW_TRACKING_URI` as GitHub Actions secrets. Forecast logs must be added
to DVC before the monitor can use them.

## Configuration

`params.yaml` contains districts, data paths, model thresholds, NASA refresh
settings, and monitoring thresholds. Do not change the production checkpoint
contract or claim a future-horizon model without retraining and re-evaluation.

For setup, API operation, workflows, DVC commands, and troubleshooting, see
[OPERATIONS_RUNBOOK.md](./OPERATIONS_RUNBOOK.md).
