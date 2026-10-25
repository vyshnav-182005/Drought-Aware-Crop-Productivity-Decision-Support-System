# Drought-Aware Crop Productivity Decision-Support System (AB11)

Final-year project for five drought-prone Maharashtra districts: Ahmednagar, Beed, Latur, Osmanabad and Solapur.

- **Weather forecasting.** A hurdle ensemble, one model per district, forecasts rainfall up to 3 days ahead. It is trained on NASA POWER data from 2000 onward.
- **Knowledge graph.** Built from the district Agriculture Contingency Plans and the DES crop data. It maps a district, crop and drought condition to the recommended actions.

## Structure

```
├── codes/
│   ├── preprocessing.ipynb        # downloads (NASA POWER, regional points, climate indices, Open-Meteo) + cleaning
│   ├── forecasting/lstm.ipynb     # archived LSTM experiment
│   ├── forecasting/hurdle/        # classifier + intensity regressor
│   └── knowledge_graph/           # PDF extraction, graph build, HTML viewer, query (see its README)
├── data/
│   ├── weather/v1/, v2/           # raw NASA POWER (v2 = 11 variables, used for training)
│   ├── weather/clean/             # gap-filled v2, written by preprocessing.ipynb, read by lstm.ipynb
│   ├── crop_production/, census/, contingency_plans/
├── models/                        # Hurdle checkpoints
├── outputs/
│   ├── knowledge_graph/           # extracted CSVs, graph (.graphml/.json), interactive viewer
│   └── lstm/<District>/           # scalers, config.json, test metrics, plots
└── docs/                          # review presentations
```

## MLOps workflow

The hurdle pipeline is the production path. It supports leakage-safe
three-day rainfall forecasts, MLflow tracking and champion promotion, a
FastAPI service, DVC-managed data, and weekly monitoring. See
[`MLOps/MLOPS_IMPLEMENTATION.md`](./MLOps/MLOPS_IMPLEMENTATION.md) for setup.

The current production baseline was trained from
`data/weather data new/`. Its authoritative metrics are stored in
`outputs/hurdle/results/hurdle_per_district_metrics.csv`, and its checkpoints
are under `outputs/hurdle/checkpoints/`. The registration command imports those
existing checkpoints; it does not retrain them:

```bash
python codes/forecasting/register_existing.py
```

The API writes forecast observations to `logs/forecasts_log.csv`. To make
those observations available to the scheduled monitor, periodically version
the log and push it to DVC:

```bash
dvc add logs/forecasts_log.csv
dvc push
git add logs/forecasts_log.csv.dvc
git commit -m "Persist forecast monitoring log"
git push
```

## How to run

```bash
pip install -r requirements.txt
```

1. **`codes/preprocessing.ipynb`.** By default this only re-runs the cleaning step. Set the `RUN_*_DOWNLOAD` flags to `True` to download fresh data. New downloads go to `data/weather/v3/`, so `v2` is never overwritten.
2. **`codes/forecasting/lstm.ipynb`.** With `RUN_TRAINING = False`, it loads the saved models and re-evaluates them. With `True`, it retrains every district.
3. **Knowledge graph.** Run these from the repo root:

   ```bash
   python codes/knowledge_graph/extract_pdfs.py
   python codes/knowledge_graph/build_graph.py
   python codes/knowledge_graph/build_html_viewer.py   # then open outputs/knowledge_graph/knowledge_graph_viewer.html
   python codes/knowledge_graph/kg_query.py --district Beed --crop Soybean --condition "Mid-season drought (dry spell)"
   ```

The earlier layout and its old scripts (`data.py`, `train_district.py`, `r2score.ipynb`, the Latur prototype notebook, and others) are preserved in git commit `b690c61`.
