# Drought-Aware Crop Productivity Decision-Support System (AB11)

B.Tech 7th-semester final-year project, Amrita Vishwa Vidyapeetham, Coimbatore.
Team AB11: Gowtham S D, Guhan K B, Vaishnavi Gupta, Vyshnav Kumar S. Mentor: Dr. Unnikrishnan.

## The problem

Five districts in Maharashtra's traditional scarcity zone (Ahmednagar, Beed, Latur, Osmanabad and Solapur) face recurring drought, delayed monsoons and long dry spells. The government already publishes an official **District Agriculture Contingency Plan** for each of them, telling farmers what to do for each crop when the monsoon is delayed or a dry spell hits. In practice those plans sit unused as 50-page PDFs that have to be read by hand. Nothing links the weather, the expected crop yield, the human impact and that official guidance together, and drought is usually judged against generic thresholds instead of each district's own history.

## What the system does

For a chosen district and season (Kharif or Rabi), the system:

1. **Forecasts the weather** with a self-trained model per district, without depending on a live weather API.
2. **Classifies the season's condition** (normal, dry spell, deficit, drought-like) against that district's own 26-year rainfall history.
3. **Predicts crop yield** with explainable ML and shows which weather factors drove the prediction (SHAP).
4. **Translates the yield loss into human impact** using Census 2011 population and workforce data, and ranks the five districts by priority.
5. **Retrieves the official contingency-plan actions** for that district, crop and condition from a knowledge graph, and has an LLM write them up as one advisory answer.

**Guardrail (non-negotiable):** the LLM only rephrases facts retrieved from the knowledge graph and numbers already produced by the models. It never predicts a yield or invents a recommendation on its own, and every recommendation cites the contingency-plan page it came from.

Scope: 5 districts, Kharif and Rabi seasons only (Summer/Zaid area is negligible in all five plans). Core crop-seasons: Kharif Jowar, Soybean, Tur, Sunflower and Bajra; Rabi Jowar, Sunflower and Gram. Cotton, Wheat and Safflower form an optional extended set. Sugarcane (perennial) and horticulture crops are not modelling targets, but appear in the knowledge graph wherever a contingency plan names them.

## Data sources

| Source | Folder | Coverage | Used for |
|---|---|---|---|
| NASA POWER daily weather (11 variables: rainfall, mean/max/min temperature, humidity, wind, wind at 50 m, surface pressure, solar radiation, dew point, cloud cover) | `data/weather/` | 2000-01-01 to 2025-12-31, one file per district | Weather forecasting, season aggregation, drought classification |
| DES district crop statistics (area, production, yield) | `data/crop_production/` | 2000-01 to 2022-23 | Yield model target; which crops each district grows |
| ICAR District Agriculture Contingency Plans | `data/contingency_plans/` | 5 PDFs, 2011 | District soil/rainfall/irrigation profile and the condition → action tables (knowledge graph) |
| Census 2011 (population, density, sex ratio, workers 1991/2001/2011) | `data/census/` | District level | Human-impact layer only, never a yield-model feature |

Data notes: the DES and Census `.xls` files are really HTML and are read with `pandas.read_html`. District names differ between sources (e.g. Ahilyanagar/Ahmadnagar/Ahmednagar, Bid/Beed, Dharashiv/Osmanabad) and are mapped to one canonical name in `codes/knowledge_graph/district_name_map.py`. Rabi Gram has no DES data in the current download and must be sourced separately.

## Pipeline: 10 stages in 4 phases

| Phase | Stage | What it does | Status |
|---|---|---|---|
| I. Weather forecasting | 1 | Select district and crop-season | Done |
| | 2 | Per-district model forecasts the next 3 days of all 11 weather variables | **Done** (LSTM, plus SARIMA, XGBoost, TFT and Hurdle benchmarks) |
| II. Condition and yield | 3 | Aggregate daily weather into season rainfall, longest dry spell and monsoon onset | Planned |
| | 4 | Rule-based (SPI / tercile) classification of the season against the district's own history | Planned |
| | 5 | Yield prediction per district-crop-season (Random Forest / Gradient Boosting, linear baseline) | Planned |
| | 6 | SHAP explanation of each yield prediction | Planned |
| III. Impact and comparison | 7 | Convert predicted yield loss into people affected using Census data | Planned |
| | 8 | Normalise severity and exposure scores and rank the five districts | Planned |
| IV. Knowledge graph and LLM | 9 | Query the knowledge graph for the matching official actions | **Done** (graph, season history, query API, Neo4j export) |
| | 10 | GraphRAG: LLM extracts entities, retrieves the subgraph, writes a grounded answer | Planned |

Deep learning is used for weather (about 9,500 daily rows per district) but not for yield, which has only about 23 seasons per district-crop.

## Stage 2: weather forecasting (built)

One PyTorch LSTM per district (`models/<district>_weather_lstm_pytorch.pt`), trained in `codes/forecasting/lstm.ipynb`.

- **Input:** a 45-day lookback of the 11 daily variables plus 8 rolling features (7/14/30-day rainfall sums; 7-day means of humidity, solar radiation, wind, wind at 50 m and cloud cover), together with day-of-year sin/cos terms. Rainfall and wind speed are log-transformed.
- **Model:** a 1-layer LSTM (128 units) followed by dense layers (384 units, dropout 0.25). It predicts all 11 variables for the next 3 days in one direct multi-step output.
- **Training:** train up to 2018, validate on 2019 to 2021, test on 2022 onwards. Huber loss, Adam optimiser (lr 5e-4), early stopping.
- **Results (blended test R² across all 11 variables):** Ahmednagar 0.749, Beed 0.750, Latur 0.738, Osmanabad 0.725, Solapur 0.736. Temperature, humidity and dew point reach R² 0.85 to 0.94. **Rainfall is the weakest variable (R² 0.10 to 0.22)** because more than 80% of days are dry.
- The team also benchmarked SARIMA, XGBoost, a Temporal Fusion Transformer and a **two-stage Hurdle model**. The Hurdle model first predicts rain or no rain (probability threshold 0.35), then predicts the amount on log(1 + rainfall). On rainfall it does much better (R² 0.79 to 0.94), so it is the stronger rainfall model to feed into season aggregation.

Per-district configs, scalers, test metrics and plots are in `outputs/plots/lstm/<District>/`.

## Stage 9: knowledge graph (built)

Built from the five contingency-plan PDFs, the DES crop data and the 2000–2025 weather history. It is stored as a NetworkX graph, exported as GraphML and JSON, ready to load into Neo4j, and has an interactive HTML viewer (`outputs/knowledge_graph/`).

- **833 nodes:**
  - District (5), SoilType (12), Season (2), Crop (35)
  - Condition (7): early-season, mid-season and terminal drought; irrigated water shortage; unusual rains; floods; extreme events
  - ConditionDetail (24 sub-types with machine-readable values, e.g. onset delayed 4 weeks, 2-week dry spell at flowering)
  - Action (462 unique recommendations)
  - Year (26) and SeasonRecord (260, one per district-season-year)
- **4,353 edges.** The core is ConditionDetail `RECOMMENDS` Action: 1,063 edges, one per contingency-table row, each carrying the district, crop, Kharif/Rabi season, soil, crop stage and **source PDF page**. Around it:
  - `SUGGESTS_CROP`: alternative and Rabi follow-up crops
  - `NOT_APPLICABLE`: conditions the plan rules out for a district
  - `HAS_SOIL`, `GROWS`, `IN_SEASON`
  - `EXPERIENCED`: which condition each past season actually had
  - `HAD_YIELD`: the DES yield that followed
- **Season labeller (stand-in for Stage 4).** `label_seasons.py` labels every past season using the plans' own definitions (e.g. 2 rainless weeks = mid-season dry spell). Stage 4's classifier can replace its output file without any change to the graph.
- **LLM entry point.** `kg_query.retrieve_context(G, district, crop, season, condition)` is the one call the Stage 10 layer needs. It returns the official actions with page citations, alternative crops, past seasons with the same condition and their yields, and the district profile.

Schema, extraction rules, checks and known limitations are in [`codes/knowledge_graph/README.md`](codes/knowledge_graph/README.md).

## Repository structure

```
├── codes/
│   ├── preprocessing.ipynb        # downloads (NASA POWER, regional points, climate indices, Open-Meteo) + cleaning
│   ├── forecasting/lstm.ipynb     # LSTM: data prep, training, evaluation, plots for all districts
│   └── knowledge_graph/           # PDF extraction, season labeller, graph build, viewer, Neo4j export, query API
├── data/
│   ├── weather/v1/, v2/           # raw NASA POWER (v2 = 11 variables, used for training)
│   ├── weather/clean/             # gap-filled v2, written by preprocessing.ipynb, read by lstm.ipynb
│   ├── crop_production/           # DES district crop statistics
│   ├── census/                    # Census 2011 tables
│   └── contingency_plans/         # 5 ICAR district contingency-plan PDFs
├── models/                        # LSTM weights: <district>_weather_lstm_pytorch.pt
├── outputs/
│   ├── knowledge_graph/           # extracted CSVs, graph (.graphml/.json), interactive viewer
│   └── plots/lstm/<District>/     # scalers, config.json, test metrics, plots
└── docs/                          # zeroth and first review presentations
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

## How to run

```bash
pip install -r requirements.txt
```

1. **`codes/preprocessing.ipynb`.** By default this only re-runs the cleaning step. Set the `RUN_*_DOWNLOAD` flags to `True` to download fresh data. New downloads go to `data/weather/v3/`, so `v2` is never overwritten.
2. **`codes/forecasting/lstm.ipynb`.** With `RUN_TRAINING = False`, it loads the saved models and re-evaluates them. With `True`, it retrains every district.
3. **Knowledge graph.** Run from the repo root:

   ```bash
   python codes/knowledge_graph/run_all.py      # extract -> label seasons -> build -> viewer -> Neo4j export
   python codes/knowledge_graph/kg_query.py --district Beed --crop Soybean --condition mid_dry_spell_flowering
   python codes/knowledge_graph/kg_query.py --list-conditions
   ```

   Then open `outputs/knowledge_graph/knowledge_graph_viewer.html` in a browser.

## Team

| Member | Built so far | Remaining stages |
|---|---|---|
| Vyshnav Kumar S | Per-district LSTM weather forecaster, knowledge graph | Feed forecasts into season aggregation; knowledge graph |
| Gowtham S D | TFT and two-stage Hurdle ensemble | GraphRAG entity extraction, subgraph retrieval, guardrailed LLM answer |
| Guhan K B | Per-district XGBoost forecaster | Season aggregation and condition classification, yield model, forecast retraining |
| Vaishnavi Gupta | Per-district SARIMA forecaster | SHAP, Census human impact, district ranking, data ingestion / monitoring |

All four share data preprocessing, EDA, model benchmarking, integration, testing and the final report.

## Known gaps

- The forecast horizon is 3 days. A season-level outlook needs a longer horizon or a hand-off to an external forecast for later days.
- Rainfall is the variable the yield stage needs most, and it is the hardest to forecast (see the Hurdle model results above).
- Rabi Gram/Chickpea is missing from the DES crop data.
- A per-capita food-consumption norm (e.g. ICMR-NIN or PDS allocation) is still needed to turn yield loss into food need.
- The contingency plans date from 2011, and soil and irrigation figures are a single static snapshot.

Older scripts from before the repository restructure (`data.py`, `train_district.py`, `r2score.ipynb`, the Latur prototype notebook) are preserved in git commit `b690c61`.
