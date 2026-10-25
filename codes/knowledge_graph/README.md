# Knowledge graph (Stages 9–10 input)

Turns the five District Agriculture Contingency Plans, the DES crop statistics and the daily weather history into one graph. The Stage 10 GraphRAG / LLM layer queries it through `kg_query.retrieve_context()`. Every fact in the graph keeps its source: a contingency-plan page, DES, or the season labeller.

## Build

Run from the project root:

```bash
python codes/knowledge_graph/run_all.py          # runs all five steps below in order
```

| Step | Script | Output (`outputs/knowledge_graph/`) |
|---|---|---|
| 1 | `extract_pdfs.py` | `district_static_features.csv`, `contingency_actions.csv` (1,063 rows), `not_applicable.csv` (27 rows) |
| 2 | `label_seasons.py` | `season_conditions.csv` (124 labels), `season_weather_summary.csv` (260 district-seasons) |
| 3 | `build_graph.py` | `knowledge_graph.json`, `knowledge_graph.graphml`, `graph_stats.json` |
| 4 | `build_html_viewer.py` | `knowledge_graph_viewer.html` (open in a browser) |
| 5 | `export_neo4j.py` | `neo4j/` with one CSV per node and relationship type, plus `import.cypher` |

Needs `pdfplumber`, `networkx`, `scipy` (viewer layout), `pandas` and `lxml`. Optional: `neo4j` (Python driver) to push straight into a database.

## Schema

**Nodes (833)**

| Type | Count | What it is |
|---|---|---|
| District | 5 | Rainfall normal and irrigated % from the plan |
| SoilType | 12 | Soil categories from plan Section 1.4, plus the soil groups used in the Section 2 tables |
| Season | 2 | Kharif, Rabi |
| Crop | 35 | 5 core, 4 extended, 26 fallback/horticulture (`category` attribute) |
| Condition | 7 | Early-season drought, mid-season drought, terminal drought, irrigated-situation water shortage, unusual rains, floods, extreme events |
| ConditionDetail | 24 | Sub-types with machine-readable values, e.g. `onset_delay_4w` (`onset_delay_weeks=4`), `mid_dry_spell_flowering` (`min_rainless_days=14`, `crop_stage`), `canal_limited_release`, `hailstorm` |
| Action | 462 | Unique recommendation text, split into the plan's own columns: change in cropping, agronomic measures, crop management, soil/moisture conservation, Rabi crop planning, remarks. Each has `action_types`, e.g. `crop_substitution;irrigation` |
| Year | 26 | 2000–2025 |
| SeasonRecord | 260 | One per district, season and year, carrying the season's weather summary (rainfall, % departure, IMD-style status, onset date, longest dry spell) |

**Edges (4,353)**

| Relation | From → To | Count | Notes |
|---|---|---|---|
| `SUBTYPE_OF` | ConditionDetail → Condition | 24 | |
| `RECOMMENDS` | ConditionDetail → Action | 1,063 | One per contingency-table row. Carries district, crop, season (+ `season_basis`), situation, soil type (normalised + raw), moisture regime, crop stage, raw PDF condition text, ditto flags and **source page** |
| `APPLIES_TO_CROP` | Crop → Action | 702 | |
| `SOIL_CONTEXT` | SoilType → Action | 399 | |
| `SUGGESTS_CROP` | Action → Crop | 321 | Crop named in the action: `role` is `alternative` or `rabi_follow_up`, with `season` and an `intercrop` flag |
| `NOT_APPLICABLE` | District → Condition / ConditionDetail | 27 | The plan explicitly says "Not applicable" (e.g. floods in Beed) |
| `HAS_SOIL` | District → SoilType | 15 | Area and % from Section 1.4 |
| `INCLUDES` | SoilType → SoilType | 6 | Section 2 soil group → Section 1.4 category. Derived from names only (`basis=name_composition`) |
| `GROWS` | District → Crop | 84 | `data_source` DES and/or contingency_plan, `seasons` |
| `IN_SEASON` | Crop → Season | 20 | |
| `HAS_SEASON_RECORD` / `IN_YEAR` / `OF_SEASON` | District → SeasonRecord → Year / Season | 260 each | |
| `EXPERIENCED` | SeasonRecord → ConditionDetail | 124 | Condition that season actually had, with `basis` and `evidence` |
| `HAD_YIELD` | SeasonRecord → Crop | 788 | Observed DES area, production and yield |

Every node has a `code` or `name`, and its key is `<Type>:<name or code>`. Filter edges by the `relation` attribute, not the edge key.

## What the extractor handles

The plans use several table layouts, so `extract_pdfs.py` reads each table's own header instead of assuming fixed columns:

- **Column meaning by layout.**
  - Delayed-onset and irrigated tables: *change in crop / agronomic measures / remarks*.
  - Normal-onset and mid-season tables: *crop management / soil nutrient & moisture conservation / remarks*.
  - Terminal-drought tables: *crop management / Rabi crop planning / remarks*.
  - Sections 2.2–2.4 (unusual rains, floods, extreme events): one column per crop stage (vegetative, flowering, maturity, post-harvest, …).
- **Wide tables** (Solapur's 8–11 columns, Osmanabad p.24). Spanned header cells are inherited from the left. Solapur's *Kharif | Rabi* sub-columns become separate rows, each with its own season.
- **Two-row headers**, as in Beed. Each column takes its role from whichever header row names it.
- **Page breaks.** Text that spills onto the next page is appended to the row it belongs to. The condition detail carries across pages, and split lines such as "Delay by 4 weeks" + "2nd week of July" are re-joined. A soil cell that continues on the next page ("… with assured rainfall") is joined too.
- **Irrigated tables** (canal / tank / groundwater) get their own category. They no longer inherit the drought heading printed above them.
- **Crop names** inside free text, e.g. "Kharif Pearlmillet Preseasonal sugarcane Maize", are found by scanning against the crop vocabulary in `district_name_map.py`. A "Rabi"/"Kharif" prefix sets the season.
- **"-do-" and "As above"** are resolved in `build_graph.py` from the row above in the same table and column. In stage tables, a ditto to the right of a filled cell repeats that cell. Every resolution is recorded in `ditto_resolved`. Five dittos have nothing above them and are left flagged as `unresolved`.

**Check.** The first 20 characters of every extracted field were searched for in that page's table cells: 4,027 fields, 99% found verbatim. The rest are deliberate joins (re-joined detail lines and page-overflow text). Beed p.19 and Solapur p.11 were also compared visually against the PDF.

## Season of each recommendation (`season_basis`)

1. **`pdf_text`**: the plan says "Rabi sorghum", or the row sits in a Kharif/Rabi sub-column.
2. **`crop_calendar`**: the crop has only one season (Soybean → Kharif, Gram → Rabi, …).
3. **`kharif_drought_table`**: a two-season crop (Jowar, Sunflower) in a Section 2.1.1 rainfed drought table. Those tables are about the Kharif monsoon.
4. **`unspecified`**: the plan doesn't say. This applies mostly to horticulture rows and the unusual-rain tables. `kg_query.recommend(season=...)` keeps these rows rather than dropping them.

## Season labeller (`label_seasons.py`): the source of `EXPERIENCED`

Each (district, season, year) is labelled from the observed daily rainfall using the plans' own definitions:

| Label | Rule |
|---|---|
| `onset_delay_{2,4,6,8}w` | Onset = first day from 1 June with a 3-day total ≥ 20 mm and no run of ≥ 14 rainless days (< 2.5 mm) in the next 30 days. Delay is counted in weeks after 8 June, because the plans equate a 2-week delay with the 4th week of June. |
| `normal_onset_dry_spell` | Onset not delayed, and a run of ≥ 15 rainless days starting within 30 days of onset. |
| `mid_dry_spell_vegetative` / `_flowering` | A run of ≥ 14 rainless days starting 31–60 / 61–90 days after onset. |
| `terminal_drought` | 15 Sep – 15 Oct rainfall below 50% of the district's own 2000–2025 mean. |
| `continuous_high_rainfall` | A 3-day total ≥ 150 mm (Kharif), or ≥ 64.5 mm in Nov–Feb (Rabi, unseasonal). |

The labels match known bad years. For example, terminal drought appears in 2014 and 2018, onset is delayed 8 weeks in 2015 and 2000, and Solapur's onset is delayed in 2012.

**This labeller stands in for Stage 4.** If Stage 4 writes a file with the same columns (`district, season, year, condition_detail_id, basis, evidence`) to `outputs/condition_classification/season_conditions.csv`, `build_graph.py` uses that file instead. The graph schema doesn't change. `build_graph.add_experienced_edge()` adds a single label in memory.

## Querying (`kg_query.py`)

```bash
python codes/knowledge_graph/kg_query.py --list-conditions
python codes/knowledge_graph/kg_query.py --district Beed --crop Soybean --condition mid_dry_spell_flowering
python codes/knowledge_graph/kg_query.py --district Solapur --crop Jowar --season Rabi --condition early_season_drought
python codes/knowledge_graph/kg_query.py --district Latur --crop Soybean --season Kharif --condition onset_delay_4w --json
```

`--condition` accepts a detail code (`onset_delay_4w`), a category code (`mid_season_drought`) or a category name.

Python API: `recommend()`, `alternatives()`, `not_applicable()`, `district_profile()`, `season_history()`, `similar_seasons()`, `yield_baseline()`.

**`retrieve_context(G, district, crop, season, condition)`** is the single call for the LLM layer. It returns one JSON-ready bundle with:
- condition definitions
- the district profile
- official recommendations with page citations
- alternative crops
- past seasons with the same condition and the yield that followed
- the all-years yield baseline
- the plan's "not applicable" statements
- the guardrail text

## Neo4j

```bash
python codes/knowledge_graph/export_neo4j.py                     # writes outputs/knowledge_graph/neo4j/
python codes/knowledge_graph/export_neo4j.py --uri bolt://localhost:7687 --password <pw>   # also pushes (pip install neo4j)
```

Copy the CSVs into the database's `import/` folder, then run `cypher-shell -f import.cypher`. In Neo4j Browser, prefix the relationship blocks with `:auto`. Every node has a unique `key`. A Cypher template for the GraphRAG step is in the `export_neo4j.py` docstring. The export has not been loaded into a live Neo4j instance yet.

## Known limitations

- **The source PDFs date from 2011.** Soil, irrigation and rainfall-normal figures are a single snapshot. Ahmednagar publishes no soil %. Latur's net-irrigated % isn't captured.
- **Soil vocabularies.** Section 1.4 soil categories and the soil groups in the Section 2 tables don't line up one-to-one. `INCLUDES` only links names that compose exactly. Latur, Beed "Shallow soils", Solapur "Medium black / light soils" and Ahmednagar's grey soils still need a manual check against the PDFs.
- **Rows not captured.** Rows whose crop cell holds no recognised crop, e.g. Solapur's "Grasses (Marvel, Madras Anjan) / forest tree plantation", are not extracted.
- **Seasons in mixed cells.** A rotation cell such as "Kharif fallow – Rabi sorghum / safflower / sunflower" tags sunflower as Kharif, because only sorghum carries the "Rabi" prefix.
- **Labeller limits.** It uses NASA POWER gridded rainfall, which smooths extremes, so heavy-rain labels are conservative. It cannot detect pest outbreaks, hail or canal releases, so those conditions only come from the plans, never from `EXPERIENCED`. Its thresholds are a transparent first version, not a validated drought index.
- **DES coverage.** The DES table covers Tur, Bajra, Jowar (K/R), Soybean and Sunflower (K/R) for 2000-01 to 2022-23. Gram, Wheat, Cotton and Safflower have no yields, so `HAD_YIELD` is missing for them.

## Files

- `district_name_map.py`: canonical district and crop names, plus spelling aliases.
- `extract_pdfs.py`: PDFs → CSVs.
- `label_seasons.py`: weather → season conditions and season summary.
- `build_graph.py`: CSVs + DES → graph.
- `build_html_viewer.py`: graph → interactive viewer. Actions and season records are hidden by default; tick them in the sidebar. Hovering a node type or relation in the sidebar shows its meaning, the **"What do the nodes and relations mean?"** button opens a full guide (layers, query path, tables of all types and relations with counts), and clicking a node or edge shows what its type means above its details. The descriptions live in `NODE_INFO` / `EDGE_INFO` at the top of the script.
- `export_neo4j.py`: graph → Neo4j CSVs, `import.cypher`, optional push.
- `kg_query.py`: query functions and `retrieve_context()`.
- `run_all.py`: runs the whole pipeline.

Earlier versions of these scripts and outputs are in git (`b690c61`, `0e37ff8`).
