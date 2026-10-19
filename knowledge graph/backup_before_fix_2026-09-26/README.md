# Knowledge graph (Stage 9-10 build)

Implements the schema in `Project_Implementation_Plan.md` Section 7.2, built
from the 5 Agriculture Contingency Plan PDFs plus the DES crop-production
data and district static profiles.

## Pipeline

```
python3 knowledge_graph/scripts/extract_pdfs.py   # PDFs -> data/*.csv
python3 knowledge_graph/scripts/build_graph.py    # data/*.csv -> output/knowledge_graph.{graphml,json}
python3 knowledge_graph/scripts/kg_query.py --district Beed --crop Soybean --condition "Mid-season drought (dry spell)"
```

## Graph contents (current build)

| Node type | Count |
|---|---|
| District | 5 |
| SoilType | 14 |
| Season | 2 (Kharif, Rabi) |
| Crop | 24 (5 core + 4 extended + 15 fallback/horticulture) |
| Condition | 4 (Early season drought, Mid-season drought, Late-season/terminal drought, Unusual rains) |
| Action | 315 (deduplicated recommendation text) |

Edges: `HAS_SOIL` (15), `GROWS` (55), `IN_SEASON` (12), `RECOMMENDS` (330,
Condition->Action), `APPLIES_TO_CROP` (391, Crop->Action).

`RECOMMENDS` edges carry `district`, `situation` (Rainfed/Irrigated),
`soil_type`, `moisture_regime` (assured/low/assured_high), `condition_detail`
(the specific delay/duration text from the PDF row) and `source_page`, so
every recommendation traces back to an exact page in the source PDF.

## Known limitations (flag for manual review, not silent gaps)

- **Floods / Extreme events categories have 0 Action nodes.** Beed's plan
  states "2.3 Floods: Not applicable"; the heat/cold/frost/hail/cyclone
  section in all 5 PDFs is narrative text, not a crop x condition x action
  table, so nothing structured to extract there.
- **SoilType from Section 2.0 tables doesn't line up 1:1 with Section 1.4's
  official soil taxonomy.** E.g. Section 2.0 groups "Medium deep to deep
  black soils" as one farming situation, while Section 1.4 lists "Deep
  black soils" and "Medium deep black soils" as two separate categories.
  This is a genuine inconsistency in the source PDFs, not a parsing bug --
  `HAS_SOIL` edges (from 1.4) and `RECOMMENDS.soil_type` (from 2.0) are two
  related but distinct vocabularies. Worth a manual reconciliation pass if
  soil-type-exact multi-hop queries matter for the final demo.
- **Ahmednagar's soil percentages are missing** (source PDF only publishes
  area in '000 ha for that district, not % of total area, unlike the other
  4). District node still gets `HAS_SOIL` edges with `pct=None`.
- **`condition_detail` is blank on some rows** where a table's severity
  column value spans a page break; pdfplumber's per-page `extract_tables()`
  doesn't forward-fill merged cells across a page boundary.
- crop_production DES data only has real Area data for 5 of the 7 core
  crop-seasons (Arhar/Tur, Bajra, Jowar-Kharif, Jowar-Rabi, Soybean,
  Sunflower-Kharif, Sunflower-Rabi) -- matches the known Gram/Chickpea gap
  in `data_audit_findings.md`. `GROWS` edges for Cotton/Wheat/Safflower/Gram
  come from their appearing in the contingency-plan tables instead.

## Not yet included: Year nodes / EXPERIENCED edges

Per the schema, `(District, Season, Year) -EXPERIENCED-> Condition` needs
Stage 4's rule-based SPI/tercile drought classifier (still at plan stage,
owned by the yield-model track) to emit one of this graph's 4 Condition
categories per district-year-season from the weather CSVs. Once that
function exists, call `kg_query.recommend()` with its output directly --
no schema change needed, just new edges from Stage 4 into this same graph.

## Files

- `scripts/district_name_map.py` -- canonical district/crop name + alias
  tables (matches Section 5's crosswalk).
- `scripts/extract_pdfs.py` -- PDF -> `data/district_static_features.csv`,
  `data/contingency_actions.csv`.
- `scripts/build_graph.py` -- CSVs -> `output/knowledge_graph.graphml` /
  `.json`.
- `scripts/kg_query.py` -- Stage 9 lookup (the template for what Stage 10's
  LLM layer calls).
