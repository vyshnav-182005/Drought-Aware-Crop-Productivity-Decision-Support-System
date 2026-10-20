# Knowledge graph (Stage 9-10 build)

Implements the schema in `Project_Implementation_Plan.md` Section 7.2, built
from the 5 Agriculture Contingency Plan PDFs plus the DES crop-production
data and district static profiles.

## Pipeline

```
python3 codes/knowledge_graph/extract_pdfs.py       # data/contingency_plans/*.pdf -> outputs/knowledge_graph/*.csv
python3 codes/knowledge_graph/build_graph.py        # CSVs -> outputs/knowledge_graph/knowledge_graph.{graphml,json}
python3 codes/knowledge_graph/build_html_viewer.py  # -> outputs/knowledge_graph/knowledge_graph_viewer.html
python3 codes/knowledge_graph/kg_query.py --district Beed --crop Soybean --condition "Mid-season drought (dry spell)"
```

## Graph contents (current build)

| Node type | Count |
|---|---|
| District | 5 |
| SoilType | 12 (8 from Section 1.4 soil tables + 4 Section 2.0 farming-situation soil groups not named identically in 1.4) |
| Season | 2 (Kharif, Rabi) |
| Crop | 24 (5 core + 4 extended + 15 fallback/horticulture) |
| Condition | 4 (Early season drought, Mid-season drought, Late-season/terminal drought, Unusual rains) |
| Action | 326 (deduplicated recommendation text, after resolving "-do-") |

Edges (1437): `HAS_SOIL` (15), `GROWS` (55), `IN_SEASON` (12),
`RECOMMENDS` (564, Condition->Action, **exactly one per contingency-table
row**), `APPLIES_TO_CROP` (420, Crop->Action), `SOIL_CONTEXT` (365,
SoilType->Action), `INCLUDES` (6, Section 2.0 soil group -> Section 1.4
soil category). No node is isolated; `build_graph.py` prints an
isolated-node check on every run.

Every edge has a `relation` attribute naming its type. Use that, not the
edge key, to filter: `RECOMMENDS` keys are `RECOMMENDS:<row>` and
`INCLUDES` keys are `INCLUDES:<district>` so that parallel edges don't
overwrite each other.

`RECOMMENDS` edges carry `district`, `situation` (Rainfed/Irrigated),
`soil_type` (normalized), `soil_type_raw` (exact PDF text),
`moisture_regime` (assured/low/assured_high), `condition_detail`,
`ditto_resolved` and `source_page`, so every recommendation traces back to
an exact page in the source PDF.

### Credibility fixes (2026-09-26)

Pre-fix outputs are in git commit `b690c61` (`knowledge graph/backup_before_fix_2026-09-26/`).

1. **Lost recommendations.** `RECOMMENDS` used a fixed edge key, so when
   the same action text was recommended for the same condition in another
   district or on another page, the later row overwrote the earlier one's
   provenance. 564 rows had collapsed to 330 edges (Solapur kept 38 of 88,
   Beed 94 of 164), and `kg_query.recommend()` silently missed those rows.
   Now one edge per row. Verified: (district, page, crop) counts in the
   graph match the CSV exactly.
2. **"-do-" (ditto) cells.** 208 rows (37%) had "-do-" in an action
   column, and all of them merged into one meaningless Action node. Each
   "-do-" is now replaced with the last real value of that column in the
   same (district, condition) table, in PDF row order, and the edge
   records which columns were resolved. Spot-checked against Beed pp.
   18-19. Four cells have no row above them in their table and are left as
   `unresolved:<col>` rather than guessed (Ahmednagar pp. 14-15 remarks,
   Osmanabad p. 25 agronomic measures). Partial dittos such as "-do- or if
   the plant population is less than 50% resow" are kept verbatim.
3. **Orphan SoilType nodes.** Section 2.0 soil names were created as nodes
   but only referenced as edge attributes, which left 6 isolated nodes.
   Added `SOIL_CONTEXT` edges (SoilType->Action).
4. **Soil spelling variants merged** (`SOIL_ALIASES`): "Medium deep to
   deep *and* black soils" (typo) and "... black *cotton* soil" (common
   name for black soil) now map to "Medium Deep To Deep Black Soils". The
   raw text is kept in `soil_type_raw`.
5. **Soil crosswalk** (`SOIL_CROSSWALK`): `INCLUDES` edges link a Section
   2.0 soil group to Section 1.4 categories only where the names compose
   exactly (e.g. "medium deep to deep black" includes "Deep Black Soils"
   in Beed, Osmanabad and Solapur). Each edge is tagged
   `basis=name_composition` because this is derived, not stated in the
   PDF.
6. **GROWS provenance.** The contingency-plan source used to overwrite
   `source="DES"`. `source` is now ";"-joined (e.g. `DES;contingency_plan`).
7. **Viewer.** New option (on by default) hides nodes that have no visible
   edge under the current filters, so nodes no longer look orphaned just
   because their edges are filtered out.

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
  related but distinct vocabularies. `INCLUDES` now bridges the
  exact-match pairs. Still unlinked, pending a manual check against the
  PDFs: Latur (1.4 "Deep/Medium Deep/Shallow Soils", no "black"; 2.0
  "Medium deep to deep black", "Shallow black"), Beed 2.0 "Shallow soils"
  vs 1.4 "Shallow Black Soils", Solapur 1.4 "Medium Black Soils" vs 2.0
  "medium deep", and Solapur "Light soils".
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

- `codes/knowledge_graph/district_name_map.py` -- canonical district/crop name + alias
  tables (matches Section 5's crosswalk).
- `codes/knowledge_graph/extract_pdfs.py` -- PDFs in `data/contingency_plans/` ->
  `outputs/knowledge_graph/district_static_features.csv`, `contingency_actions.csv`.
- `codes/knowledge_graph/build_graph.py` -- CSVs -> `outputs/knowledge_graph/knowledge_graph.graphml` /
  `.json`.
- `codes/knowledge_graph/kg_query.py` -- Stage 9 lookup (the template for what Stage 10's
  LLM layer calls).
