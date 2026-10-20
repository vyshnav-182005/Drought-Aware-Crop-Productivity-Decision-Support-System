"""
Extract knowledge-graph source data from the 5 Agriculture Contingency Plan PDFs.

Produces two structured intermediate files (per Project_Implementation_Plan.md
Section 5, steps 6-7):
  - outputs/knowledge_graph/district_static_features.csv
      (district, soil_type, soil_area_kha, soil_pct, rainfall_normal_mm,
       net_irrigated_pct)
  - outputs/knowledge_graph/contingency_actions.csv
      (district, condition_category, situation, condition_detail, soil_type,
       crop, change_in_cropping, agronomic_measures, remarks, source_page)

Run from the project root:  python3 codes/knowledge_graph/extract_pdfs.py

Notes on data quality (this is a one-time semi-automated ETL over 5
differently-formatted government PDFs, per the project plan's own framing --
some manual spot-checking of the output CSVs is expected, not a bug):
  - Ahmednagar's soil-area table has no percent-of-area column in the source
    PDF (only area in '000 ha), unlike the other 4 districts.
  - Solapur's soil lines aren't numbered ("1.", "2.", ...) unlike the others.
  - condition_category / situation are assigned from the nearest preceding
    section heading; a table with no district-specific heading above it
    (rare) falls back to "Unclassified" / "Unspecified".
"""
import csv
import re
import sys
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).parent))
from district_name_map import (  # noqa: E402
    CONTINGENCY_PDF_DISTRICT,
    split_and_normalize_crops,
)

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "data" / "contingency_plans"
OUT_DIR = ROOT / "outputs" / "knowledge_graph"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "")


# ---------------------------------------------------------------------------
# Section 1: static district profile (soil types, rainfall normal, irrigation)
# ---------------------------------------------------------------------------

SOIL_LINE_RE = re.compile(
    r"^\d*\.?\s*([A-Za-z][A-Za-z /]+?soils?)\s+([\d.]+)\s+([\d.]+)\s*$"
)
RAINFALL_RE = re.compile(
    r"SW monsoon\s*\(\s*June\s*-?\s*Sep\s*\)\s*:?\s*([\d.]+)", re.I
)
NET_IRRIGATED_RE = re.compile(
    r"Net\s*[Ii]rrigated area\s+([\d.]+)\s+([\d.]+)"
)


def extract_static_features_regex(header_text: str, district: str, rainfall_normal, net_irrigated_pct):
    rows = []
    for line in header_text.split("\n"):
        m = SOIL_LINE_RE.match(line.strip())
        if m:
            soil_name, area_kha, pct = m.groups()
            rows.append({
                "district": district,
                "soil_type": soil_name.strip().title(),
                "soil_area_kha": float(area_kha),
                "soil_pct": float(pct),
                "rainfall_normal_mm": rainfall_normal,
                "net_irrigated_pct": net_irrigated_pct,
            })
    return rows


def extract_static_features_table_fallback(pdf, district, rainfall_normal, net_irrigated_pct):
    """Fallback for PDFs (e.g. Ahmednagar) whose soil table doesn't extract as
    clean text lines. Looks for a table whose first cell mentions 'Major Soil'."""
    rows = []
    for page in pdf.pages[:4]:
        for table in page.extract_tables():
            if not table:
                continue
            first_cell = norm_ws(table[0][0] or "")
            if "major soil" not in first_cell.lower():
                continue
            soil_names = [s.strip() for s in (table[0][0] or "").split("\n")
                          if s.strip() and "major soil" not in s.lower() and not re.match(r"^\d", s.strip())]
            areas = []
            for r in table[1:]:
                for cell in r[1:]:
                    if cell and re.match(r"^[\d.]+$", cell.strip()):
                        areas.append(float(cell.strip()))
                        break
            for name, area in zip(soil_names, areas):
                rows.append({
                    "district": district,
                    "soil_type": name.title(),
                    "soil_area_kha": area,
                    "soil_pct": None,  # not published for this district
                    "rainfall_normal_mm": rainfall_normal,
                    "net_irrigated_pct": net_irrigated_pct,
                })
    return rows


def extract_static_features(pdf, district) -> list:
    header_text = "\n".join((pdf.pages[i].extract_text() or "") for i in range(min(4, len(pdf.pages))))

    rainfall_normal = None
    m = RAINFALL_RE.search(header_text)
    if m:
        rainfall_normal = float(m.group(1))

    net_irrigated_pct = None
    m = NET_IRRIGATED_RE.search(header_text)
    if m:
        net_irrigated_pct = float(m.group(2))

    rows = extract_static_features_regex(header_text, district, rainfall_normal, net_irrigated_pct)
    if not rows:
        rows = extract_static_features_table_fallback(pdf, district, rainfall_normal, net_irrigated_pct)
    if not rows:
        rows.append({
            "district": district, "soil_type": None, "soil_area_kha": None,
            "soil_pct": None, "rainfall_normal_mm": rainfall_normal,
            "net_irrigated_pct": net_irrigated_pct,
        })
    return rows


# ---------------------------------------------------------------------------
# Section 2: condition -> action tables (Section 2.0 of each plan)
# ---------------------------------------------------------------------------

CATEGORY_PATTERNS = [
    (re.compile(r"mid[- ]?season drought", re.I), "Mid-season drought (dry spell)"),
    (re.compile(r"late season drought|terminal drought", re.I), "Late-season/terminal drought"),
    (re.compile(r"early season drought", re.I), "Early season drought (delayed onset)"),
    (re.compile(r"unusual rains", re.I), "Unusual rains"),
    (re.compile(r"2\.3\s*floods", re.I), "Floods"),
    (re.compile(r"extreme events", re.I), "Extreme events (heat/cold/frost/hail/cyclone)"),
]
SITUATION_PATTERNS = [
    (re.compile(r"rainfed situation", re.I), "Rainfed"),
    (re.compile(r"irrigated situation", re.I), "Irrigated"),
]
STOP_PATTERN = re.compile(r"2\.5\s*Contingent strategies for Livestock", re.I)

HEADER_CELL_WORDS = {
    "condition", "normal", "crop/cropping", "suggested", "farming", "situation",
    "remarks", "agronomic", "implementation", "change", "system", "measures",
    "including", "variety", "management", "soil", "nutrient", "conservation",
    "moisture", "contingency",
}


def is_header_row(row: list) -> bool:
    cells = [c for c in row if c]
    if not cells:
        return True
    text = " ".join(cells).lower()
    words = set(re.findall(r"[a-z/]+", text))
    return len(words & HEADER_CELL_WORDS) >= 3


def extract_condition_actions(pdf, district: str) -> list:
    records = []
    category = None
    situation = None
    started = False

    for pg_idx, page in enumerate(pdf.pages):
        raw_text = page.extract_text() or ""
        text = norm_ws(raw_text)
        if not started and "2.0 Strategies" in text:
            started = True
        if not started:
            continue
        if STOP_PATTERN.search(text):
            break

        # Situation ("Rainfed situation"/"Irrigated situation") and the
        # non-drought categories (Unusual rains/Floods/Extreme events) are
        # standalone heading lines above their tables and extract cleanly
        # from page-level text.
        for pat, label in SITUATION_PATTERNS:
            if pat.search(text):
                situation = label
        for pat, label in CATEGORY_PATTERNS:
            if pat.search(text) and label in (
                "Unusual rains", "Floods", "Extreme events (heat/cold/frost/hail/cyclone)"
            ):
                category = label

        tables = page.extract_tables()
        for table in tables:
            # Condition category text lives inside a table header cell, and
            # page-level linear text extraction interleaves it with sibling
            # column headers (multi-column tables read left-to-right by
            # position, not by cell) -- so scan header ROW CELLS directly
            # instead of the mangled page text.
            for row in table:
                if not is_header_row(row):
                    break
                for cell in row:
                    cell_norm = norm_ws(cell)
                    for pat, label in CATEGORY_PATTERNS:
                        if pat.search(cell_norm):
                            category = label

            last_condition_detail = None
            last_soil_type = None
            for row in table:
                if is_header_row(row):
                    continue
                row = (row + [None] * 6)[:6]
                col0, col1, col2, col3, col4, col5 = row

                if col0 and col0.strip():
                    last_condition_detail = norm_ws(col0)
                if col1 and col1.strip():
                    last_soil_type = norm_ws(col1)

                crop_raw = norm_ws(col2)
                crops = split_and_normalize_crops(crop_raw)
                if not crops:
                    continue

                change = norm_ws(col3)
                agronomic = norm_ws(col4)
                remarks = norm_ws(col5)
                if not (change or agronomic or remarks):
                    continue

                for crop in crops:
                    records.append({
                        "district": district,
                        "condition_category": category or "Unclassified",
                        "situation": situation or "Unspecified",
                        "condition_detail": last_condition_detail,
                        "soil_type": last_soil_type,
                        "crop": crop,
                        "change_in_cropping": change,
                        "agronomic_measures": agronomic,
                        "remarks": remarks,
                        "source_page": pg_idx + 1,
                    })
    return records


def main():
    static_rows, action_rows = [], []

    for fname, district in CONTINGENCY_PDF_DISTRICT.items():
        path = PDF_DIR / fname
        if not path.exists():
            print(f"WARNING: missing {path}", file=sys.stderr)
            continue
        print(f"Processing {district}: {fname}")
        with pdfplumber.open(path) as pdf:
            static_rows.extend(extract_static_features(pdf, district))
            action_rows.extend(extract_condition_actions(pdf, district))

    with open(OUT_DIR / "district_static_features.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "district", "soil_type", "soil_area_kha", "soil_pct",
            "rainfall_normal_mm", "net_irrigated_pct",
        ])
        w.writeheader()
        w.writerows(static_rows)

    with open(OUT_DIR / "contingency_actions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "district", "condition_category", "situation", "condition_detail",
            "soil_type", "crop", "change_in_cropping",
            "agronomic_measures", "remarks", "source_page",
        ])
        w.writeheader()
        w.writerows(action_rows)

    print(f"\ndistrict_static_features.csv: {len(static_rows)} rows")
    print(f"contingency_actions.csv: {len(action_rows)} rows")


if __name__ == "__main__":
    main()
