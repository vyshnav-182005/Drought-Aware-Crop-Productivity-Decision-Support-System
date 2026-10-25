"""
Extract knowledge-graph source data from the 5 Agriculture Contingency Plan PDFs.

Outputs (outputs/knowledge_graph/):
  - district_static_features.csv  Section 1: soil types (area, %), SW-monsoon
                                    rainfall normal, net irrigated %.
  - contingency_actions.csv        Section 2.1-2.4: one row per (PDF table row,
                                    crop[, crop stage]).
  - not_applicable.csv             Conditions a plan explicitly marks
                                    "Not applicable" for its district.

Run from the project root:  python3 codes/knowledge_graph/extract_pdfs.py

How the Section 2 tables are read
---------------------------------
The plans use several table layouts, so columns are mapped from each table's
OWN header row, never by position alone:

  2.1.1 rainfed drought, delayed onset:  Change in crop | Agronomic measures | Remarks
  2.1.1 normal onset / mid-season:       Crop management | Soil nutrient & moisture conservation | Remarks
  2.1.1 terminal drought:                Crop management | Rabi crop planning | Remarks
  2.1.2 irrigated (canal/tank/well):     Change in crop | Agronomic measures | Remarks
  2.2 unusual rains, 2.3 floods,
  2.4 extreme events:                    one column per crop stage (vegetative,
                                         flowering, maturity, post-harvest ...)

* Wide tables (Solapur, Osmanabad p.24) have spanned header cells; an empty
  header cell inherits the role of the column to its left.
* Solapur's 8-column tables carry a "Kharif | Rabi" sub-header under the crop
  and change columns; each half becomes its own row with that season.
* A table with no header row (continued from the previous page) reuses the
  previous table's column map when the widths match.
* A row with no crop, soil or condition text is a cell that overflowed the
  page; its text is appended to the previous row instead of being dropped.
* The condition detail ("Delay by 4 weeks", "At flowering stage") is carried
  forward across page breaks within the same condition block, and split
  lines ("Delay by 4 weeks" / "2nd week of July") are re-joined.
* Irrigated-situation tables (canal / tank / groundwater shortages) get
  their own category instead of inheriting the previous drought heading.
"""
import csv
import re
import sys
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).parent))
from district_name_map import CONTINGENCY_PDF_DISTRICT, CROP_ALIASES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "data" / "contingency_plans"
OUT_DIR = ROOT / "outputs" / "knowledge_graph"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def norm_ws(text) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# ---------------------------------------------------------------------------
# Section 1: static district profile (unchanged logic)
# ---------------------------------------------------------------------------

SOIL_LINE_RE = re.compile(r"^\d*\.?\s*([A-Za-z][A-Za-z /]+?soils?)\s+([\d.]+)\s+([\d.]+)\s*$")
RAINFALL_RE = re.compile(r"SW monsoon\s*\(\s*June\s*-?\s*Sep\s*\)\s*:?\s*([\d.]+)", re.I)
NET_IRRIGATED_RE = re.compile(r"Net\s*[Ii]rrigated area\s+([\d.]+)\s+([\d.]+)")


def extract_static_features(pdf, district) -> list:
    header_text = "\n".join((pdf.pages[i].extract_text() or "") for i in range(min(4, len(pdf.pages))))
    m = RAINFALL_RE.search(header_text)
    rainfall_normal = float(m.group(1)) if m else None
    m = NET_IRRIGATED_RE.search(header_text)
    net_irrigated_pct = float(m.group(2)) if m else None

    rows = []
    for line in header_text.split("\n"):
        m = SOIL_LINE_RE.match(line.strip())
        if m:
            name, area, pct = m.groups()
            rows.append({"district": district, "soil_type": name.strip().title(),
                         "soil_area_kha": float(area), "soil_pct": float(pct),
                         "rainfall_normal_mm": rainfall_normal, "net_irrigated_pct": net_irrigated_pct})
    if not rows:  # Ahmednagar: soil table only extracts as a table, no % column
        for page in pdf.pages[:4]:
            for table in page.extract_tables():
                if not table or "major soil" not in norm_ws(table[0][0]).lower():
                    continue
                names = [s.strip() for s in (table[0][0] or "").split("\n")
                         if s.strip() and "major soil" not in s.lower() and not re.match(r"^\d", s.strip())]
                areas = []
                for r in table[1:]:
                    for cell in r[1:]:
                        if cell and re.match(r"^[\d.]+$", cell.strip()):
                            areas.append(float(cell.strip()))
                            break
                for name, area in zip(names, areas):
                    rows.append({"district": district, "soil_type": name.title(), "soil_area_kha": area,
                                 "soil_pct": None, "rainfall_normal_mm": rainfall_normal,
                                 "net_irrigated_pct": net_irrigated_pct})
    if not rows:
        rows.append({"district": district, "soil_type": None, "soil_area_kha": None, "soil_pct": None,
                     "rainfall_normal_mm": rainfall_normal, "net_irrigated_pct": net_irrigated_pct})
    return rows


# ---------------------------------------------------------------------------
# Section 2: condition -> action tables
# ---------------------------------------------------------------------------

CAT_EARLY = "Early season drought (delayed onset)"
CAT_MID = "Mid-season drought (dry spell)"
CAT_LATE = "Late-season/terminal drought"
CAT_IRRIG = "Drought - irrigated situation (water shortage)"
CAT_UNUSUAL = "Unusual rains"
CAT_FLOOD = "Floods"
CAT_EXTREME = "Extreme events"

CATEGORY_PATTERNS = [
    (re.compile(r"mid[- ]?season drought", re.I), CAT_MID),
    (re.compile(r"late season drought|terminal drought", re.I), CAT_LATE),
    (re.compile(r"early season drought", re.I), CAT_EARLY),
]
IRRIGATION_DETAIL_RE = re.compile(r"canal|tank|groundwater|ground water", re.I)

# Header text -> column role. Checked in this order (first match wins).
ROLE_KEYWORDS = [
    ("rabicropplanning", "rabi_crop_planning"),
    ("soilnutrient", "soil_moisture_conservation"),
    ("cropmanagement", "crop_management"),
    ("agronomic", "agronomic_measures"),
    ("changeincrop", "change_in_cropping"),
    ("remarks", "remarks"),
    ("farmingsituation", "soil_type"),
    ("crop/cropping", "crop"),
    ("condition", "condition_detail"),
]
ROLE_RANK = {"condition_detail": 0, "soil_type": 1, "crop": 2,
             "change_in_cropping": 3, "crop_management": 3,
             "agronomic_measures": 4, "soil_moisture_conservation": 4, "rabi_crop_planning": 4,
             "remarks": 5}
TEXT_ROLES = ["change_in_cropping", "agronomic_measures", "crop_management",
              "soil_moisture_conservation", "rabi_crop_planning", "remarks"]
DEFAULT_6 = {
    "change": ["condition_detail", "soil_type", "crop", "change_in_cropping", "agronomic_measures", "remarks"],
    "crop_management": ["condition_detail", "soil_type", "crop", "crop_management",
                        "soil_moisture_conservation", "remarks"],
    "terminal": ["condition_detail", "soil_type", "crop", "crop_management", "rabi_crop_planning", "remarks"],
}
# Solapur 8-col tables with a Kharif|Rabi sub-header and no role header.
SEASON_SPLIT_8 = ["condition_detail", "soil_type", "crop", "crop", "change_in_cropping",
                  "change_in_cropping", "agronomic_measures", "remarks"]

STAGE_KEYWORDS = [
    ("seedling", "seedling/nursery"), ("nursery", "seedling/nursery"),
    ("vegetative", "vegetative"), ("flowering", "flowering"), ("reproductive", "reproductive"),
    ("maturity", "maturity"), ("post harvest", "post-harvest"), ("at harvest", "harvest"),
]
GROUP_ROWS = {"horticulture": "horticulture", "vegetable crops": "vegetable", "vegetables": "vegetable"}
NA_RE = re.compile(r"^(not applicable|na|n\.a\.?|-+)$", re.I)
DITTO_CELL = re.compile(r"^\s*(?:[-\u2013]*\s*do\s*[-\u2013]*\.?|as above\.?)\s*$", re.I)
STOP_PATTERN = re.compile(r"2\.5\s*Contingent strategies for Livestock", re.I)

# Condition-detail lines that START a new detail; anything else in the
# condition column continues the previous line ("2nd week of July").
DETAIL_START_RE = re.compile(
    r"^(delay|normal onset|mid ?season|at |terminal|non[- ]release|lack|insufficient|delayed|limited|"
    r"continuous|heavy|outbreak|transient|sea water|heat|cold|frost|hail|cyclone)", re.I)


def unusual_category(text: str):
    t = text.lower()
    if re.search(r"heat|cold wave|frost|hail|cyclone|extreme event", t):
        return CAT_EXTREME
    if re.search(r"inundat|submergence|sea water", t):
        return CAT_FLOOD
    if re.search(r"rain|pest|disease", t):   # "high rainfall ... leading to water logging" is 2.2
        return CAT_UNUSUAL
    if re.search(r"water ?logging", t):
        return CAT_FLOOD
    return None


def header_key(cell) -> str:
    return re.sub(r"\s+", "", (cell or "").lower()).replace("crop/croppingsystem", "crop/cropping")


def role_of(cell):
    k = header_key(cell)
    if not k:
        return None
    for kw, role in ROLE_KEYWORDS:
        if kw in k:
            return role
    return None


def build_role_map(header_rows):
    """Column roles from the table's header row(s). Some plans split the
    header over two rows ("Major farming situation | Normal crop" on one,
    "Crop management | Rabi crop planning" on the next), so each column takes
    the role from whichever header row names it. Spanned (empty) cells
    inherit from the left; out-of-order roles (a mis-extracted header, e.g.
    Ahmednagar pp.17-18 repeating 'Major farming situation' in the agronomic
    column) fall back to the positional default."""
    width = max(len(r) for r in header_rows)
    roles = [None] * width
    for row in header_rows:
        for i, c in enumerate(row):
            r = role_of(c)
            if r and r != "condition_detail":
                roles[i] = r
    roles[0] = "condition_detail"
    for i in range(1, width):
        if roles[i] is None:
            roles[i] = roles[i - 1]
    keys = " ".join(header_key(c) for row in header_rows for c in row)
    scheme = ("terminal" if "rabicropplanning" in keys else
              "crop_management" if "cropmanagement" in keys else "change")
    max_rank = -1
    for i, r in enumerate(roles):
        rank = ROLE_RANK.get(r, -1)
        if rank < max_rank and r != roles[i - 1]:
            roles[i] = DEFAULT_6[scheme][i] if width == 6 else None
            rank = ROLE_RANK.get(roles[i], -1)
        max_rank = max(max_rank, rank)
    return roles, scheme


def find_role_header(table):
    """Index of the LAST header row (rows naming a column role, or the
    'Condition | Suggested contingency measures' super-header), or None if
    the table has no 'Major farming situation' header at all."""
    if not any(role_of(c) == "soil_type" for row in table[:4] for c in row):
        return None
    last = None
    for i, row in enumerate(table[:4]):
        named = [role_of(c) for c in row[1:]]
        if any(r and r != "condition_detail" for r in named) or \
                any("suggested contingency" in (c or "").lower() for c in row):
            last = i
        elif last is not None:
            break
    return last


def find_stage_header(table):
    for i, row in enumerate(table[:3]):
        stages = [s for c in row[1:] for kw, s in STAGE_KEYWORDS if kw in (c or "").lower()]
        if len(stages) >= 2:
            return i
    return None


def stage_map(row):
    out = [None]
    for c in row[1:]:
        t = (c or "").lower()
        out.append(next((s for kw, s in STAGE_KEYWORDS if kw in t), None))
    return out


def season_subheader(row):
    cells = [norm_ws(c).lower() for c in row]
    nonempty = [c for c in cells if c]
    if len(nonempty) >= 2 and all(c in ("kharif", "rabi") for c in nonempty):
        return [c.title() if c else None for c in cells]
    return None


_ALIAS_RE = re.compile(
    r"\b(?:(kharif|rabi|summer)[\s-]+)?(" +
    "|".join(sorted((re.escape(a) for a in CROP_ALIASES), key=len, reverse=True)) + r")\b", re.I)


def crops_with_season(raw: str):
    """(canonical crop, season prefix or '') pairs found in a crop cell.
    Splits on / , + first; a part that is not a bare crop name (e.g. Solapur's
    space-separated "Kharif Pearlmillet Preseasonal sugarcane Maize") is
    scanned for known crop names. Variety lists in parentheses are ignored."""
    out = []
    text = re.sub(r"\([^)]*\)?", " ", (raw or "").replace("&", "/"))
    for part in re.split(r"[/,+]| and ", text):
        part = part.strip()
        found = [(CROP_ALIASES[m.group(2).lower()], (m.group(1) or "").title())
                 for m in _ALIAS_RE.finditer(part)]
        for crop, season in found:
            if crop not in [c for c, _ in out]:
                out.append((crop, season))
    return out


class Extractor:
    def __init__(self, district):
        self.district = district
        self.records, self.not_applicable = [], []
        self.category, self.situation = None, "Rainfed"
        self.detail = None
        self.role_map, self.scheme, self.width = None, None, None
        self.stage_cols, self.stage_detail = None, None
        self.crop_group = "field"
        self.last_row_records = []

    # -- helpers ------------------------------------------------------------
    def set_detail(self, text):
        # drop repeated header fragments that leak into the condition column
        text = re.sub(r"\bcondition\b|any other condition \(specify\)|early season drought\s*\(\s*(delayed|normal) onset\s*\)",
                      " ", norm_ws(text), flags=re.I)
        text = norm_ws(text)
        if not text:
            return
        if self.detail and not DETAIL_START_RE.match(text):
            self.detail = f"{self.detail} {text}"          # continuation line
            for r in self.last_row_records:
                if r["condition_category"] == self.category:
                    r["condition_detail"] = self.detail
        else:
            self.detail = text
        if IRRIGATION_DETAIL_RE.search(self.detail):
            self.category, self.situation = CAT_IRRIG, "Irrigated"

    def append_overflow(self, row, roles):
        for r in self.last_row_records:
            for i, c in enumerate(row):
                role = roles[i] if i < len(roles) else None
                c = norm_ws(c)
                if c and role in TEXT_ROLES:
                    r[role] = f"{r[role]} {c}".strip()
            r["continued_on_page"] = self.page

    # -- drought tables (2.1.1 / 2.1.2) ---------------------------------------
    def drought_table(self, table):
        hdr = find_role_header(table)
        season_cols = None
        if hdr is not None:
            self.last_row_records = []   # a new header: never append into the previous table
            # category + detail can sit in the header rows' first cell
            for row in table[:hdr + 1]:
                c0 = norm_ws(row[0])
                for pat, label in CATEGORY_PATTERNS:
                    if pat.search(c0):
                        if label != self.category:
                            self.detail = None
                        self.category, self.situation = label, "Rainfed"
                rest = re.sub(r"^condition\s*", "", c0, flags=re.I)
                if rest and IRRIGATION_DETAIL_RE.search(rest):
                    self.detail = None
                    self.set_detail(rest)
                elif self.category == CAT_LATE and re.search(r"terminal|withdrawal", c0, re.I):
                    self.detail = None
                    self.set_detail(re.sub(r"^condition\s*", "", c0, flags=re.I))
            self.role_map, self.scheme = build_role_map(table[:hdr + 1])
            self.width = len(table[hdr])
            body = table[hdr + 1:]
        else:
            width = len(table[0])
            sub = next((season_subheader(r) for r in table[:2] if season_subheader(r)), None)
            if sub and width == 8:
                self.role_map, self.width, season_cols = SEASON_SPLIT_8, 8, sub
            elif self.role_map is None or width != self.width:
                return 0
            body = table

        roles = self.role_map
        n = 0
        for row in body:
            row = list(row) + [None] * (len(roles) - len(row))
            sub = season_subheader(row)
            if sub:
                season_cols = sub
                continue
            vals = {}
            for i, c in enumerate(row[:len(roles)]):
                if roles[i] and norm_ws(c):
                    vals.setdefault(roles[i], []).append((i, norm_ws(c)))
            if not vals:
                continue
            det = " ".join(v for _, v in vals.get("condition_detail", []))
            soil = " ".join(v for _, v in vals.get("soil_type", []))
            crop_cells = vals.get("crop", [])
            row_text = " ".join(norm_ws(c) for c in row[1:])
            if det and not crop_cells and (bool(soil and NA_RE.match(soil))
                                            or bool(re.search(r"not applicable", row_text, re.I))):
                self.set_detail(det)
                self.not_applicable.append({"district": self.district, "condition_category": self.category,
                                            "condition_detail": self.detail, "source_page": self.page})
                continue
            if det:
                for pat, label in CATEGORY_PATTERNS:   # e.g. "Terminal drought" written in the body
                    if pat.search(det) and label != self.category:
                        self.category, self.situation, self.detail = label, "Rainfed", None
                self.set_detail(det)
            if soil and not NA_RE.match(soil):
                # "with assured rainfall" at the top of a page continues the soil cell above
                if re.match(r"^(with\b|\(|-+\s*open well)", soil, re.I) and getattr(self, "soil", None) \
                        and not (det and DETAIL_START_RE.match(det)):
                    self.soil = f"{self.soil} {soil}"
                else:
                    self.soil = soil
            if not crop_cells:
                if not det and not soil and self.last_row_records:
                    self.append_overflow(row, roles)
                continue
            if self.category is None:
                continue
            if not any(vals.get(role) for role in TEXT_ROLES):
                continue   # crop listed with every action cell blank in the PDF (e.g. Beed p.19)

            new_records = []
            for col, cell in crop_cells:
                col_season = season_cols[col] if season_cols and col < len(season_cols) else ""
                for crop, season in crops_with_season(cell):
                    rec = {"district": self.district, "condition_category": self.category,
                           "condition_detail": self.detail, "situation": self.situation,
                           "soil_type": getattr(self, "soil", None), "crop": crop, "crop_raw": cell,
                           "crop_group": "field", "season_hint": season or (col_season or ""),
                           "crop_stage": "", "table_scheme": self.scheme if hdr is not None or not season_cols
                           else "change_season_split", "source_page": self.page, "continued_on_page": ""}
                    for role in TEXT_ROLES:
                        parts = [v for i, v in vals.get(role, [])
                                 if not (season_cols and season_cols[i] and col_season
                                         and season_cols[i] != col_season)]
                        rec[role] = " ".join(parts)
                    new_records.append(rec)
            if new_records:
                self.records.extend(new_records)
                self.last_row_records = new_records
                n += len(new_records)
        return n

    # -- stage tables (2.2 unusual rains / 2.3 floods / 2.4 extreme events) --
    def stage_table(self, table, page_category):
        hdr = find_stage_header(table)
        if hdr is not None:
            self.stage_cols = stage_map(table[hdr])
            c0 = norm_ws(table[hdr][0])
            cat = unusual_category(c0) if c0 else None
            if cat and not (page_category == CAT_EXTREME and cat != CAT_EXTREME):
                self.stage_detail, self.category = c0, cat
            else:
                self.category = page_category
            self.crop_group = "field"
            body = table[hdr + 1:]
        elif self.stage_cols and len(table[0]) == len(self.stage_cols):
            body = table
        else:
            return 0
        n = 0
        for row in body:
            c0 = norm_ws(row[0])
            rest = [norm_ws(c) for c in row[1:]]
            if c0.lower() in GROUP_ROWS:
                self.crop_group = GROUP_ROWS[c0.lower()]
                continue
            if c0.lower().startswith("extreme event type") or c0.lower() == "condition":
                continue
            cat = unusual_category(c0) if c0 else None
            crops = crops_with_season(c0) if c0 else []
            if c0 and cat and not crops:
                self.category = cat if not (page_category == CAT_EXTREME and cat != CAT_EXTREME) else CAT_EXTREME
                self.stage_detail = c0.rstrip("3")
                self.crop_group = "field"
                if rest and NA_RE.match(rest[0] or "-") and not any(rest[1:]):
                    if rest[0]:
                        self.not_applicable.append({"district": self.district, "condition_category": self.category,
                                                    "condition_detail": self.stage_detail, "source_page": self.page})
                continue
            if not c0:
                if self.last_row_records:   # overflow from the row above
                    for r in self.last_row_records:
                        i = self.stage_cols.index(r["crop_stage"]) if r["crop_stage"] in self.stage_cols else None
                        if i and i < len(row) and norm_ws(row[i]):
                            r["crop_management"] = f"{r['crop_management']} {norm_ws(row[i])}".strip()
                            r["continued_on_page"] = self.page
                continue
            if not crops or self.category is None:
                continue
            new_records = []
            prev_txt = None
            for i, txt in enumerate(rest, start=1):
                stage = self.stage_cols[i] if i < len(self.stage_cols) else None
                row_ditto = ""
                # "As above" right of a filled cell repeats that cell (same crop,
                # previous stage); otherwise build_graph resolves it from the crop above.
                if txt and DITTO_CELL.match(txt) and prev_txt:
                    txt, row_ditto = prev_txt, "left"
                if txt and not DITTO_CELL.match(txt) and not NA_RE.match(txt):
                    prev_txt = txt
                if not stage or not txt or NA_RE.match(txt):
                    continue
                for crop, season in crops:
                    rec = {"district": self.district, "condition_category": self.category,
                           "condition_detail": self.stage_detail, "situation": "Rainfed and irrigated",
                           "soil_type": None, "crop": crop, "crop_raw": c0, "crop_group": self.crop_group,
                           "season_hint": season, "crop_stage": stage, "table_scheme": "stage",
                           "source_page": self.page, "continued_on_page": ""}
                    for role in TEXT_ROLES:
                        rec[role] = ""
                    rec["crop_management"] = txt
                    rec["row_ditto"] = row_ditto
                    new_records.append(rec)
            if new_records:
                self.records.extend(new_records)
                self.last_row_records = new_records
                n += len(new_records)
        return n

    # -- driver -----------------------------------------------------------------
    def run(self, pdf):
        started = False
        section = None   # "drought" | unusual/flood/extreme category
        for idx, page in enumerate(pdf.pages):
            self.page = idx + 1
            text = norm_ws(page.extract_text())
            if not started and "2.0 Strategies" in text:
                started = True
            if not started:
                continue
            stop = STOP_PATTERN.search(text)
            # page-level section headings, in reading order
            heads = [(m.start(), CAT_UNUSUAL) for m in re.finditer(r"2\.2\s*Unusual rains", text, re.I)]
            heads += [(m.start(), CAT_FLOOD) for m in re.finditer(r"2\.3\s*Floods", text, re.I)]
            heads += [(m.start(), CAT_EXTREME) for m in re.finditer(r"2\.4\s*Extreme events", text, re.I)]
            heads.sort()
            if re.search(r"2\.3\s*Floods\s*:?\s*Not applicable", text, re.I):
                self.not_applicable.append({"district": self.district, "condition_category": CAT_FLOOD,
                                            "condition_detail": "", "source_page": self.page})
            tables = page.find_tables()
            for t in tables:
                if stop and t.bbox[1] > (page.search("Contingent strategies for Livestock") or [{"top": 1e9}])[0]["top"]:
                    break
                table = t.extract()
                if not table or not table[0]:
                    continue
                # which section does this table sit under?
                top = t.bbox[1]
                hits = page.search(r"2\.[234]\s", regex=True)
                above = [h for h in hits if h["top"] < top]
                if above:
                    label = above[-1]["text"]
                    section = {"2.2": CAT_UNUSUAL, "2.3": CAT_FLOOD, "2.4": CAT_EXTREME}.get(label.strip()[:3], section)
                elif heads and section is None:
                    pass
                if find_role_header(table) is not None:
                    self.drought_table(table)
                elif find_stage_header(table) is not None or section in (CAT_UNUSUAL, CAT_FLOOD, CAT_EXTREME):
                    self.stage_table(table, section or CAT_UNUSUAL)
                else:
                    self.drought_table(table)
            if stop:
                break
        return self.records, self.not_applicable


ACTION_FIELDS = ["district", "condition_category", "condition_detail", "situation", "soil_type",
                 "crop", "crop_raw", "crop_group", "season_hint", "crop_stage",
                 "change_in_cropping", "agronomic_measures", "crop_management",
                 "soil_moisture_conservation", "rabi_crop_planning", "remarks",
                 "table_scheme", "row_ditto", "source_page", "continued_on_page"]


def main():
    static_rows, action_rows, na_rows = [], [], []
    for fname, district in CONTINGENCY_PDF_DISTRICT.items():
        path = PDF_DIR / fname
        if not path.exists():
            print(f"WARNING: missing {path}", file=sys.stderr)
            continue
        with pdfplumber.open(path) as pdf:
            static_rows.extend(extract_static_features(pdf, district))
            recs, na = Extractor(district).run(pdf)
        action_rows.extend(recs)
        na_rows.extend(na)
        print(f"{district:<11} {len(recs):4d} action rows, {len(na)} not-applicable statements")

    with open(OUT_DIR / "district_static_features.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["district", "soil_type", "soil_area_kha", "soil_pct",
                                          "rainfall_normal_mm", "net_irrigated_pct"])
        w.writeheader()
        w.writerows(static_rows)
    with open(OUT_DIR / "contingency_actions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ACTION_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(action_rows)
    with open(OUT_DIR / "not_applicable.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["district", "condition_category", "condition_detail", "source_page"])
        w.writeheader()
        w.writerows(na_rows)
    print(f"\ncontingency_actions.csv: {len(action_rows)} rows | not_applicable.csv: {len(na_rows)} rows")


if __name__ == "__main__":
    main()
