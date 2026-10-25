"""
Build the knowledge graph (Stage 9) from the extracted CSVs.

Inputs (outputs/knowledge_graph/, written by extract_pdfs.py and label_seasons.py):
  district_static_features.csv, contingency_actions.csv, not_applicable.csv,
  season_conditions.csv, season_weather_summary.csv
plus the DES crop table in data/crop_production/.

Outputs: knowledge_graph.graphml, knowledge_graph.json, graph_stats.json

Schema
------
Node attribute `code` is the stable short id (ConditionDetail "onset_delay_4w",
Action "action_<hash>"); the node key itself is "<Type>:<name or code>".
Nodes: District, SoilType, Season, Crop, Condition (7 categories),
       ConditionDetail (sub-types with machine-readable thresholds), Action,
       Year, SeasonRecord (one district-season-year).
Edges:
  District  -HAS_SOIL->        SoilType      (area, % of district; Section 1.4)
  SoilType  -INCLUDES->        SoilType      (Section 2.0 soil group -> 1.4 category, name-derived)
  District  -GROWS->           Crop          (data_source DES and/or contingency_plan; seasons)
  Crop      -IN_SEASON->       Season
  ConditionDetail -SUBTYPE_OF-> Condition
  ConditionDetail -RECOMMENDS-> Action       (one edge per contingency-table row: district,
                                              crop, season, soil, crop_stage, source_page ...)
  Crop      -APPLIES_TO_CROP-> Action
  SoilType  -SOIL_CONTEXT->    Action
  Action    -SUGGESTS_CROP->   Crop          (alternative / intercrop / Rabi follow-up crop
                                              named in the action text)
  District  -NOT_APPLICABLE->  Condition / ConditionDetail   (plan says "Not applicable")
  District  -HAS_SEASON_RECORD-> SeasonRecord
  SeasonRecord -IN_YEAR->      Year
  SeasonRecord -OF_SEASON->    Season
  SeasonRecord -EXPERIENCED->  ConditionDetail   (from season_conditions.csv)
  SeasonRecord -HAD_YIELD->    Crop          (observed DES area / production / yield)

Run from the project root:  python3 codes/knowledge_graph/build_graph.py
"""
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from district_name_map import CANONICAL_DISTRICTS  # noqa: E402
from extract_pdfs import (CAT_EARLY, CAT_EXTREME, CAT_FLOOD, CAT_IRRIG, CAT_LATE,  # noqa: E402
                          CAT_MID, CAT_UNUSUAL, crops_with_season)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = OUT_DIR = ROOT / "outputs" / "knowledge_graph"
STAGE4_LABELS = ROOT / "outputs" / "condition_classification" / "season_conditions.csv"

# ---------------------------------------------------------------------------
# Crops and seasons
# ---------------------------------------------------------------------------
CROP_SEASONS = {
    "Jowar": ["Kharif", "Rabi"], "Sunflower": ["Kharif", "Rabi"],
    "Soybean": ["Kharif"], "Tur": ["Kharif"], "Bajra": ["Kharif"], "Cotton": ["Kharif"],
    "Maize": ["Kharif"], "Black gram": ["Kharif"], "Green gram": ["Kharif"], "Moth bean": ["Kharif"],
    "Horse gram": ["Kharif"], "Cowpea": ["Kharif"], "Sesamum": ["Kharif"], "Niger": ["Kharif"],
    "Castor": ["Kharif"], "Gram": ["Rabi"], "Wheat": ["Rabi"], "Safflower": ["Rabi"],
    "Sugarcane": [],  # perennial, excluded from the seasonal model
}
CORE_7 = {"Jowar", "Soybean", "Tur", "Sunflower", "Gram"}
EXTENDED = {"Bajra", "Cotton", "Wheat", "Safflower"}
DROUGHT_CATEGORIES = {CAT_EARLY, CAT_MID, CAT_LATE}

CONDITIONS = {  # category -> (short id, description)
    CAT_EARLY: ("early_season_drought", "Rainfed Kharif: delayed monsoon onset, or a dry spell soon after a normal onset (plan 2.1.1)"),
    CAT_MID: ("mid_season_drought", "Rainfed Kharif: long dry spell during crop growth (plan 2.1.1)"),
    CAT_LATE: ("terminal_drought", "Rainfed Kharif: early withdrawal of monsoon (plan 2.1.1)"),
    CAT_IRRIG: ("irrigation_shortage", "Irrigated situation: canal / tank / groundwater shortage (plan 2.1.2)"),
    CAT_UNUSUAL: ("unusual_rains", "Untimely or unseasonal rains, rainfed and irrigated (plan 2.2)"),
    CAT_FLOOD: ("floods", "Waterlogging, submergence, sea-water intrusion (plan 2.3)"),
    CAT_EXTREME: ("extreme_events", "Heat wave, cold wave, frost, hailstorm, cyclone (plan 2.4)"),
}


def crop_category(name):
    return "core" if name in CORE_7 else "extended" if name in EXTENDED else "fallback"


# ---------------------------------------------------------------------------
# Condition details: free PDF text -> one canonical sub-type with thresholds
# ---------------------------------------------------------------------------
ONSET_WINDOWS = {2: "4th week of June", 4: "2nd week of July", 6: "4th week of July", 8: "2nd week of August"}


def parse_detail(category, text):
    """Return (id, name, attrs) for a condition-detail string."""
    t = (text or "").lower()
    if category == CAT_EARLY:
        m = re.search(r"delay\s*by\s*(\d)\s*weeks?", t)
        if m:
            w = int(m.group(1))
            return (f"onset_delay_{w}w", f"Monsoon onset delayed by {w} weeks ({ONSET_WINDOWS[w]})",
                    {"onset_delay_weeks": w, "onset_window": ONSET_WINDOWS[w]})
        if "normal onset" in t:
            return ("normal_onset_dry_spell", "Normal onset followed by a 15-20 day dry spell after sowing",
                    {"dry_spell_days_min": 15, "dry_spell_days_max": 20, "days_after_onset_max": 30})
    if category == CAT_MID:
        base = {"min_rainless_days": 14, "rainless_threshold_mm": 2.5}
        if "flowering" in t or "fruiting" in t:
            return ("mid_dry_spell_flowering", "Mid-season dry spell (2+ rainless weeks) at flowering/fruiting stage",
                    {**base, "crop_stage": "flowering/fruiting"})
        return ("mid_dry_spell_vegetative", "Mid-season dry spell (2+ rainless weeks) at vegetative stage",
                {**base, "crop_stage": "vegetative"})
    if category == CAT_LATE:
        return ("terminal_drought", "Terminal drought (early withdrawal of monsoon)", {})
    if category == CAT_IRRIG:
        if "canal" in t:
            if "non" in t:
                return ("canal_non_release", "No release of canal water (delayed onset in catchment)", {"water_source": "canal"})
            if "delayed" in t and "limited" in t:
                return ("canal_delayed_or_limited_release", "Delayed or limited release of canal water", {"water_source": "canal"})
            if "limited" in t:
                return ("canal_limited_release", "Limited release of canal water due to low rainfall", {"water_source": "canal"})
            return ("canal_delayed_release", "Delayed release of canal water due to low rainfall", {"water_source": "canal"})
        if "tank" in t:
            return ("tank_low_inflow", "Lack of inflows into tanks (insufficient / delayed monsoon)", {"water_source": "tank"})
        if "groundwater" in t or "ground water" in t:
            return ("groundwater_low_recharge", "Insufficient groundwater recharge due to low rainfall", {"water_source": "groundwater"})
    if category == CAT_UNUSUAL:
        if "pest" in t or "disease" in t:
            return ("pest_disease_outbreak", "Outbreak of pests and diseases due to unseasonal rains", {})
        if "wind" in t:
            return ("heavy_rain_high_winds", "Heavy rainfall with high-speed winds in a short span", {})
        return ("continuous_high_rainfall", "Continuous high rainfall in a short span (waterlogging)", {})
    if category == CAT_FLOOD:
        if "submergence" in t:
            return ("continuous_submergence", "Continuous submergence for more than 2 days", {"min_days": 2})
        if "sea water" in t:
            return ("sea_water_intrusion", "Sea water intrusion", {})
        if t:
            return ("transient_waterlogging", "Transient waterlogging / partial inundation", {})
    if category == CAT_EXTREME:
        for key, name in [("heat", "Heat wave"), ("cold", "Cold wave"), ("frost", "Frost"),
                          ("hail", "Hailstorm"), ("cyclone", "Cyclone")]:
            if key in t:
                return (key.replace("hail", "hailstorm").replace("heat", "heat_wave").replace("cold", "cold_wave"), name, {})
    return (None, None, {})


# ---------------------------------------------------------------------------
# Soil normalisation (unchanged rules, see README "Credibility fixes")
# ---------------------------------------------------------------------------
SOIL_QUALIFIER_RE = re.compile(r"with\s+(assured( and high)?|low)\s+rainfall", re.I)
PAREN_RE = re.compile(r"\([^)]*\)?")
IRRIG_SUFFIX_RE = re.compile(r"--?\s*open well irrigated|(with\s+)?canal irrigation.*|,?\s*open well.*", re.I)
GARBAGE_SOIL = {"", "with", "assured rainfall", "low rainfall", "kharif", "scarcity", "not applicable for this district"}
SOIL_ALIASES = {
    "Medium Deep To Deep And Black Soils": "Medium Deep To Deep Black Soils",
    "Medium Deep To Deep Black Cotton Soils": "Medium Deep To Deep Black Soils",
    "Medium Deep To Deep Black Cotton Soils With": "Medium Deep To Deep Black Soils",
    "Scarcity Low Rainfall Medium Black Soils": "Medium Black Soils",
    "Light Soils Canal Irrigation": "Light Soils",
    "Shallow Grey Soils-": "Shallow Grey Soils",
}
SOIL_CROSSWALK = {
    ("Beed", "Medium Deep To Deep Black Soils"): ["Deep Black Soils", "Medium Deep Black Soils"],
    ("Osmanabad", "Medium Deep To Deep Black Soils"): ["Deep Black Soils", "Medium Deep Black Soils"],
    ("Solapur", "Medium Deep To Deep Black Soils"): ["Deep Black Soils"],
    ("Solapur", "Shallow To Medium Deep Black Soils"): ["Shallow Black Soils"],
}


def normalize_soil_type(raw):
    if not raw:
        return None, None
    t = raw.strip()
    regime = ("assured_high" if re.search(r"assured.*high|high.*rainfall", t, re.I) else
              "assured" if re.search(r"assured", t, re.I) else
              "low" if re.search(r"low rainfall", t, re.I) else None)
    t2 = PAREN_RE.sub("", t)
    t2 = SOIL_QUALIFIER_RE.sub("", t2)
    t2 = IRRIG_SUFFIX_RE.sub("", t2)
    t2 = re.sub(r"\s+", " ", t2).strip(" ,-")
    t2 = re.sub(r"\bwith\s*$", "", t2, flags=re.I).strip(" ,")
    t2 = re.sub(r"\bsoil\b(?!s)", "soils", t2, flags=re.I)
    if t2.lower() in GARBAGE_SOIL or not re.search(r"soil", t2, re.I):
        return None, regime
    name = t2.title()
    return SOIL_ALIASES.get(name, name), regime


def fill_soil_types(rows):
    last = {}
    for r in rows:
        r["soil_type_raw"] = r["soil_type"]
        if r["table_scheme"] == "stage":       # 2.2-2.4 tables have no soil column
            r["soil_type"], r["moisture_regime"] = None, None
            continue
        soil, regime = normalize_soil_type(r["soil_type"])
        block = (r["district"], r["condition_category"])
        if soil is None:
            soil = last.get(block)
        else:
            last[block] = soil
        r["soil_type"], r["moisture_regime"] = soil, regime
    return rows


# ---------------------------------------------------------------------------
# "-do-" / "As above" (ditto) resolution
# ---------------------------------------------------------------------------
DITTO_FULL = re.compile(r"^\s*(?:[-–]*\s*do\s*[-–]*\.?|as above\.?)\s*$", re.I)
DITTO_PREFIX = re.compile(r"^\s*(?:[-–]+\s*do\s*[-–]+|as above)\s*[+,]?\s*(.+)$", re.I)
TEXT_COLS = ("change_in_cropping", "agronomic_measures", "crop_management",
             "soil_moisture_conservation", "rabi_crop_planning", "remarks")


def resolve_dittos(rows):
    """Replace "-do-"/"As above" with the last real value of the same column in
    the same table block (district, condition; plus sub-condition and stage
    column for the 2.2-2.4 stage tables), in PDF order. A
    ditto followed by more text ("As above + spraying ...") becomes the
    previous value plus that text. A ditto with nothing above it is flagged
    "unresolved:<col>" and left as is."""
    by_block = {}
    for r in rows:
        # stage tables (2.2-2.4): "-do-" means the crop above, same stage column
        block = (r["district"], r["condition_category"], r["condition_detail"] if r["crop_stage"] else "",
                 r["crop_stage"])
        last = by_block.setdefault(block, {})
        flags = []
        for c in TEXT_COLS:
            v = (r.get(c) or "").strip()
            if DITTO_FULL.match(v):
                if last.get(c):
                    r[c] = last[c]
                    flags.append(c)
                else:
                    flags.append(f"unresolved:{c}")
                continue
            m = DITTO_PREFIX.match(v)
            if m and last.get(c):
                r[c] = f"{last[c]} + {m.group(1)}"
                flags.append(f"partial:{c}")
            if r[c].strip():
                last[c] = r[c]
        if r.get("row_ditto"):
            flags.append("left_cell:crop_management")
        r["ditto_resolved"] = ",".join(flags)
    return rows


# ---------------------------------------------------------------------------
# Season of each recommendation row
# ---------------------------------------------------------------------------
def assign_season(r):
    if r["season_hint"] in ("Kharif", "Rabi"):
        return r["season_hint"], "pdf_text"
    seasons = CROP_SEASONS.get(r["crop"])
    if seasons is not None and len(seasons) == 1:
        return seasons[0], "crop_calendar"
    if r["condition_category"] in DROUGHT_CATEGORIES and r["crop_group"] == "field" and seasons:
        return "Kharif", "kharif_drought_table"   # 2.1.1 tables are about the Kharif monsoon
    return "", "unspecified"


# ---------------------------------------------------------------------------
# Action typing
# ---------------------------------------------------------------------------
ACTION_TYPE_RULES = [
    ("irrigation", r"irrigat|sprinkler|drip"),
    ("moisture_conservation", r"mulch|conservation furrow|ridge|furrow|bund|in situ|harrow|hoeing|interculture|tied ridg|bbf"),
    ("nutrient_management", r"urea|dap|fertili|\bkcl\b|potash|nutrient|top dress"),
    ("plant_protection", r"spray|pest|insect|fung|disease|drench|carbendaz|dithane|imidacl|malathion"),
    ("drainage", r"drain"),
    ("thinning_or_resowing", r"thinning|gap filling|resow|re-sow|replant"),
    ("harvest", r"harvest"),
    ("fodder", r"fodder"),
    ("input_supply_linkage", r"seed source|linkage|supply of seed|nsc|mssc|mau|mpkv|kvk|rkvy"),
]


def action_types(row, suggests):
    text = " ".join(row.get(c, "") for c in TEXT_COLS).lower()
    types = []
    change = (row.get("change_in_cropping") or "").strip().lower()
    if suggests:
        types.append("crop_substitution")
    elif change and change not in ("no change", "-", "--") and re.search(r"variet|cultivar|\(", change):
        types.append("variety_change")
    if row.get("rabi_crop_planning", "").strip():
        types.append("rabi_planning")
    for name, pat in ACTION_TYPE_RULES:
        if re.search(pat, text):
            types.append(name)
    return types or ["agronomic"]


def suggested_crops(row):
    """Crops named in the change-in-cropping / Rabi-planning text that differ
    from the row's normal crop: (crop, role, season, intercrop)."""
    out = []
    for col, role in (("change_in_cropping", "alternative"), ("crop_management", "alternative"),
                      ("rabi_crop_planning", "rabi_follow_up")):
        text = row.get(col) or ""
        if col == "crop_management" and not re.search(r"\bsow|\bcrop\b|instead|replace|follow", text, re.I):
            continue
        for crop, season in crops_with_season(text):
            if crop == row["crop"] and role == "alternative":
                continue
            s = season or ("Rabi" if role == "rabi_follow_up" else
                           (CROP_SEASONS.get(crop) or [""])[0] if len(CROP_SEASONS.get(crop) or []) == 1 else "")
            inter = bool(re.search(r"\+", text)) and role == "alternative"
            if (crop, role) not in [(c, r) for c, r, _, _ in out]:
                out.append((crop, role, s, inter))
    return out


def action_id(row):
    key = "|".join(row.get(c, "") for c in TEXT_COLS)
    return "action_" + hashlib.md5(key.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# DES crop data
# ---------------------------------------------------------------------------
DES_CROPS = {"Arhar/Tur": "Tur", "Bajra": "Bajra", "Jowar": "Jowar", "Soyabean": "Soybean", "Sunflower": "Sunflower"}
DES_DISTRICTS = {"ahilyanagar": "Ahmednagar", "beed": "Beed", "latur": "Latur",
                 "osmanabad": "Osmanabad", "dharashiv": "Osmanabad", "solapur": "Solapur"}


def load_des(path):
    """Rows of (district, crop, season, year, area_ha, production_t, yield_t_ha)."""
    import pandas as pd
    try:
        df = pd.read_html(path)[0]
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not read {path}: {e}", file=sys.stderr)
        return []
    out = []
    for _, row in df.iterrows():
        draw = re.sub(r"^\d+\.\s*", "", str(row[("District", "District", "District")])).strip().lower()
        district = DES_DISTRICTS.get(draw)
        m = re.match(r"(\d{4})", str(row[("Year", "Year", "Year")]))
        if not district or not m:
            continue
        year = int(m.group(1))
        for crop_raw, crop in DES_CROPS.items():
            for season in ("Kharif", "Rabi"):
                try:
                    area = row[(crop_raw, season, "Area (Hectare)")]
                except KeyError:
                    continue
                prod = row.get((crop_raw, season, "Production (Tonnes)"))
                yld = row.get((crop_raw, season, "Yield (Tonne/Hectare)"))
                f = lambda v: float(v) if v is not None and str(v) not in ("nan", "", "-") else None  # noqa: E731
                if f(area) is None and f(yld) is None:
                    continue
                out.append((district, crop, season, year, f(area), f(prod), f(yld)))
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def read_csv(name):
    path = DATA_DIR / name
    if not path.exists():
        print(f"WARNING: {path} missing", file=sys.stderr)
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build():
    G = nx.MultiDiGraph()

    # District + static features --------------------------------------------
    static = {}
    for row in read_csv("district_static_features.csv"):
        static.setdefault(row["district"], []).append(row)
    for d in CANONICAL_DISTRICTS:
        rows = static.get(d, [])
        rf = rows[0]["rainfall_normal_mm"] if rows else None
        irr = next((r["net_irrigated_pct"] for r in rows if r["net_irrigated_pct"]), None)
        G.add_node(f"District:{d}", type="District", name=d,
                   rainfall_normal_mm=float(rf) if rf else None,
                   net_irrigated_pct=float(irr) if irr else None)
    for d, rows in static.items():
        for r in rows:
            if not r["soil_type"]:
                continue
            soil = re.sub(r"\bsoil\b(?!s)", "soils", r["soil_type"], flags=re.I).title()
            G.add_node(f"SoilType:{soil}", type="SoilType", name=soil)
            G.add_edge(f"District:{d}", f"SoilType:{soil}", key="HAS_SOIL", relation="HAS_SOIL",
                       area_kha=float(r["soil_area_kha"]) if r["soil_area_kha"] else None,
                       pct=float(r["soil_pct"]) if r["soil_pct"] else None)

    for s in ("Kharif", "Rabi"):
        G.add_node(f"Season:{s}", type="Season", name=s)

    # Conditions ------------------------------------------------------------------
    for cat, (cid, desc) in CONDITIONS.items():
        G.add_node(f"Condition:{cat}", type="Condition", name=cat, category=cat, code=cid, description=desc)

    def detail_node(cat, text, record_variant=True):
        did, name, attrs = parse_detail(cat, text)
        if did is None:
            did = "other_" + re.sub(r"[^a-z0-9]+", "_", (text or "unspecified").lower()).strip("_")[:40]
            name, attrs = text or f"{cat} (unspecified)", {"parsed": False}
        key = f"ConditionDetail:{did}"
        if key not in G:
            G.add_node(key, type="ConditionDetail", code=did, name=name, category=cat, **attrs)
            G.add_edge(key, f"Condition:{cat}", key="SUBTYPE_OF", relation="SUBTYPE_OF")
        if text and record_variant:
            variants = set(filter(None, (G.nodes[key].get("raw_text_variants") or "").split(" || ")))
            variants.add(text)
            G.nodes[key]["raw_text_variants"] = " || ".join(sorted(variants))
        return key

    # Contingency rows ----------------------------------------------------------------
    rows = resolve_dittos(fill_soil_types(read_csv("contingency_actions.csv")))
    crops = set(CROP_SEASONS) | {r["crop"] for r in rows if r["crop"]}
    for c in sorted(crops):
        G.add_node(f"Crop:{c}", type="Crop", name=c, category=crop_category(c))
        for s in CROP_SEASONS.get(c, []):
            G.add_edge(f"Crop:{c}", f"Season:{s}", key="IN_SEASON", relation="IN_SEASON")

    grows = {}
    for i, r in enumerate(rows):
        season, season_basis = assign_season(r)
        suggests = suggested_crops(r)
        aid = action_id(r)
        akey = f"Action:{aid}"
        if akey not in G:
            types = action_types(r, suggests)
            G.add_node(akey, type="Action", code=aid,
                       **{c: r.get(c, "") for c in TEXT_COLS},
                       action_category=types[0], action_types=";".join(types))
        dkey = detail_node(r["condition_category"], r["condition_detail"])
        soil_key = f"SoilType:{r['soil_type']}" if r["soil_type"] else None
        if soil_key and soil_key not in G:
            G.add_node(soil_key, type="SoilType", name=r["soil_type"])

        G.add_edge(dkey, akey, key=f"RECOMMENDS:{i}", relation="RECOMMENDS",
                   district=r["district"], crop=r["crop"], crop_group=r["crop_group"],
                   season=season or None, season_basis=season_basis,
                   situation=r["situation"], soil_type=r["soil_type"] or None,
                   soil_type_raw=r["soil_type_raw"] or None, moisture_regime=r.get("moisture_regime"),
                   crop_stage=r["crop_stage"] or None, condition_category=r["condition_category"],
                   condition_detail_raw=r["condition_detail"] or None, table_scheme=r["table_scheme"],
                   ditto_resolved=r["ditto_resolved"] or None, source_page=int(r["source_page"]),
                   continued_on_page=int(r["continued_on_page"]) if r["continued_on_page"] else None)
        G.add_edge(f"Crop:{r['crop']}", akey, key="APPLIES_TO_CROP", relation="APPLIES_TO_CROP")
        if soil_key:
            G.add_edge(soil_key, akey, key="SOIL_CONTEXT", relation="SOIL_CONTEXT")
        for crop, role, s, inter in suggests:
            if f"Crop:{crop}" not in G:
                G.add_node(f"Crop:{crop}", type="Crop", name=crop, category=crop_category(crop))
            G.add_edge(akey, f"Crop:{crop}", key=f"SUGGESTS_CROP:{role}", relation="SUGGESTS_CROP",
                       role=role, season=s or None, intercrop=inter)
        g = grows.setdefault((r["district"], r["crop"]), {"sources": set(), "seasons": set()})
        g["sources"].add("contingency_plan")
        if season:
            g["seasons"].add(season)

    # Not applicable ------------------------------------------------------------
    for r in read_csv("not_applicable.csv"):
        target = (detail_node(r["condition_category"], r["condition_detail"]) if r["condition_detail"]
                  else f"Condition:{r['condition_category']}")
        G.add_edge(f"District:{r['district']}", target, key=f"NOT_APPLICABLE:{r['source_page']}",
                   relation="NOT_APPLICABLE", source_page=int(r["source_page"]),
                   statement=r["condition_detail"] or f"{r['condition_category']}: not applicable")

    # Every condition the season labeller can emit gets a node, even if no
    # contingency row used that exact wording.
    for cat, text in [(CAT_EARLY, f"Delay by {w} weeks") for w in (2, 4, 6, 8)] + [
            (CAT_EARLY, "Normal onset"), (CAT_MID, "At vegetative stage"), (CAT_MID, "At flowering stage"),
            (CAT_LATE, "Terminal drought"), (CAT_UNUSUAL, "Continuous high rainfall")]:
        detail_node(cat, text, record_variant=False)
    detail_ids = {G.nodes[n]["code"]: n for n in G if G.nodes[n].get("type") == "ConditionDetail"}

    # DES yields + GROWS -----------------------------------------------------------------
    des = load_des(ROOT / "data" / "crop_production" / "horizontal_crop_vertical_year_report.xls")
    for district, crop, season, year, area, prod, yld in des:
        if area and area > 0:
            g = grows.setdefault((district, crop), {"sources": set(), "seasons": set()})
            g["sources"].add("DES")
            g["seasons"].add(season)
    for (district, crop), g in sorted(grows.items()):
        G.add_edge(f"District:{district}", f"Crop:{crop}", key="GROWS", relation="GROWS",
                   data_source=";".join(sorted(g["sources"])), seasons=";".join(sorted(g["seasons"])))

    # Years, season records, EXPERIENCED, HAD_YIELD ----------------------------------------
    summary = read_csv("season_weather_summary.csv")
    labels_path = STAGE4_LABELS if STAGE4_LABELS.exists() else DATA_DIR / "season_conditions.csv"
    labels = []
    if labels_path.exists():
        with open(labels_path, encoding="utf-8") as f:
            labels = list(csv.DictReader(f))

    def season_record(district, season, year):
        key = f"SeasonRecord:{district}|{season}|{year}"
        if key not in G:
            G.add_node(key, type="SeasonRecord", name=f"{district} {season} {year}",
                       district=district, season=season, year=int(year))
            ykey = f"Year:{year}"
            if ykey not in G:
                G.add_node(ykey, type="Year", name=str(year), year=int(year))
            G.add_edge(f"District:{district}", key, key="HAS_SEASON_RECORD", relation="HAS_SEASON_RECORD")
            G.add_edge(key, ykey, key="IN_YEAR", relation="IN_YEAR")
            G.add_edge(key, f"Season:{season}", key="OF_SEASON", relation="OF_SEASON")
        return key

    for s in summary:
        key = season_record(s["district"], s["season"], s["year"])
        for k, v in s.items():
            if k not in ("district", "season", "year") and v != "":
                try:
                    G.nodes[key][k] = float(v)
                except ValueError:
                    G.nodes[key][k] = v
    for lab in labels:
        cid = lab["condition_detail_id"]
        if cid not in detail_ids:
            print(f"WARNING: label {cid} has no ConditionDetail node", file=sys.stderr)
            continue
        key = season_record(lab["district"], lab["season"], lab["year"])
        G.add_edge(key, detail_ids[cid], key=f"EXPERIENCED:{cid}", relation="EXPERIENCED",
                   basis=lab.get("basis"), evidence=lab.get("evidence"))
    for district, crop, season, year, area, prod, yld in des:
        key = season_record(district, season, year)
        G.add_edge(key, f"Crop:{crop}", key="HAD_YIELD", relation="HAD_YIELD", data_source="DES",
                   area_ha=area, production_t=prod, yield_t_ha=yld)

    # Soil crosswalk ------------------------------------------------------------------
    for (district, group), members in SOIL_CROSSWALK.items():
        for m in members:
            if f"SoilType:{group}" in G and f"SoilType:{m}" in G:
                G.add_edge(f"SoilType:{group}", f"SoilType:{m}", key=f"INCLUDES:{district}",
                           relation="INCLUDES", district=district, basis="name_composition")
    return G


def add_experienced_edge(G, district, season, year, condition_detail_id, basis="stage4", evidence=""):
    """Hook for Stage 4: attach one classified condition to a season record."""
    node = next(n for n, d in G.nodes(data=True)
                if d.get("type") == "ConditionDetail" and d.get("code") == condition_detail_id)
    key = f"SeasonRecord:{district}|{season}|{year}"
    if key not in G:
        G.add_node(key, type="SeasonRecord", name=f"{district} {season} {year}",
                   district=district, season=season, year=int(year))
    G.add_edge(key, node, key=f"EXPERIENCED:{condition_detail_id}", relation="EXPERIENCED",
               basis=basis, evidence=evidence)


def main():
    G = build()
    n_by_type = Counter(d.get("type") for _, d in G.nodes(data=True))
    e_by_rel = Counter(d["relation"] for _, _, d in G.edges(data=True))
    isolated = sorted(n for n in G if G.degree(n) == 0)
    print("Nodes:", G.number_of_nodes(), dict(sorted(n_by_type.items())))
    print("Edges:", G.number_of_edges(), dict(sorted(e_by_rel.items())))
    print("Isolated nodes:", len(isolated), isolated)
    unparsed = [d["name"] for _, d in G.nodes(data=True) if d.get("type") == "ConditionDetail" and d.get("parsed") is False]
    print("Unparsed condition details:", unparsed)

    G_gml = G.copy()
    for _, d in list(G_gml.nodes(data=True)) + [(None, d) for *_, d in G_gml.edges(data=True)]:
        for k, v in list(d.items()):
            if v is None:
                d[k] = ""
    nx.write_graphml(G_gml, OUT_DIR / "knowledge_graph.graphml")
    with open(OUT_DIR / "knowledge_graph.json", "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G, edges="edges"), f, indent=1, default=str)
    with open(OUT_DIR / "graph_stats.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                   "nodes_by_type": dict(n_by_type), "edges_by_relation": dict(e_by_rel),
                   "isolated_nodes": isolated}, f, indent=1)
    print(f"\nWritten: {OUT_DIR / 'knowledge_graph.graphml'}, knowledge_graph.json, graph_stats.json")


if __name__ == "__main__":
    main()
