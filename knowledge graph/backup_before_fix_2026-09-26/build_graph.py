"""
Build the Stage 9 knowledge graph (District / SoilType / Crop / Season /
Condition / Action) from the extracted CSVs, per the schema in
Project_Implementation_Plan.md Section 7.2.

Graph store: NetworkX (per Section 7.4, "NetworkX for a lighter, local,
no-server alternative" -- genuinely sufficient at this scale: 5 districts,
~15 crops, 4 condition categories). A Neo4j loader using the same node/edge
data would just replace this file's `nx.MultiDiGraph()` calls with Cypher
CREATE statements against the same three CSVs.

NOT included yet (deliberately -- these depend on Stage 3-4, which is still
at plan stage per data_audit_findings.md, not owned by the knowledge-graph
task): Year nodes and (District,Season,Year)-EXPERIENCED->Condition edges.
Once Stage 4's rule-based SPI/tercile classifier exists and emits one of
this graph's Condition categories per (district, season, year), those edges
plug in directly -- see `add_experienced_edge()` at the bottom, which is
ready to call from that pipeline.

Run from the project root:  python3 knowledge_graph/scripts/build_graph.py
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from district_name_map import CANONICAL_DISTRICTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "knowledge_graph" / "data"
OUT_DIR = ROOT / "knowledge_graph" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical crop -> season(s), per Project_Implementation_Plan.md Section 2.2.
# Fallback/contingency-only crops (horticulture, minor pulses) are left
# unmapped here -- they get a Crop node but no IN_SEASON edge, since the
# plan explicitly excludes them from the seasonal yield model (Section 2.2:
# "may still appear as nodes ... but are not modeling targets").
CROP_SEASONS = {
    "Jowar": ["Kharif", "Rabi"],
    "Soybean": ["Kharif"],
    "Tur": ["Kharif"],
    "Sunflower": ["Kharif", "Rabi"],
    "Gram": ["Rabi"],
    "Bajra": ["Kharif"],
    "Cotton": ["Kharif"],
    "Wheat": ["Rabi"],
    "Safflower": ["Rabi"],
    "Sugarcane": [],  # perennial, excluded from seasonal model (Section 2.2/6)
    "Maize": ["Kharif"],
}
CORE_7 = {"Jowar", "Soybean", "Tur", "Sunflower", "Gram"}
EXTENDED = {"Bajra", "Cotton", "Wheat", "Safflower"}


def crop_category(name: str) -> str:
    if name in CORE_7:
        return "core"
    if name in EXTENDED:
        return "extended"
    return "fallback"


def classify_action(change: str, agronomic: str, remarks: str) -> str:
    text = f"{change} {agronomic} {remarks}".lower()
    if "irrigat" in text:
        return "irrigation"
    if "variety" in text or "seed" in text or change.strip().lower() not in ("", "no change", "-do-"):
        return "crop_substitution"
    return "agronomic"


SOIL_QUALIFIER_RE = re.compile(r"with\s+(assured( and high)?|low)\s+rainfall", re.I)
PAREN_RE = re.compile(r"\([^)]*\)")
IRRIG_SUFFIX_RE = re.compile(r"--?\s*open well irrigated|canal irrigation.*", re.I)
GARBAGE_SOIL = {"", "with", "assured rainfall", "low rainfall"}


def normalize_soil_type(raw: str):
    """Split a raw contingency-plan farming-situation string into a clean
    SoilType name plus a separate moisture-regime qualifier, e.g.
    "Medium deep to deep black soils with low rainfall (Asthi, patoda &
    Sirur kasar tehsils)" -> ("Medium Deep To Deep Black Soils", "low").
    Returns (None, regime) for fragments with no real soil descriptor
    (table-row continuations that leaked into the soil-type column)."""
    if not raw:
        return None, None
    t = raw.strip()
    regime = None
    if re.search(r"assured.*high|high.*rainfall", t, re.I):
        regime = "assured_high"
    elif re.search(r"assured", t, re.I):
        regime = "assured"
    elif re.search(r"low rainfall", t, re.I):
        regime = "low"

    t2 = PAREN_RE.sub("", t)
    t2 = SOIL_QUALIFIER_RE.sub("", t2)
    t2 = IRRIG_SUFFIX_RE.sub("", t2)
    t2 = re.sub(r"\s+", " ", t2).strip(" ,")
    t2 = re.sub(r"\bwith\s*$", "", t2, flags=re.I).strip(" ,")
    t2 = re.sub(r"\bsoil\b(?!s)", "soils", t2, flags=re.I)

    if t2.lower() in GARBAGE_SOIL:
        return None, regime
    return t2.title(), regime


def fill_soil_types(rows: list) -> list:
    """Normalize soil_type on every row, forward-filling within each
    district when a row's raw text has no real soil descriptor (a table
    continuation row whose soil-type cell was blank/merged upstream)."""
    last_by_district = {}
    for r in rows:
        soil, regime = normalize_soil_type(r["soil_type"])
        if soil is None:
            soil = last_by_district.get(r["district"])
        else:
            last_by_district[r["district"]] = soil
        r["soil_type"] = soil
        r["moisture_regime"] = regime
    return rows


def action_id(row: dict) -> str:
    key = f"{row['change_in_cropping']}|{row['agronomic_measures']}|{row['remarks']}"
    return "action_" + hashlib.md5(key.encode("utf-8")).hexdigest()[:10]


def load_des_crop_district_links(path: Path):
    """Which (district, crop, season) triples have real DES Area data.
    Returns a set of (canonical_district, canonical_crop, season)."""
    import pandas as pd

    des_crop_to_canonical = {
        "Arhar/Tur": "Tur", "Bajra": "Bajra", "Jowar": "Jowar",
        "Soyabean": "Soybean", "Sunflower": "Sunflower",
    }
    des_district_to_canonical = {
        "1. Ahilyanagar": "Ahmednagar", "2. Beed": "Beed", "3. Latur": "Latur",
        "4. Osmanabad": "Osmanabad", "5. Solapur": "Solapur",
        # tolerate un-prefixed variants too
        "Ahilyanagar": "Ahmednagar", "Beed": "Beed", "Latur": "Latur",
        "Osmanabad": "Osmanabad", "Solapur": "Solapur",
    }
    links = set()
    try:
        df = pd.read_html(path)[0]
    except Exception as e:
        print(f"WARNING: could not read {path}: {e}", file=sys.stderr)
        return links

    for _, row in df.iterrows():
        district_raw = str(row[("District", "District", "District")]).strip()
        district = des_district_to_canonical.get(district_raw)
        if not district:
            continue
        for col in df.columns:
            if col[0] not in des_crop_to_canonical or col[2] != "Area (Hectare)":
                continue
            val = row[col]
            try:
                if float(val) > 0:
                    links.add((district, des_crop_to_canonical[col[0]], col[1]))
            except (ValueError, TypeError):
                continue
    return links


def build():
    G = nx.MultiDiGraph()

    # --- District nodes + static features -----------------------------
    static_by_district = {}
    with open(DATA_DIR / "district_static_features.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            static_by_district.setdefault(row["district"], []).append(row)

    for d in CANONICAL_DISTRICTS:
        rows = static_by_district.get(d, [])
        rainfall = rows[0]["rainfall_normal_mm"] if rows else None
        irrigated_pct = next((r["net_irrigated_pct"] for r in rows if r["net_irrigated_pct"]), None)
        G.add_node(f"District:{d}", type="District", name=d,
                   rainfall_normal_mm=float(rainfall) if rainfall else None,
                   net_irrigated_pct=float(irrigated_pct) if irrigated_pct else None)

    # --- SoilType nodes + District-HAS_SOIL->SoilType -------------------
    for d, rows in static_by_district.items():
        for r in rows:
            if not r["soil_type"]:
                continue
            soil_name = re.sub(r"\bsoil\b(?!s)", "soils", r["soil_type"], flags=re.I).title()
            soil_key = f"SoilType:{soil_name}"
            if soil_key not in G:
                G.add_node(soil_key, type="SoilType", name=soil_name)
            G.add_edge(f"District:{d}", soil_key, key="HAS_SOIL",
                       area_kha=float(r["soil_area_kha"]) if r["soil_area_kha"] else None,
                       pct=float(r["soil_pct"]) if r["soil_pct"] else None)

    # --- Season nodes ----------------------------------------------------
    for s in ["Kharif", "Rabi"]:
        G.add_node(f"Season:{s}", type="Season", name=s)

    # --- Crop nodes + Crop-IN_SEASON->Season ------------------------------
    all_crops = set(CROP_SEASONS)
    with open(DATA_DIR / "contingency_actions.csv", encoding="utf-8") as f:
        action_rows = fill_soil_types(list(csv.DictReader(f)))
    all_crops |= {r["crop"] for r in action_rows if r["crop"]}

    for crop in sorted(all_crops):
        G.add_node(f"Crop:{crop}", type="Crop", name=crop, category=crop_category(crop))
        for season in CROP_SEASONS.get(crop, []):
            G.add_edge(f"Crop:{crop}", f"Season:{season}", key="IN_SEASON")

    # --- District-GROWS->Crop --------------------------------------------
    des_path = ROOT / "Data" / "crop_production" / "horizontal_crop_vertical_year_report.xls"
    des_links = load_des_crop_district_links(des_path)
    for district, crop, _season in des_links:
        G.add_edge(f"District:{district}", f"Crop:{crop}", key="GROWS", source="DES")

    for r in action_rows:
        if r["crop"]:
            G.add_edge(f"District:{r['district']}", f"Crop:{r['crop']}", key="GROWS",
                       source="contingency_plan")

    # --- Condition nodes ---------------------------------------------------
    categories = sorted({r["condition_category"] for r in action_rows})
    for cat in categories:
        G.add_node(f"Condition:{cat}", type="Condition", category=cat)

    # --- Action nodes + (Condition,Crop,SoilType)-RECOMMENDS->Action -------
    seen_actions = {}
    for r in action_rows:
        aid = action_id(r)
        act_key = f"Action:{aid}"
        if act_key not in G:
            G.add_node(act_key, type="Action", id=aid,
                       change_in_cropping=r["change_in_cropping"],
                       agronomic_measures=r["agronomic_measures"],
                       remarks=r["remarks"],
                       action_category=classify_action(
                           r["change_in_cropping"], r["agronomic_measures"], r["remarks"]))
            seen_actions[aid] = 0
        seen_actions[aid] += 1

        cond_key = f"Condition:{r['condition_category']}"
        crop_key = f"Crop:{r['crop']}"
        soil_key = f"SoilType:{r['soil_type']}" if r["soil_type"] else None
        if soil_key and soil_key not in G:
            G.add_node(soil_key, type="SoilType", name=r["soil_type"])

        G.add_edge(cond_key, act_key, key="RECOMMENDS",
                   crop=r["crop"], soil_type=r["soil_type"] or None,
                   moisture_regime=r.get("moisture_regime"),
                   district=r["district"], situation=r["situation"],
                   condition_detail=r["condition_detail"] or None,
                   source_page=int(r["source_page"]))
        # keep the crop/soil linkage explicit for direct traversal too
        G.add_edge(crop_key, act_key, key="APPLIES_TO_CROP")

    return G


def main():
    G = build()

    n_by_type = {}
    for _, d in G.nodes(data=True):
        n_by_type[d.get("type")] = n_by_type.get(d.get("type"), 0) + 1
    e_by_key = {}
    for _, _, k in G.edges(keys=True):
        e_by_key[k] = e_by_key.get(k, 0) + 1

    print("Nodes:", G.number_of_nodes(), n_by_type)
    print("Edges:", G.number_of_edges(), e_by_key)

    # GraphML has no null type -- replace None attrs with "" for that export only.
    G_gml = G.copy()
    for _, d in G_gml.nodes(data=True):
        for k, v in list(d.items()):
            if v is None:
                d[k] = ""
    for _, _, d in G_gml.edges(data=True):
        for k, v in list(d.items()):
            if v is None:
                d[k] = ""
    nx.write_graphml(G_gml, OUT_DIR / "knowledge_graph.graphml")

    data = nx.node_link_data(G, edges="edges")
    with open(OUT_DIR / "knowledge_graph.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nWritten to {OUT_DIR / 'knowledge_graph.graphml'}")
    print(f"Written to {OUT_DIR / 'knowledge_graph.json'}")


if __name__ == "__main__":
    main()
