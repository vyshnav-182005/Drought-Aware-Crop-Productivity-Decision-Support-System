"""
Stage 9 lookup: given a district, crop and condition category (the output of
Stage 4's classifier, once it exists), retrieve the matching official
contingency-plan recommendation(s) from the knowledge graph.

This is a *lookup*, not a prediction (Project_Implementation_Plan.md Section
4.1, Stage 9) -- every Action node traces back to a specific PDF page via the
RECOMMENDS edge's `source_page` property, so the LLM synthesis layer (Stage
10) can cite exactly where a recommendation came from.

Usage:
    python3 knowledge_graph/scripts/kg_query.py --district Beed --crop Soybean --condition "Mid-season drought (dry spell)"
"""
import argparse
import json
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = ROOT / "knowledge_graph" / "output" / "knowledge_graph.json"


def load_graph() -> nx.MultiDiGraph:
    data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, edges="edges")


def recommend(G: nx.MultiDiGraph, district: str, crop: str, condition_category: str,
              soil_type: str | None = None):
    """Traverse Condition -RECOMMENDS-> Action, filtered to this district's
    plan and this crop (and optionally soil type), per Section 7.3's
    "templated Cypher query" pattern -- this is that same query expressed
    as a NetworkX edge filter instead of Cypher."""
    cond_key = f"Condition:{condition_category}"
    crop_key = f"Crop:{crop}"
    if cond_key not in G:
        return []
    results = []
    for _, act_key, data in G.out_edges(cond_key, data=True):
        if data.get("relation") != "RECOMMENDS":
            continue
        if data.get("crop") != crop:
            continue
        if data.get("district") != district:
            continue
        if soil_type and data.get("soil_type") != soil_type:
            continue
        act = G.nodes[act_key]
        results.append({
            "district": data.get("district"),
            "situation": data.get("situation"),
            "soil_type": data.get("soil_type"),
            "moisture_regime": data.get("moisture_regime"),
            "condition_detail": data.get("condition_detail"),
            "crop": data.get("crop"),
            "change_in_cropping": act.get("change_in_cropping"),
            "agronomic_measures": act.get("agronomic_measures"),
            "remarks": act.get("remarks"),
            "action_category": act.get("action_category"),
            "source_page": data.get("source_page"),
        })
    return results


def district_soil_profile(G: nx.MultiDiGraph, district: str):
    """District -HAS_SOIL-> SoilType, for the static district features."""
    d_key = f"District:{district}"
    if d_key not in G:
        return []
    out = []
    for _, soil_key, data in G.out_edges(d_key, data=True):
        if data.get("relation") == "HAS_SOIL":
            out.append({"soil_type": G.nodes[soil_key]["name"],
                        **{k: v for k, v in data.items() if k != "relation"}})
    return out


def crops_grown(G: nx.MultiDiGraph, district: str):
    d_key = f"District:{district}"
    if d_key not in G:
        return []
    return sorted({
        G.nodes[crop_key]["name"]
        for _, crop_key, data in G.out_edges(d_key, data=True)
        if G.nodes[crop_key].get("type") == "Crop"
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--district", required=True)
    p.add_argument("--crop", required=True)
    p.add_argument("--condition", required=True,
                    help='One of: "Early season drought (delayed onset)", '
                         '"Mid-season drought (dry spell)", '
                         '"Late-season/terminal drought", "Unusual rains"')
    p.add_argument("--soil-type", default=None)
    args = p.parse_args()

    G = load_graph()
    results = recommend(G, args.district, args.crop, args.condition, args.soil_type)

    if not results:
        print(f"No contingency-plan action found for {args.district} / {args.crop} / {args.condition}"
              + (f" / {args.soil_type}" if args.soil_type else ""))
        print("Available crops in this district:", crops_grown(G, args.district))
        return

    for r in results:
        print(f"\n[{r['soil_type'] or 'soil type not captured'}"
              f"{' - ' + r['moisture_regime'] if r['moisture_regime'] else ''}]"
              f" {r['crop']} ({r['situation']})")
        if r["condition_detail"]:
            print(f"  Condition detail: {r['condition_detail']}")
        if r["change_in_cropping"]:
            print(f"  Change in crop/cropping: {r['change_in_cropping']}")
        if r["agronomic_measures"]:
            print(f"  Agronomic measures: {r['agronomic_measures']}")
        if r["remarks"]:
            print(f"  Remarks: {r['remarks']}")
        print(f"  Source: {r['district']} contingency plan, page {r['source_page']}")


if __name__ == "__main__":
    main()
