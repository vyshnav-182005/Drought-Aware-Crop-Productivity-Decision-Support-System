"""
Query layer over the knowledge graph (Stage 9). These functions are what the
Stage 10 GraphRAG / LLM layer calls: every fact returned carries its source
(contingency-plan page, DES, or the season labeller), and nothing here
predicts or invents anything.

Main entry point for the LLM layer:
    retrieve_context(G, district, crop, season, condition)  -> dict (JSON-ready)

`condition` can be a ConditionDetail id ("onset_delay_4w",
"mid_dry_spell_flowering", ...), a Condition category name ("Mid-season
drought (dry spell)") or a category id ("mid_season_drought").

CLI examples (run from the project root):
    python codes/knowledge_graph/kg_query.py --district Beed --crop Soybean --condition mid_dry_spell_flowering
    python codes/knowledge_graph/kg_query.py --district Solapur --crop Jowar --season Rabi --condition terminal_drought
    python codes/knowledge_graph/kg_query.py --district Latur --crop Soybean --condition onset_delay_4w --json
    python codes/knowledge_graph/kg_query.py --list-conditions
"""
import argparse
import json
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = ROOT / "outputs" / "knowledge_graph" / "knowledge_graph.json"
ACTION_FIELDS = ("change_in_cropping", "agronomic_measures", "crop_management",
                 "soil_moisture_conservation", "rabi_crop_planning", "remarks")


def load_graph(path=GRAPH_PATH) -> nx.MultiDiGraph:
    return nx.node_link_graph(json.loads(Path(path).read_text(encoding="utf-8")), edges="edges")


def _nodes(G, type_):
    return [(n, d) for n, d in G.nodes(data=True) if d.get("type") == type_]


def _out(G, node, relation):
    return [(v, d) for _, v, d in G.out_edges(node, data=True) if d.get("relation") == relation]


def _in(G, node, relation):
    return [(u, d) for u, _, d in G.in_edges(node, data=True) if d.get("relation") == relation]


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
def list_conditions(G):
    out = []
    for cnode, c in sorted(_nodes(G, "Condition"), key=lambda x: x[1]["name"]):
        details = [{"id": G.nodes[d]["code"], "name": G.nodes[d]["name"],
                    **{k: v for k, v in G.nodes[d].items()
                       if k not in ("type", "code", "name", "category", "raw_text_variants")}}
                   for d, _ in _in(G, cnode, "SUBTYPE_OF")]
        out.append({"category": c["name"], "id": c.get("code"), "description": c.get("description"),
                    "details": sorted(details, key=lambda x: x["id"])})
    return out


def resolve_condition(G, condition):
    """-> list of ConditionDetail node keys matching a detail id, category name or category id."""
    key = f"ConditionDetail:{condition}"
    if key in G:
        return [key]
    for cnode, c in _nodes(G, "Condition"):
        if condition in (c["name"], c.get("code")):
            return [d for d, _ in _in(G, cnode, "SUBTYPE_OF")]
    raise KeyError(f"Unknown condition '{condition}'. Use --list-conditions.")


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
def recommend(G, district, crop, condition, season=None, soil_type=None, crop_stage=None):
    """Official contingency-plan actions for this district + crop + condition.
    Season filter keeps rows whose season matches or is unspecified in the plan."""
    results = []
    for dkey in resolve_condition(G, condition):
        for akey, e in _out(G, dkey, "RECOMMENDS"):
            if e.get("district") != district or e.get("crop") != crop:
                continue
            if season and e.get("season") not in (season, None):
                continue
            if soil_type and e.get("soil_type") != soil_type:
                continue
            if crop_stage and e.get("crop_stage") not in (crop_stage, None):
                continue
            a = G.nodes[akey]
            alts = [{"crop": G.nodes[c]["name"], "role": s.get("role"), "season": s.get("season"),
                     "intercrop": s.get("intercrop")} for c, s in _out(G, akey, "SUGGESTS_CROP")]
            results.append({
                "condition": G.nodes[dkey]["name"], "condition_id": G.nodes[dkey]["code"],
                "category": G.nodes[dkey]["category"], "district": district, "crop": crop,
                "season": e.get("season"), "season_basis": e.get("season_basis"),
                "situation": e.get("situation"), "soil_type": e.get("soil_type"),
                "moisture_regime": e.get("moisture_regime"), "crop_stage": e.get("crop_stage"),
                **{f: a.get(f) for f in ACTION_FIELDS if a.get(f)},
                "action_types": (a.get("action_types") or "").split(";"),
                "suggested_crops": alts,
                "source": {"document": f"{district} District Agriculture Contingency Plan",
                           "page": e.get("source_page"), "continued_on_page": e.get("continued_on_page"),
                           "pdf_condition_text": e.get("condition_detail_raw"),
                           "ditto_resolved": e.get("ditto_resolved")},
            })
    return sorted(results, key=lambda r: (r["source"]["page"] or 0, r["soil_type"] or ""))


def alternatives(G, district, crop, condition, season=None):
    """Crops the plan suggests instead of / after `crop` under this condition."""
    seen = {}
    for r in recommend(G, district, crop, condition, season):
        for s in r["suggested_crops"]:
            k = (s["crop"], s["role"])
            seen.setdefault(k, {**s, "pages": set()})["pages"].add(r["source"]["page"])
    return [{**v, "pages": sorted(v["pages"])} for v in seen.values()]


def not_applicable(G, district):
    return [{"condition": G.nodes[t].get("name"), "page": e.get("source_page"), "statement": e.get("statement")}
            for t, e in _out(G, f"District:{district}", "NOT_APPLICABLE")]


# ---------------------------------------------------------------------------
# District profile and history
# ---------------------------------------------------------------------------
def district_profile(G, district):
    d = G.nodes[f"District:{district}"]
    soils = [{"soil_type": G.nodes[s]["name"], "area_kha": e.get("area_kha"), "pct": e.get("pct")}
             for s, e in _out(G, f"District:{district}", "HAS_SOIL")]
    crops = [{"crop": G.nodes[c]["name"], "seasons": e.get("seasons"), "source": e.get("data_source")}
             for c, e in _out(G, f"District:{district}", "GROWS")]
    return {"district": district, "rainfall_normal_mm": d.get("rainfall_normal_mm"),
            "net_irrigated_pct": d.get("net_irrigated_pct"), "soils": soils,
            "crops": sorted(crops, key=lambda x: x["crop"])}


def season_history(G, district, season=None, crop=None):
    """Every recorded season: weather summary, conditions experienced, observed yield."""
    rows = []
    for sr, _ in _out(G, f"District:{district}", "HAS_SEASON_RECORD"):
        s = G.nodes[sr]
        if season and s["season"] != season:
            continue
        conds = [{"id": G.nodes[c]["code"], "name": G.nodes[c]["name"], "basis": e.get("basis"),
                  "evidence": e.get("evidence")} for c, e in _out(G, sr, "EXPERIENCED")]
        yields = {G.nodes[c]["name"]: {"area_ha": e.get("area_ha"), "production_t": e.get("production_t"),
                                       "yield_t_ha": e.get("yield_t_ha")}
                  for c, e in _out(G, sr, "HAD_YIELD") if not crop or G.nodes[c]["name"] == crop}
        rows.append({"year": s["year"], "season": s["season"],
                     **{k: s.get(k) for k in ("rainfall_mm", "pct_departure_from_mean", "rainfall_status",
                                               "onset_date", "longest_dry_spell_days") if s.get(k) not in (None, "")},
                     "conditions": conds, "observed_yield": yields})
    return sorted(rows, key=lambda r: (r["year"], r["season"]))


def similar_seasons(G, district, condition, season=None, crop=None):
    """Past seasons in this district that experienced the condition, with the yield that followed."""
    ids = {G.nodes[k]["code"] for k in resolve_condition(G, condition)}
    return [r for r in season_history(G, district, season, crop)
            if any(c["id"] in ids for c in r["conditions"])]


def yield_baseline(G, district, crop, season):
    ys = [r["observed_yield"][crop]["yield_t_ha"] for r in season_history(G, district, season, crop)
          if crop in r["observed_yield"] and r["observed_yield"][crop]["yield_t_ha"]]
    return {"n_years": len(ys), "mean_yield_t_ha": round(sum(ys) / len(ys), 3) if ys else None}


# ---------------------------------------------------------------------------
# One call for the LLM layer
# ---------------------------------------------------------------------------
def retrieve_context(G, district, crop, season, condition, soil_type=None):
    recs = recommend(G, district, crop, condition, season, soil_type)
    past = similar_seasons(G, district, condition, season, crop)
    base = yield_baseline(G, district, crop, season) if season else None
    return {
        "query": {"district": district, "crop": crop, "season": season, "condition": condition,
                  "soil_type": soil_type},
        "condition_definitions": [{"id": G.nodes[k]["code"], "name": G.nodes[k]["name"],
                                   "category": G.nodes[k]["category"]} for k in resolve_condition(G, condition)],
        "district_profile": district_profile(G, district),
        "recommendations": recs,
        "alternative_crops": alternatives(G, district, crop, condition, season),
        "past_seasons_with_this_condition": past,
        "yield_baseline_all_years": base,
        "plan_marks_not_applicable": not_applicable(G, district),
        "guardrail": "Use only these facts and the model outputs passed alongside them; cite source pages; "
                     "do not invent recommendations or numbers.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--district")
    p.add_argument("--crop")
    p.add_argument("--condition", help="detail id, category name or category id")
    p.add_argument("--season", choices=["Kharif", "Rabi"])
    p.add_argument("--soil-type")
    p.add_argument("--json", action="store_true", help="print the full retrieve_context() bundle")
    p.add_argument("--list-conditions", action="store_true")
    args = p.parse_args()
    G = load_graph()

    if args.list_conditions:
        for c in list_conditions(G):
            print(f"\n{c['category']}  [{c['id']}]")
            for d in c["details"]:
                print(f"  {d['id']:<34} {d['name']}")
        return
    if not (args.district and args.crop and args.condition):
        p.error("--district, --crop and --condition are required (or use --list-conditions)")

    if args.json:
        print(json.dumps(retrieve_context(G, args.district, args.crop, args.season, args.condition,
                                          args.soil_type), indent=1, default=str))
        return

    results = recommend(G, args.district, args.crop, args.condition, args.season, args.soil_type)
    if not results:
        print(f"No contingency-plan action for {args.district} / {args.crop} / {args.condition}")
        print("Crops in this district:", [c["crop"] for c in district_profile(G, args.district)["crops"]])
        for na in not_applicable(G, args.district):
            print(f"  plan says not applicable: {na['statement']} (p.{na['page']})")
        return
    labels = {"change_in_cropping": "Change in crop", "agronomic_measures": "Agronomic measures",
              "crop_management": "Crop management", "soil_moisture_conservation": "Soil/moisture conservation",
              "rabi_crop_planning": "Rabi crop planning", "remarks": "Remarks"}
    for r in results:
        head = f"\n[{r['condition']}] {r['crop']}"
        head += f" ({r['season'] or 'season not stated'}, {r['situation']})"
        if r["soil_type"]:
            head += f" on {r['soil_type']}" + (f", {r['moisture_regime']} rainfall" if r["moisture_regime"] else "")
        if r["crop_stage"]:
            head += f" at {r['crop_stage']} stage"
        print(head)
        for f, lab in labels.items():
            if r.get(f):
                print(f"  {lab}: {r[f]}")
        if r["suggested_crops"]:
            print("  Suggested crops: " + ", ".join(f"{s['crop']} ({s['role']})" for s in r["suggested_crops"]))
        print(f"  Source: {r['source']['document']}, p.{r['source']['page']}")
    past = similar_seasons(G, args.district, args.condition, args.season or "Kharif", args.crop)
    if past:
        print(f"\nPast {args.district} seasons with this condition:")
        for s in past:
            y = s["observed_yield"].get(args.crop, {}).get("yield_t_ha")
            print(f"  {s['season']} {s['year']}: rainfall {s.get('pct_departure_from_mean')}% vs mean"
                  + (f", {args.crop} yield {y} t/ha" if y else ""))


if __name__ == "__main__":
    main()
