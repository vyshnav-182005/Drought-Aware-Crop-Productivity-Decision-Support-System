"""
Render knowledge_graph.json as a self-contained, interactive HTML page
(vis-network via CDN, data embedded inline -- works standalone via
file://, no local server needed).

Run from the project root:  python3 codes/knowledge_graph/build_html_viewer.py
Then open outputs/knowledge_graph/knowledge_graph_viewer.html in a browser.
"""
import json
import math
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
GRAPH_JSON = ROOT / "outputs" / "knowledge_graph" / "knowledge_graph.json"
OUT_HTML = ROOT / "outputs" / "knowledge_graph" / "knowledge_graph_viewer.html"

TYPE_COLORS = {
    "District": "#0C447C",     # blue 800
    "SoilType": "#633806",     # amber 800
    "Season": "#27500A",       # green 800
    "Crop": "#085041",         # teal 800
    "Condition": "#791F1F",    # red 800
    "Action": "#3C3489",       # purple 800
    "ConditionDetail": "#712B13",  # coral 800
    "Year": "#444441",         # gray 800
    "SeasonRecord": "#72243E",  # pink 800
}
TYPE_FILL = {
    "District": "#B5D4F4",
    "SoilType": "#FAC775",
    "Season": "#C0DD97",
    "Crop": "#9FE1CB",
    "Condition": "#F7C1C1",
    "Action": "#CECBF6",
    "ConditionDetail": "#F5C4B3",
    "Year": "#D3D1C7",
    "SeasonRecord": "#F4C0D1",
}
TYPE_SIZE = {
    "District": 26, "SoilType": 16, "Season": 22,
    "Crop": 18, "Condition": 22, "Action": 8,
    "ConditionDetail": 14, "Year": 10, "SeasonRecord": 6,
}

EDGE_LABELS = {
    "HAS_SOIL": "has soil",
    "GROWS": "grows",
    "IN_SEASON": "in season",
    "RECOMMENDS": "recommends",
    "APPLIES_TO_CROP": "applies to",
    "SOIL_CONTEXT": "soil context",
    "INCLUDES": "includes",
    "SUBTYPE_OF": "subtype of",
    "SUGGESTS_CROP": "suggests crop",
    "NOT_APPLICABLE": "not applicable",
    "HAS_SEASON_RECORD": "season record",
    "IN_YEAR": "in year",
    "OF_SEASON": "of season",
    "EXPERIENCED": "experienced",
    "HAD_YIELD": "had yield",
}


# Shown in the sidebar (click "?"), the Guide overlay, and the Details panel.
NODE_INFO = {
    "District": ("One of the 5 drought-prone districts.",
                 "Holds the plan's normal SW-monsoon rainfall and net irrigated %.", "Beed"),
    "SoilType": ("A soil category or farming situation named in the contingency plan.",
                 "Section 1.4 soil classes plus the soil groups used in the Section 2 action tables.",
                 "Medium Deep To Deep Black Soils"),
    "Season": ("Cropping season.", "Kharif (monsoon, Jun-Oct) or Rabi (post-monsoon, Oct-Feb).", "Kharif"),
    "Crop": ("A crop grown in a district or named in the plans.",
             "category = core (yield-model target), extended (optional) or fallback (named only in the plans).",
             "Soybean"),
    "Condition": ("Broad weather problem the plans prepare for (7).",
                  "Early-season, mid-season and terminal drought; irrigation water shortage; unusual rains; floods; "
                  "extreme events.", "Mid-season drought (dry spell)"),
    "ConditionDetail": ("The exact situation inside a Condition, with machine-readable thresholds.",
                        "This is what the Stage 4 classifier output and the recommendations attach to.",
                        "onset_delay_4w: monsoon onset delayed by 4 weeks (2nd week of July)"),
    "Action": ("One official recommendation from a contingency-plan table.",
               "Text split into the plan's own columns: change in crop, agronomic measures, crop management, "
               "soil/moisture conservation, Rabi crop planning, remarks.",
               "Give protective irrigation; foliar spray of 2% urea"),
    "Year": ("A calendar year, 2000-2025.", "Groups the season records.", "2015"),
    "SeasonRecord": ("One district's season in one year.",
                     "Carries that season's observed rainfall, % departure from normal, onset date and longest dry spell.",
                     "Beed Kharif 2015"),
}
EDGE_INFO = {
    "HAS_SOIL": ("District", "SoilType", "The district has this soil type (area and % of district, plan Section 1.4)."),
    "GROWS": ("District", "Crop", "The district grows this crop (seasons; source DES statistics and/or the plan)."),
    "IN_SEASON": ("Crop", "Season", "The crop is grown in this season."),
    "SUBTYPE_OF": ("ConditionDetail", "Condition", "This exact situation is a kind of the broader condition."),
    "RECOMMENDS": ("ConditionDetail", "Action",
                   "Under this situation the plan recommends this action. One edge per PDF table row, with district, "
                   "crop, season, soil, crop stage and source page."),
    "APPLIES_TO_CROP": ("Crop", "Action", "The action is prescribed for this crop (lets you start a search from a crop)."),
    "SOIL_CONTEXT": ("SoilType", "Action", "The action is prescribed for this soil / farming situation."),
    "SUGGESTS_CROP": ("Action", "Crop",
                      "The action names this crop as an alternative, an intercrop, or the Rabi crop to grow after "
                      "a failed Kharif crop."),
    "NOT_APPLICABLE": ("District", "Condition / ConditionDetail",
                       "The district's plan explicitly says this condition is not applicable (e.g. floods)."),
    "INCLUDES": ("SoilType", "SoilType",
                 "A soil group used in the action tables includes this official soil category (matched by name)."),
    "HAS_SEASON_RECORD": ("District", "SeasonRecord", "The district's record for one season of one year."),
    "IN_YEAR": ("SeasonRecord", "Year", "The year this season record belongs to."),
    "OF_SEASON": ("SeasonRecord", "Season", "Whether the record is a Kharif or a Rabi season."),
    "EXPERIENCED": ("SeasonRecord", "ConditionDetail",
                    "That season actually had this condition, labelled from the daily weather (evidence attached)."),
    "HAD_YIELD": ("SeasonRecord", "Crop", "Observed DES area, production and yield of this crop in that season."),
}


def node_label(n):
    t = n.get("type")
    if t == "Action":
        return (n.get("change_in_cropping") or n.get("crop_management") or n.get("agronomic_measures")
                or "action")[:28]
    return n.get("name", n.get("id", "?"))[:28]


def node_title(n):
    # HTML tooltip shown on hover
    lines = [f"<b>{t}</b>: {v}" for t, v in n.items() if v not in (None, "")]
    return "<br>".join(lines)


def edge_title(e):
    lines = [f"<b>{k}</b>: {v}" for k, v in e.items()
             if k not in ("source", "target", "key", "relation") and v not in (None, "")]
    return "<br>".join(lines) if lines else ""


def compute_layouts(data):
    """Positions are computed here, once, deterministically (seed=42), and
    embedded in the HTML. The browser runs no physics, so the graph never
    drifts or collapses into vis-network's initial ring.

    Two layouts:
    - `core` (default view, no Action nodes): Condition nodes only touch
      Actions, so for layout purposes only they get a virtual link to each
      Crop they have recommendations for; otherwise they'd float off alone.
      These virtual links are never drawn.
    - `full` (Actions shown): laid out on the real graph, starting from the
      core positions so the overall arrangement stays recognisable."""
    types = {n["id"]: n.get("type") for n in data["nodes"]}
    U = nx.Graph()
    U.add_nodes_from(types)
    U.add_edges_from((e["source"], e["target"]) for e in data["edges"])

    # Season records / years are hidden by default and would crowd the core
    # layout; they are placed only in the full layout.
    core = [n for n in U if types[n] not in ("Action", "SeasonRecord", "Year")]
    C = nx.Graph(U.subgraph(core))
    # Layout-only links: nodes that meet at the same Action (its Condition,
    # Crop and SoilType) are pulled together. Without this, nodes that only
    # touch Actions are isolated in the core graph and get flung to the
    # edge, which shrinks everything else into a dot.
    for a in (n for n in U if types[n] == "Action"):
        nb = [m for m in U.neighbors(a) if m in C]
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                C.add_edge(nb[i], nb[j])
    iso = [n for n in C if C.degree(n) == 0]
    assert not iso, f"isolated in core layout: {iso}"
    pos_core = nx.spring_layout(C, seed=42, k=1.8 / math.sqrt(len(C)), iterations=600)
    pos_full = nx.spring_layout(U, pos=dict(pos_core), seed=42,
                                k=1.6 / math.sqrt(len(U)), iterations=400)
    return pos_core, pos_full


def main():
    data = json.loads(GRAPH_JSON.read_text(encoding="utf-8"))
    pos_core, pos_full = compute_layouts(data)
    SCALE = 900

    vis_nodes = []
    for n in data["nodes"]:
        t = n.get("type", "Unknown")
        xf, yf = pos_full[n["id"]]
        xc, yc = pos_core.get(n["id"], (xf, yf))
        vis_nodes.append({
            "id": n["id"],
            # Action text lives in the tooltip/details panel, not on the
            # canvas -- 326 overlapping labels are unreadable.
            "label": "" if t == "Action" else node_label(n),
            "text": node_label(n),
            "title": node_title(n),
            "ntype": t,
            "color": {"background": TYPE_FILL.get(t, "#D3D1C7"), "border": TYPE_COLORS.get(t, "#444441")},
            "font": {"color": TYPE_COLORS.get(t, "#2C2C2A"), "size": 13},
            "size": TYPE_SIZE.get(t, 14),
            "xc": round(xc * SCALE, 1), "yc": round(yc * SCALE, 1),
            "xf": round(xf * SCALE, 1), "yf": round(yf * SCALE, 1),
        })

    # Collapse parallel edges (e.g. one Condition->Action RECOMMENDS per PDF
    # row) into one drawn edge; every underlying row stays in the tooltip.
    groups = {}
    for e in data["edges"]:
        rel = e.get("relation", e.get("key", ""))
        groups.setdefault((e["source"], e["target"], rel), []).append(e)

    vis_edges = []
    for i, ((src, tgt, rel), es) in enumerate(groups.items()):
        head = f"<b>{EDGE_LABELS.get(rel, rel)}</b>" + (f" &times; {len(es)} source rows" if len(es) > 1 else "")
        body = "<hr>".join(t for t in (edge_title(x) for x in es[:10]) if t)
        if len(es) > 10:
            body += f"<br>&hellip; and {len(es) - 10} more"
        vis_edges.append({
            "id": f"e{i}",
            "from": src,
            "to": tgt,
            "title": head + ("<br>" + body if body else ""),
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.4}},
            "color": {"color": "#B4B2A9", "opacity": 0.45},
            "width": 1 + min(3, math.log2(len(es))),
            "relation": rel,
            "count": len(es),
        })

    node_types = sorted({n["ntype"] for n in vis_nodes})
    edge_types = sorted({e["relation"] for e in vis_edges})

    payload = json.dumps({"nodes": vis_nodes, "edges": vis_edges}, ensure_ascii=False)
    type_colors_json = json.dumps(TYPE_FILL)

    counts_n = {}
    for n in vis_nodes:
        counts_n[n["ntype"]] = counts_n.get(n["ntype"], 0) + 1
    counts_e = {}
    for e in data["edges"]:
        counts_e[e["relation"]] = counts_e.get(e["relation"], 0) + 1
    info = {"nodes": {t: {"short": v[0], "long": v[1], "example": v[2], "count": counts_n.get(t, 0)}
                      for t, v in NODE_INFO.items()},
            "edges": {k: {"from": v[0], "to": v[1], "meaning": v[2], "count": counts_e.get(k, 0)}
                      for k, v in EDGE_INFO.items()}}

    html = HTML_TEMPLATE.replace("__PAYLOAD__", payload) \
                         .replace("__INFO__", json.dumps(info, ensure_ascii=False)) \
                         .replace("__NODE_TYPES__", json.dumps(node_types)) \
                         .replace("__EDGE_TYPES__", json.dumps(edge_types)) \
                         .replace("__TYPE_COLORS__", type_colors_json)

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Written: {OUT_HTML}")
    print(f"Nodes: {len(vis_nodes)}  Edges: {len(vis_edges)}")


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Drought KG viewer</title>
<script src="vendor/vis-network.min.js"></script>
<style>
  html, body { margin:0; padding:0; height:100%; font-family: -apple-system, Segoe UI, sans-serif; background:#F1EFE8; }
  #app { display:flex; height:100vh; }
  #sidebar { width:280px; flex-shrink:0; background:#fff; border-right:1px solid #D3D1C7; padding:16px; overflow-y:auto; box-sizing:border-box; }
  #network { flex:1; }
  h1 { font-size:16px; margin:0 0 4px; }
  .sub { font-size:12px; color:#5F5E5A; margin-bottom:16px; }
  .section { margin-bottom:18px; }
  .section h2 { font-size:12px; text-transform:uppercase; letter-spacing:.03em; color:#5F5E5A; margin:0 0 8px; }
  label { display:flex; align-items:center; gap:6px; font-size:13px; margin-bottom:6px; cursor:pointer; }
  .swatch { width:11px; height:11px; border-radius:3px; flex-shrink:0; display:inline-block; }
  input[type=text] { width:100%; box-sizing:border-box; padding:6px 8px; font-size:13px; border:1px solid #D3D1C7; border-radius:6px; margin-bottom:8px; }
  #details { font-size:12px; line-height:1.5; background:#F1EFE8; border-radius:8px; padding:10px; white-space:normal; word-break:break-word; }
  #details b { color:#2C2C2A; }
  #details .empty { color:#888780; }
  #legend-note { font-size:11px; color:#888780; }
  button { font-size:12px; padding:5px 10px; border-radius:6px; border:1px solid #D3D1C7; background:#fff; cursor:pointer; margin-right:6px; margin-bottom:6px; }
  button:hover { background:#F1EFE8; }
  #guide code { font-size:11px; background:#fff; padding:0 3px; border-radius:3px; }
  .meaning { background:#fff; border-left:3px solid #888780; padding:6px 8px; margin-bottom:8px; border-radius:4px; }
  #guide-overlay { display:none; position:fixed; inset:0; background:rgba(44,44,42,.45); z-index:10; }
  #guide-overlay.open { display:flex; align-items:center; justify-content:center; }
  #guide { background:#fff; border-radius:10px; width:min(920px, 94vw); max-height:88vh; overflow-y:auto;
           padding:20px 24px; box-sizing:border-box; font-size:13px; line-height:1.5; color:#2C2C2A; }
  #guide h3 { font-size:15px; margin:18px 0 8px; }
  #guide h3:first-of-type { margin-top:4px; }
  #guide table { border-collapse:collapse; width:100%; }
  #guide th, #guide td { text-align:left; vertical-align:top; padding:6px 8px; border-bottom:1px solid #E5E3DC; }
  #guide th { font-size:11px; text-transform:uppercase; letter-spacing:.03em; color:#5F5E5A; }
  #guide .num { color:#5F5E5A; white-space:nowrap; }
  #guide td:first-child { white-space:nowrap; }
  #guide .flow { font-family: ui-monospace, Consolas, monospace; font-size:12px; background:#F1EFE8; border-radius:6px;
                 padding:10px 12px; white-space:pre; overflow-x:auto; }
  #guide-close { float:right; }
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <h1>Drought decision-support KG</h1>
    <div class="sub">Condition &rarr; ConditionDetail &rarr; Action, by district, crop, soil and season. Tick Action or SeasonRecord to see recommendations and 2000&ndash;2025 history.</div>

    <button id="guide-btn">What do the nodes and relations mean?</button>

    <div class="section" style="margin-top:12px">
      <h2>Search</h2>
      <input type="text" id="search" placeholder="e.g. Beed, Soybean, drought...">
    </div>

    <div class="section">
      <h2>Node types</h2>
      <div id="type-filters"></div>
    </div>

    <div class="section">
      <h2>Relations</h2>
      <div id="edge-filters"></div>
    </div>

    <div class="section">
      <h2>View</h2>
      <button id="fit-btn">Fit to screen</button>
      <button id="physics-btn">Freeze layout</button>
      <label><input type="checkbox" id="hide-isolated" checked> Hide nodes with no visible edges</label>
      <div id="legend-note">Nodes that only look isolated because their edges are filtered out are hidden. E.g. Condition nodes connect only to Actions, so tick "Action" to see them.</div>
    </div>

    <div class="section">
      <h2>Details</h2>
      <div id="details"><span class="empty">Click a node or edge for details.</span></div>
    </div>
  </div>
  <div id="network"></div>
</div>

<div id="guide-overlay">
  <div id="guide">
    <button id="guide-close">Close</button>
    <h3>How the graph is organised</h3>
    <p>Three layers. <b>Profile</b>: what each district has (soils, crops). <b>Rules</b>: what the official
    contingency plan says to do under each condition. <b>History</b>: what actually happened in every season
    from 2000 to 2025, and the yield that followed.</p>
    <div class="flow">District --GROWS--> Crop --APPLIES_TO_CROP--+
   |                                        v
   |      Condition <-SUBTYPE_OF- ConditionDetail --RECOMMENDS--> Action --SUGGESTS_CROP--> Crop
   |                                        ^
   +-HAS_SEASON_RECORD-> SeasonRecord --EXPERIENCED
                             +--HAD_YIELD--> Crop</div>
    <p>Example: <i>Beed, Soybean, dry spell at flowering</i> &rarr; start at ConditionDetail
    <code>mid_dry_spell_flowering</code>, follow RECOMMENDS edges where district = Beed and crop = Soybean to get
    the actions and PDF pages; SUGGESTS_CROP gives alternatives; EXPERIENCED (backwards) gives the past Beed
    seasons with the same dry spell, and HAD_YIELD their soybean yield.</p>
    <h3>Node types</h3>
    <table id="guide-nodes"><tr><th>Type</th><th>Count</th><th>What it is</th><th>Example</th></tr></table>
    <h3>Relations</h3>
    <table id="guide-edges"><tr><th>Relation</th><th>From &rarr; To</th><th>Count</th><th>Meaning</th></tr></table>
  </div>
</div>

<script>
const graphData = __PAYLOAD__;
const NODE_TYPES = __NODE_TYPES__;
const EDGE_TYPES = __EDGE_TYPES__;
const TYPE_COLORS = __TYPE_COLORS__;
const INFO = __INFO__;

function nodeMeaning(t) {
  const i = INFO.nodes[t];
  return i ? "<div class='meaning'><b>" + t + "</b>: " + i.short + " " + i.long + "</div>" : "";
}
function edgeMeaning(k) {
  const i = INFO.edges[k];
  return i ? "<div class='meaning'><b>" + k + "</b> (" + i.from + " &rarr; " + i.to + "): " + i.meaning + "</div>" : "";
}

// vis-network shows a string title as plain text, so tooltips become DOM
// elements; the HTML string is kept in .html for the Details panel and search.
function asTooltip(item) {
  item.html = item.title || "";
  if (item.title) {
    const d = document.createElement("div");
    d.innerHTML = item.title;
    d.style.cssText = "max-width:360px;font-size:12px;line-height:1.45;white-space:normal;";
    item.title = d;
  }
}
graphData.nodes.forEach(asTooltip);
graphData.edges.forEach(asTooltip);

const allNodes = new vis.DataSet(graphData.nodes);
const allEdges = new vis.DataSet(graphData.edges);

// Actions (463) and the 260 season records start hidden; tick them on in the sidebar.
const activeTypes = new Set(NODE_TYPES.filter(t => !["Action", "SeasonRecord", "Year"].includes(t)));
const activeEdgeTypes = new Set(EDGE_TYPES.filter(k => k !== "APPLIES_TO_CROP"));
let network = null;
let physicsOn = false;
let showingFull = null;

// Filters toggle `hidden` in place instead of clearing and re-adding nodes,
// so positions are kept and nothing re-lays-out.
function applyFilters() {
  const hideIsolated = document.getElementById("hide-isolated").checked;
  const typeOn = new Set(allNodes.get().filter(n => activeTypes.has(n.ntype)).map(n => n.id));
  const touched = new Set();
  const edgeUpdates = allEdges.get().map(e => {
    const on = activeEdgeTypes.has(e.relation) && typeOn.has(e.from) && typeOn.has(e.to);
    if (on) { touched.add(e.from); touched.add(e.to); }
    return { id: e.id, hidden: !on };
  });
  const full = activeTypes.has("Action");
  const nodeUpdates = allNodes.get().map(n => {
    const u = { id: n.id, hidden: !(typeOn.has(n.id) &&
      (!hideIsolated || touched.has(n.id) || n.ntype === "District")) };
    if (full !== showingFull) { u.x = full ? n.xf : n.xc; u.y = full ? n.yf : n.yc; }
    return u;
  });
  allNodes.update(nodeUpdates);
  allEdges.update(edgeUpdates);
  const switched = full !== showingFull;
  showingFull = full;
  if (network && switched) fitVisible(true);
}

applyFilters();
document.getElementById("hide-isolated").addEventListener("change", applyFilters);

const container = document.getElementById("network");
// Positions come precomputed from Python; physics stays off so the graph is
// still. "Unfreeze layout" turns live physics on if you want to tidy it.
network = new vis.Network(container, { nodes: allNodes, edges: allEdges }, {
  physics: {
    enabled: false,
    solver: "barnesHut",
    barnesHut: { gravitationalConstant: -3000, springLength: 110, springConstant: 0.03, damping: 0.5 },
    minVelocity: 1
  },
  interaction: { hover: true, tooltipDelay: 100, hideEdgesOnDrag: true },
  edges: { smooth: false },
  nodes: { shape: "dot", borderWidth: 1.5 },
  layout: { improvedLayout: false }
});

// Fit only to visible nodes (vis includes hidden nodes in a plain fit()).
function fitVisible(animate) {
  const ids = allNodes.get().filter(n => !n.hidden).map(n => n.id);
  network.fit({ nodes: ids, animation: animate ? { duration: 400 } : false });
}
network.once("afterDrawing", () => fitVisible(false));

function fieldRow(k, v) { return "<div><b>" + k + ":</b> " + v + "</div>"; }

network.on("click", function (params) {
  const details = document.getElementById("details");
  if (params.nodes.length) {
    const n = allNodes.get(params.nodes[0]);
    let html = nodeMeaning(n.ntype) + fieldRow("type", n.ntype) + fieldRow("label", n.text || n.label);
    if (n.html) html += "<hr style='border:none;border-top:1px solid #D3D1C7;margin:6px 0'>" + n.html;
    details.innerHTML = html;
  } else if (params.edges.length) {
    const e = allEdges.get(params.edges[0]);
    let html = edgeMeaning(e.relation) + fieldRow("relation", e.relation);
    if (e.html) html += "<hr style='border:none;border-top:1px solid #D3D1C7;margin:6px 0'>" + e.html;
    details.innerHTML = html;
  } else {
    details.innerHTML = "<span class='empty'>Click a node or edge for details.</span>";
  }
});

// --- type filter checkboxes ---
const typeFiltersEl = document.getElementById("type-filters");
// One filter row: checkbox label; hovering it shows the meaning.
function filterRow(parent, labelHtml, descHtml, tooltip) {
  const label = document.createElement("label");
  label.innerHTML = labelHtml;
  label.title = tooltip;
  parent.appendChild(label);
  return label;
}

NODE_TYPES.forEach(t => {
  const id = "type-" + t;
  const i = INFO.nodes[t] || {short: "", long: "", example: "", count: ""};
  const label = filterRow(typeFiltersEl,
    '<input type="checkbox" id="' + id + '" ' + (activeTypes.has(t) ? "checked" : "") + '>' +
    '<span class="swatch" style="background:' + (TYPE_COLORS[t] || "#ccc") + '"></span>' + t +
    ' <span style="color:#888780;font-size:11px">' + i.count + '</span>',
    i.short + " " + i.long + (i.example ? "<br><i>e.g.</i> <code>" + i.example + "</code>" : ""), i.short);
  label.querySelector("input").addEventListener("change", (e) => {
    if (e.target.checked) activeTypes.add(t); else activeTypes.delete(t);
    applyFilters();
  });
});

const edgeFiltersEl = document.getElementById("edge-filters");
EDGE_TYPES.forEach(k => {
  const id = "edge-" + k;
  const i = INFO.edges[k] || {from: "?", to: "?", meaning: "", count: ""};
  const label = filterRow(edgeFiltersEl,
    '<input type="checkbox" id="' + id + '" ' + (activeEdgeTypes.has(k) ? "checked" : "") + '>' + k +
    ' <span style="color:#888780;font-size:11px">' + i.count + '</span>',
    "<b>" + i.from + " &rarr; " + i.to + "</b><br>" + i.meaning, i.from + " -> " + i.to + ": " + i.meaning);
  label.querySelector("input").addEventListener("change", (e) => {
    if (e.target.checked) activeEdgeTypes.add(k); else activeEdgeTypes.delete(k);
    applyFilters();
  });
});

// --- search ---
document.getElementById("search").addEventListener("input", (e) => {
  const q = e.target.value.trim().toLowerCase();
  if (!q) { network.unselectAll(); return; }
  const matches = allNodes.get().filter(n => !n.hidden &&
    ((n.text && n.text.toLowerCase().includes(q)) || (n.html && n.html.toLowerCase().includes(q)))
  ).map(n => n.id);
  network.selectNodes(matches);
  if (matches.length) network.focus(matches[0], { scale: 1.2, animation: true });
});

// --- guide overlay ---
const gn = document.getElementById("guide-nodes");
Object.entries(INFO.nodes).forEach(([t, i]) => {
  gn.insertAdjacentHTML("beforeend", "<tr><td><span class='swatch' style='background:" + (TYPE_COLORS[t] || "#ccc") +
    "'></span> <b>" + t + "</b></td><td class='num'>" + i.count + "</td><td>" + i.short + " " + i.long +
    "</td><td><code>" + i.example + "</code></td></tr>");
});
const ge = document.getElementById("guide-edges");
Object.entries(INFO.edges).forEach(([k, i]) => {
  ge.insertAdjacentHTML("beforeend", "<tr><td><b>" + k + "</b></td><td class='num'>" + i.from + " &rarr; " + i.to +
    "</td><td class='num'>" + i.count + "</td><td>" + i.meaning + "</td></tr>");
});
const overlay = document.getElementById("guide-overlay");
document.getElementById("guide-btn").addEventListener("click", () => overlay.classList.add("open"));
document.getElementById("guide-close").addEventListener("click", () => overlay.classList.remove("open"));
overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.classList.remove("open"); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") overlay.classList.remove("open"); });

document.getElementById("fit-btn").addEventListener("click", () => fitVisible(true));

document.getElementById("physics-btn").textContent = "Unfreeze layout";
document.getElementById("physics-btn").addEventListener("click", (e) => {
  physicsOn = !physicsOn;
  network.setOptions({ physics: { enabled: physicsOn } });
  e.target.textContent = physicsOn ? "Freeze layout" : "Unfreeze layout";
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
