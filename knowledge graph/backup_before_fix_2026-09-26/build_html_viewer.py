"""
Render knowledge_graph.json as a self-contained, interactive HTML page
(vis-network via CDN, data embedded inline -- works standalone via
file://, no local server needed).

Run from the project root:  python3 knowledge_graph/scripts/build_html_viewer.py
Then open knowledge_graph/output/knowledge_graph_viewer.html in a browser.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRAPH_JSON = ROOT / "knowledge_graph" / "output" / "knowledge_graph.json"
OUT_HTML = ROOT / "knowledge_graph" / "output" / "knowledge_graph_viewer.html"

TYPE_COLORS = {
    "District": "#0C447C",     # blue 800
    "SoilType": "#633806",     # amber 800
    "Season": "#27500A",       # green 800
    "Crop": "#085041",         # teal 800
    "Condition": "#791F1F",    # red 800
    "Action": "#3C3489",       # purple 800
}
TYPE_FILL = {
    "District": "#B5D4F4",
    "SoilType": "#FAC775",
    "Season": "#C0DD97",
    "Crop": "#9FE1CB",
    "Condition": "#F7C1C1",
    "Action": "#CECBF6",
}
TYPE_SIZE = {
    "District": 26, "SoilType": 16, "Season": 22,
    "Crop": 18, "Condition": 22, "Action": 8,
}

EDGE_LABELS = {
    "HAS_SOIL": "has soil",
    "GROWS": "grows",
    "IN_SEASON": "in season",
    "RECOMMENDS": "recommends",
    "APPLIES_TO_CROP": "applies to",
}


def node_label(n):
    t = n.get("type")
    if t == "Action":
        return (n.get("change_in_cropping") or n.get("agronomic_measures") or "action")[:28]
    return n.get("name", n.get("id", "?"))[:28]


def node_title(n):
    # HTML tooltip shown on hover
    lines = [f"<b>{t}</b>: {v}" for t, v in n.items() if v not in (None, "")]
    return "<br>".join(lines)


def edge_title(e):
    lines = [f"<b>{k}</b>: {v}" for k, v in e.items()
             if k not in ("source", "target", "key") and v not in (None, "")]
    return "<br>".join(lines) if lines else ""


def main():
    data = json.loads(GRAPH_JSON.read_text(encoding="utf-8"))

    vis_nodes = []
    for n in data["nodes"]:
        t = n.get("type", "Unknown")
        vis_nodes.append({
            "id": n["id"],
            "label": node_label(n),
            "title": node_title(n),
            "group": t,
            "color": {"background": TYPE_FILL.get(t, "#D3D1C7"), "border": TYPE_COLORS.get(t, "#444441")},
            "font": {"color": TYPE_COLORS.get(t, "#2C2C2A"), "size": 12 if t != "Action" else 9},
            "size": TYPE_SIZE.get(t, 14),
        })

    vis_edges = []
    for e in data["edges"]:
        key = e.get("key", "")
        vis_edges.append({
            "from": e["source"],
            "to": e["target"],
            "label": EDGE_LABELS.get(key, key),
            "title": edge_title(e),
            "arrows": "to",
            "color": {"color": "#B4B2A9", "opacity": 0.5},
            "font": {"size": 8, "color": "#888780", "strokeWidth": 0},
            "relation": key,
        })

    node_types = sorted({n["group"] for n in vis_nodes})
    edge_types = sorted({e["relation"] for e in vis_edges})

    payload = json.dumps({"nodes": vis_nodes, "edges": vis_edges}, ensure_ascii=False)
    type_colors_json = json.dumps(TYPE_FILL)

    html = HTML_TEMPLATE.replace("__PAYLOAD__", payload) \
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
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <h1>Drought decision-support KG</h1>
    <div class="sub">District &rarr; SoilType &rarr; Crop &rarr; Season &rarr; Condition &rarr; Action</div>

    <div class="section">
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
    </div>

    <div class="section">
      <h2>Details</h2>
      <div id="details"><span class="empty">Click a node or edge for details.</span></div>
    </div>
  </div>
  <div id="network"></div>
</div>

<script>
const graphData = __PAYLOAD__;
const NODE_TYPES = __NODE_TYPES__;
const EDGE_TYPES = __EDGE_TYPES__;
const TYPE_COLORS = __TYPE_COLORS__;

const allNodes = new vis.DataSet(graphData.nodes);
const allEdges = new vis.DataSet(graphData.edges);

// Default: hide Action nodes (315 of them) to start with a readable view.
const activeTypes = new Set(NODE_TYPES.filter(t => t !== "Action"));
const activeEdgeTypes = new Set(EDGE_TYPES.filter(k => k !== "APPLIES_TO_CROP"));

const displayedNodes = new vis.DataSet();
const displayedEdges = new vis.DataSet();

function applyFilters() {
  const visibleNodeIds = new Set(
    allNodes.get().filter(n => activeTypes.has(n.group)).map(n => n.id)
  );
  const visibleEdges = allEdges.get().filter(e =>
    activeEdgeTypes.has(e.relation) &&
    visibleNodeIds.has(e.from) && visibleNodeIds.has(e.to)
  );
  // drop nodes left fully disconnected only if they're Action/SoilType-scale (keep Districts always)
  displayedNodes.clear();
  displayedEdges.clear();
  displayedNodes.add(allNodes.get().filter(n => visibleNodeIds.has(n.id)));
  displayedEdges.add(visibleEdges);
}

applyFilters();

const container = document.getElementById("network");
const network = new vis.Network(container, { nodes: displayedNodes, edges: displayedEdges }, {
  physics: { stabilization: { iterations: 150 }, barnesHut: { gravitationalConstant: -6000, springLength: 110, springConstant: 0.03 } },
  interaction: { hover: true, tooltipDelay: 100 },
  edges: { smooth: { type: "continuous", roundness: 0.3 } },
  nodes: { shape: "dot", borderWidth: 1.5 }
});

function fieldRow(k, v) { return "<div><b>" + k + ":</b> " + v + "</div>"; }

network.on("click", function (params) {
  const details = document.getElementById("details");
  if (params.nodes.length) {
    const n = allNodes.get(params.nodes[0]);
    let html = fieldRow("type", n.group) + fieldRow("label", n.label);
    if (n.title) html += "<hr style='border:none;border-top:1px solid #D3D1C7;margin:6px 0'>" + n.title;
    details.innerHTML = html;
  } else if (params.edges.length) {
    const e = allEdges.get(params.edges[0]);
    let html = fieldRow("relation", e.relation);
    if (e.title) html += "<hr style='border:none;border-top:1px solid #D3D1C7;margin:6px 0'>" + e.title;
    details.innerHTML = html;
  } else {
    details.innerHTML = "<span class='empty'>Click a node or edge for details.</span>";
  }
});

// --- type filter checkboxes ---
const typeFiltersEl = document.getElementById("type-filters");
NODE_TYPES.forEach(t => {
  const id = "type-" + t;
  const label = document.createElement("label");
  label.innerHTML = '<input type="checkbox" id="' + id + '" ' + (activeTypes.has(t) ? "checked" : "") + '>' +
    '<span class="swatch" style="background:' + (TYPE_COLORS[t] || "#ccc") + '"></span>' + t;
  typeFiltersEl.appendChild(label);
  label.querySelector("input").addEventListener("change", (e) => {
    if (e.target.checked) activeTypes.add(t); else activeTypes.delete(t);
    applyFilters();
  });
});

const edgeFiltersEl = document.getElementById("edge-filters");
EDGE_TYPES.forEach(k => {
  const id = "edge-" + k;
  const label = document.createElement("label");
  label.innerHTML = '<input type="checkbox" id="' + id + '" ' + (activeEdgeTypes.has(k) ? "checked" : "") + '>' + k;
  edgeFiltersEl.appendChild(label);
  label.querySelector("input").addEventListener("change", (e) => {
    if (e.target.checked) activeEdgeTypes.add(k); else activeEdgeTypes.delete(k);
    applyFilters();
  });
});

// --- search ---
document.getElementById("search").addEventListener("input", (e) => {
  const q = e.target.value.trim().toLowerCase();
  if (!q) { network.unselectAll(); return; }
  const matches = displayedNodes.get().filter(n =>
    (n.label && n.label.toLowerCase().includes(q)) || (n.title && n.title.toLowerCase().includes(q))
  ).map(n => n.id);
  network.selectNodes(matches);
  if (matches.length) network.focus(matches[0], { scale: 1.2, animation: true });
});

document.getElementById("fit-btn").addEventListener("click", () => network.fit({ animation: true }));

let physicsOn = true;
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
