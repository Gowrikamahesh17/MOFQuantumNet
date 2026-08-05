"""Interactive (drag/zoom/pan) node-link graph viewer for the Streamlit app.

Renders a readable *sample* of a similarity graph (see
`graph_construction.py::sample_subgraph_for_viz`) — the full graphs (2,000-14,296
nodes) are unreadable as a node-link drawing. Built as an inline CCv2 component
(vanilla JS, no bundler/npm dependency) since this interaction (drag nodes, zoom,
pan, hover tooltips) has no native Streamlit widget.
"""

import networkx as nx
import streamlit as st

CATEGORY_COLORS = {
    "nonporous": "#64748b",
    "small pore": "#38bdf8",
    "medium pore": "#a78bfa",
    "large pore": "#22c55e",
}

_HTML = """
<div id="root">
  <div id="toolbar" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
    <span id="subtitle" style="font-size:12.5px; color:var(--st-text-color); opacity:0.72;"></span>
    <button id="reset-btn" type="button" style="
      background:transparent; color:var(--st-text-color); border:1px solid var(--st-border-color);
      border-radius:8px; padding:5px 12px; font-size:12.5px; cursor:pointer;
      font-family:inherit;">Reset view</button>
  </div>
  <div id="canvas-wrap" style="position:relative; border:1px solid var(--st-border-color);
      border-radius:12px; overflow:hidden; background:var(--st-secondary-background-color);">
    <svg id="graph-svg" width="100%" style="display:block; touch-action:none; cursor:grab;"></svg>
    <div id="tooltip" style="position:absolute; pointer-events:none; display:none;
        background:rgba(15,23,42,0.94); color:#e5e7eb; padding:6px 10px; border-radius:8px;
        font-size:12px; white-space:nowrap; z-index:5;"></div>
  </div>
  <div id="legend" style="display:flex; gap:16px; flex-wrap:wrap; margin-top:10px; font-size:11.5px;"></div>
</div>
"""

_JS = """
export default function (component) {
  const { data, parentElement } = component
  const nodes = data.nodes || []
  const edges = data.edges || []
  const categoryColors = data.categoryColors || {}
  const height = data.height || 480
  const vbW = 900

  parentElement.querySelector("#subtitle").textContent = data.subtitle || ""

  const svg = parentElement.querySelector("#graph-svg")
  const tooltip = parentElement.querySelector("#tooltip")
  const legend = parentElement.querySelector("#legend")
  const wrap = parentElement.querySelector("#canvas-wrap")

  if (nodes.length === 0) {
    svg.style.height = "80px"
    svg.innerHTML = ""
    parentElement.querySelector("#toolbar").style.display = "none"
    legend.innerHTML = ""
    const empty = document.createElement("div")
    empty.style.cssText = "padding:20px; text-align:center; color:var(--st-text-color); opacity:0.6; font-size:13px;"
    empty.textContent = "No nodes survived pruning at this threshold."
    wrap.appendChild(empty)
    return
  }

  const xs = nodes.map((n) => n.x)
  const ys = nodes.map((n) => n.y)
  const minX = Math.min(...xs), maxX = Math.max(...xs)
  const minY = Math.min(...ys), maxY = Math.max(...ys)
  const spanX = (maxX - minX) || 1
  const spanY = (maxY - minY) || 1
  const pad = 36
  const vbH = height

  const originalPositions = {}
  nodes.forEach((n) => {
    originalPositions[n.id] = {
      px: pad + ((n.x - minX) / spanX) * (vbW - 2 * pad),
      py: pad + ((n.y - minY) / spanY) * (vbH - 2 * pad),
    }
  })
  const nodePositions = {}
  Object.keys(originalPositions).forEach((id) => { nodePositions[id] = { ...originalPositions[id] } })

  const nodesById = {}
  nodes.forEach((n) => { nodesById[n.id] = n })

  svg.setAttribute("viewBox", `0 0 ${vbW} ${vbH}`)
  svg.style.height = height + "px"

  let markup = '<g id="viewport">'
  edges.forEach((e, i) => {
    const a = nodePositions[e.source]
    const b = nodePositions[e.target]
    if (!a || !b) return
    markup += `<line data-edge="${i}" data-source="${e.source}" data-target="${e.target}" `
      + `x1="${a.px}" y1="${a.py}" x2="${b.px}" y2="${b.py}" `
      + 'stroke="#94a3b8" stroke-opacity="0.35" stroke-width="1.2"></line>'
  })
  nodes.forEach((n) => {
    const p = nodePositions[n.id]
    const color = categoryColors[n.category] || "#94a3b8"
    markup += `<circle data-id="${n.id}" cx="${p.px}" cy="${p.py}" r="7" fill="${color}" `
      + 'stroke="#0b1120" stroke-width="1.4" style="cursor:grab;"></circle>'
  })
  markup += "</g>"
  svg.innerHTML = markup

  const viewport = svg.querySelector("#viewport")

  legend.innerHTML = Object.entries(categoryColors)
    .map(([cat, color]) => (
      '<span style="display:inline-flex; align-items:center; gap:5px; color:var(--st-text-color); opacity:.85;">'
      + `<span style="width:10px; height:10px; border-radius:3px; background:${color}; display:inline-block;"></span>${cat}`
      + "</span>"
    ))
    .join("")

  let scale = 1, tx = 0, ty = 0
  function applyTransform() {
    viewport.setAttribute("transform", `translate(${tx},${ty}) scale(${scale})`)
  }

  function updateEdgesFor(nodeId) {
    const p = nodePositions[nodeId]
    svg.querySelectorAll(`line[data-source="${nodeId}"]`).forEach((l) => { l.setAttribute("x1", p.px); l.setAttribute("y1", p.py) })
    svg.querySelectorAll(`line[data-target="${nodeId}"]`).forEach((l) => { l.setAttribute("x2", p.px); l.setAttribute("y2", p.py) })
  }

  function screenToViewport(evt) {
    const rect = svg.getBoundingClientRect()
    const k = vbW / rect.width
    const x = (evt.clientX - rect.left) * k
    const y = (evt.clientY - rect.top) * k
    return { x: (x - tx) / scale, y: (y - ty) / scale }
  }

  let draggingId = null

  svg.querySelectorAll("circle[data-id]").forEach((circle) => {
    const id = circle.getAttribute("data-id")

    circle.addEventListener("pointerdown", (evt) => {
      evt.stopPropagation()
      draggingId = id
      circle.setPointerCapture(evt.pointerId)
      circle.style.cursor = "grabbing"
    })
    circle.addEventListener("pointerup", (evt) => {
      draggingId = null
      circle.style.cursor = "grab"
    })
    circle.addEventListener("pointermove", (evt) => {
      const wrapRect = wrap.getBoundingClientRect()
      tooltip.style.left = (evt.clientX - wrapRect.left + 14) + "px"
      tooltip.style.top = (evt.clientY - wrapRect.top + 10) + "px"
      if (draggingId !== id) return
      const p = screenToViewport(evt)
      circle.setAttribute("cx", p.x)
      circle.setAttribute("cy", p.y)
      nodePositions[id] = { px: p.x, py: p.y }
      updateEdgesFor(id)
    })
    circle.addEventListener("pointerenter", () => {
      const n = nodesById[id]
      tooltip.style.display = "block"
      tooltip.textContent = `${n.label} — ${n.category}`
    })
    circle.addEventListener("pointerleave", () => { tooltip.style.display = "none" })
  })

  let panning = false
  let panStart = null
  svg.addEventListener("pointerdown", (evt) => {
    if (evt.target.tagName === "circle") return
    panning = true
    panStart = { x: evt.clientX, y: evt.clientY, tx, ty }
    svg.setPointerCapture(evt.pointerId)
    svg.style.cursor = "grabbing"
  })
  svg.addEventListener("pointermove", (evt) => {
    if (!panning) return
    const rect = svg.getBoundingClientRect()
    const k = vbW / rect.width
    tx = panStart.tx + (evt.clientX - panStart.x) * k
    ty = panStart.ty + (evt.clientY - panStart.y) * k
    applyTransform()
  })
  svg.addEventListener("pointerup", () => { panning = false; svg.style.cursor = "grab" })
  svg.addEventListener("pointerleave", () => { panning = false })

  svg.addEventListener("wheel", (evt) => {
    evt.preventDefault()
    const factor = evt.deltaY < 0 ? 1.1 : 0.9
    scale = Math.min(4, Math.max(0.3, scale * factor))
    applyTransform()
  }, { passive: false })

  parentElement.querySelector("#reset-btn").onclick = () => {
    scale = 1; tx = 0; ty = 0
    applyTransform()
    Object.keys(originalPositions).forEach((id) => {
      nodePositions[id] = { ...originalPositions[id] }
      const circle = svg.querySelector(`circle[data-id="${id}"]`)
      if (circle) {
        circle.setAttribute("cx", nodePositions[id].px)
        circle.setAttribute("cy", nodePositions[id].py)
      }
      updateEdgesFor(id)
    })
  }
}
"""

_GRAPH_COMPONENT = st.components.v2.component(
    "mofquantumnet_interactive_graph",
    html=_HTML,
    js=_JS,
)


def render_interactive_graph(
    graph: nx.Graph,
    node_ids: list[int],
    positions: dict[int, tuple[float, float]],
    refcodes,
    categories,
    key: str,
    subtitle: str = "",
    height: int = 480,
) -> None:
    """Renders `graph`'s induced subgraph on `node_ids` as a draggable/zoomable
    node-link diagram. `positions` (from nx.spring_layout, keyed by node id) is
    passed in rather than recomputed here, so a "before" and "after" pruning view
    can reuse identical coordinates and stay visually comparable."""
    graph_nodes = set(graph.nodes())
    node_id_set = {n for n in node_ids if n in graph_nodes and n in positions}
    nodes = [
        {
            "id": str(n),
            "x": float(positions[n][0]),
            "y": float(positions[n][1]),
            "label": str(refcodes[n]),
            "category": str(categories[n]),
        }
        for n in node_ids
        if n in node_id_set
    ]
    edges = [
        {"source": str(u), "target": str(v)}
        for u, v in graph.subgraph(node_id_set).edges()
        if u in positions and v in positions
    ]

    _GRAPH_COMPONENT(
        key=key,
        data={
            "nodes": nodes,
            "edges": edges,
            "categoryColors": CATEGORY_COLORS,
            "subtitle": subtitle,
            "height": height,
        },
    )
