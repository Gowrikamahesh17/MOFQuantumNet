"""Standalone architecture-diagram page for the MOFQuantumNet pipeline.

Run with: .venv/bin/streamlit run pipeline_diagram.py

Renders the end-to-end pipeline as one SVG diagram with a "Download as image"
button (client-side SVG -> canvas -> PNG export via a CCv2 component, no
server round-trip needed). Save the downloaded file to
outputs/pipeline_architecture.png to have it appear in README.md.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ui_theme import card_title, eyebrow, inject_theme, note  # noqa: E402

st.set_page_config(page_title="MOFQuantumNet — Pipeline Architecture", layout="wide", page_icon="🧭")
inject_theme()

with st.container(key="hero"):
    eyebrow("MOFQuantumNet")
    st.markdown("# Pipeline architecture")
    st.markdown(
        '<p style="color:var(--muted); font-size:16px; max-width:900px;">'
        "The end-to-end flow from raw MOF data to a final model comparison table. "
        "Use the button below the diagram to export it as a PNG."
        "</p>",
        unsafe_allow_html=True,
    )

with st.container(key="card-diagram"):
    card_title("End-to-end flow", "Two datasets, one pipeline — everything cascades from the dataset switch.")

    _DIAGRAM_HTML = """
<div style="background:#ffffff; border-radius:16px; padding:18px; overflow-x:auto;">
  <svg id="pipeline-svg" viewBox="0 0 1200 1260" width="100%" style="display:block; min-width:960px; font-family:Inter,ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;">
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"></path>
      </marker>
    </defs>

    <rect x="0" y="0" width="1200" height="1260" fill="#ffffff"></rect>

    <text x="600" y="45" text-anchor="middle" font-size="26" font-weight="800" fill="#1a2233">MOFQuantumNet — pipeline architecture</text>
    <text x="600" y="70" text-anchor="middle" font-size="13" fill="#64748b">Small dataset -&gt; classification only  ·  Large dataset -&gt; classification or regression</text>

    <!-- connectors -->
    <g stroke="#94a3b8" stroke-width="2" fill="none">
      <line x1="360" y1="172" x2="600" y2="212" marker-end="url(#arrow)"></line>
      <line x1="840" y1="172" x2="600" y2="212" marker-end="url(#arrow)"></line>

      <line x1="600" y1="284" x2="360" y2="324" marker-end="url(#arrow)"></line>
      <line x1="600" y1="284" x2="840" y2="324" marker-end="url(#arrow)"></line>

      <line x1="360" y1="412" x2="600" y2="432"></line>
      <line x1="840" y1="412" x2="600" y2="432"></line>
      <line x1="600" y1="432" x2="360" y2="452" marker-end="url(#arrow)"></line>
      <line x1="600" y1="432" x2="840" y2="452" marker-end="url(#arrow)"></line>

      <line x1="360" y1="540" x2="360" y2="580" marker-end="url(#arrow)"></line>
      <line x1="840" y1="540" x2="840" y2="1068" marker-end="url(#arrow)"></line>

      <line x1="360" y1="668" x2="360" y2="708" marker-end="url(#arrow)"></line>

      <line x1="360" y1="796" x2="208.5" y2="836" marker-end="url(#arrow)"></line>
      <line x1="360" y1="796" x2="360.5" y2="836" marker-end="url(#arrow)"></line>
      <line x1="360" y1="796" x2="512.5" y2="836" marker-end="url(#arrow)"></line>

      <line x1="208.5" y1="900" x2="360" y2="940" marker-end="url(#arrow)"></line>
      <line x1="360.5" y1="900" x2="360" y2="940" marker-end="url(#arrow)"></line>
      <line x1="512.5" y1="900" x2="360" y2="940" marker-end="url(#arrow)"></line>

      <line x1="360" y1="1028" x2="360" y2="1068" marker-end="url(#arrow)"></line>
    </g>
    <text x="855" y="805" font-size="10.5" font-style="italic" fill="#64748b">flat features — bypasses graph</text>
    <text x="855" y="818" font-size="10.5" font-style="italic" fill="#64748b">construction &amp; sparsification</text>

    <!-- Row A: data sources -->
    <g>
      <rect x="140" y="100" width="440" height="72" rx="12" fill="#eaf1ff" stroke="#2f6fed" stroke-width="1.5"></rect>
      <text x="360" y="126" text-anchor="middle" font-size="14.5" font-weight="700" fill="#1d4ed8">Small dataset</text>
      <text x="360" y="144" text-anchor="middle" font-size="11" fill="#475569">2,000 MOFs</text>
      <text x="360" y="158" text-anchor="middle" font-size="11" fill="#475569">SMILES_METAL_2000_NoPLD.csv</text>
    </g>
    <g>
      <rect x="620" y="100" width="440" height="72" rx="12" fill="#eaf1ff" stroke="#2f6fed" stroke-width="1.5"></rect>
      <text x="840" y="126" text-anchor="middle" font-size="14.5" font-weight="700" fill="#1d4ed8">Large dataset</text>
      <text x="840" y="144" text-anchor="middle" font-size="11" fill="#475569">14,296 MOFs</text>
      <text x="840" y="158" text-anchor="middle" font-size="11" fill="#475569">MOFCSD.csv</text>
    </g>

    <!-- Row B: ingestion -->
    <g>
      <rect x="140" y="212" width="920" height="72" rx="12" fill="#eaf1ff" stroke="#2f6fed" stroke-width="1.5"></rect>
      <text x="600" y="242" text-anchor="middle" font-size="14.5" font-weight="700" fill="#1d4ed8">Data ingestion — canonical schema</text>
      <text x="600" y="262" text-anchor="middle" font-size="11" fill="#475569">refcode · linker_smiles · metal · pld_category · pld_value</text>
    </g>

    <!-- Row C: feature engineering -->
    <g>
      <rect x="140" y="324" width="440" height="88" rx="12" fill="#f1ecff" stroke="#7c5cff" stroke-width="1.5"></rect>
      <text x="360" y="350" text-anchor="middle" font-size="14.5" font-weight="700" fill="#6d28d9">Compact features</text>
      <text x="360" y="368" text-anchor="middle" font-size="11" fill="#475569">7-dim — 6 metal descriptors + linker MW</text>
      <text x="360" y="384" text-anchor="middle" font-size="11" fill="#475569">small dataset only</text>
    </g>
    <g>
      <rect x="620" y="324" width="440" height="88" rx="12" fill="#f1ecff" stroke="#7c5cff" stroke-width="1.5"></rect>
      <text x="840" y="350" text-anchor="middle" font-size="14.5" font-weight="700" fill="#6d28d9">Fingerprint features</text>
      <text x="840" y="368" text-anchor="middle" font-size="11" fill="#475569">1,079-dim — Morgan fp + geometry + metal one-hot</text>
      <text x="840" y="384" text-anchor="middle" font-size="11" fill="#475569">large dataset only</text>
    </g>

    <!-- Row D: graph construction (left) / baseline models (right) -->
    <g>
      <rect x="140" y="452" width="440" height="88" rx="12" fill="#e6fbf7" stroke="#14b8a6" stroke-width="1.5"></rect>
      <text x="360" y="478" text-anchor="middle" font-size="14.5" font-weight="700" fill="#0f766e">Graph construction</text>
      <text x="360" y="496" text-anchor="middle" font-size="11" fill="#475569">4 candidates — threshold φ=0.9, k-NN (k=3,5,10)</text>
      <text x="360" y="512" text-anchor="middle" font-size="11" fill="#475569">linker Tanimoto + metal similarity</text>
    </g>
    <g>
      <rect x="620" y="452" width="440" height="88" rx="12" fill="#eef1f5" stroke="#64748b" stroke-width="1.5"></rect>
      <text x="840" y="478" text-anchor="middle" font-size="14.5" font-weight="700" fill="#334155">Baseline models</text>
      <text x="840" y="496" text-anchor="middle" font-size="11" fill="#475569">Random Forest · k-NN</text>
      <text x="840" y="512" text-anchor="middle" font-size="11" fill="#475569">flat feature vectors — no graph involved</text>
    </g>

    <!-- Row E: best graph selected -->
    <g>
      <rect x="140" y="580" width="440" height="88" rx="12" fill="#e6fbf7" stroke="#14b8a6" stroke-width="1.5"></rect>
      <text x="360" y="606" text-anchor="middle" font-size="14.5" font-weight="700" fill="#0f766e">Best graph selected</text>
      <text x="360" y="624" text-anchor="middle" font-size="11" fill="#475569">lowest isolated-node rate → highest modularity</text>
      <text x="360" y="640" text-anchor="middle" font-size="10" font-style="italic" fill="#64748b">cross-checked against downstream accuracy</text>
    </g>

    <!-- Row F: black hole sparsification -->
    <g>
      <rect x="140" y="708" width="440" height="88" rx="12" fill="#fff4e0" stroke="#f59e0b" stroke-width="1.5"></rect>
      <text x="360" y="734" text-anchor="middle" font-size="14.5" font-weight="700" fill="#b45309">Black Hole sparsification</text>
      <text x="360" y="752" text-anchor="middle" font-size="11" fill="#475569">gravity = degree + betweenness + edge-weight-sum</text>
      <text x="360" y="768" text-anchor="middle" font-size="11" fill="#475569">PLD-stratified node/edge pruning</text>
    </g>

    <!-- Row G: sparsification outputs -->
    <g>
      <rect x="140" y="836" width="137" height="64" rx="10" fill="#fff4e0" stroke="#f59e0b" stroke-width="1.5"></rect>
      <text x="208.5" y="862" text-anchor="middle" font-size="12.5" font-weight="700" fill="#b45309">Full graph</text>
      <text x="208.5" y="878" text-anchor="middle" font-size="10" fill="#475569">100% retained</text>
    </g>
    <g>
      <rect x="292" y="836" width="137" height="64" rx="10" fill="#fff4e0" stroke="#f59e0b" stroke-width="1.5"></rect>
      <text x="360.5" y="862" text-anchor="middle" font-size="12.5" font-weight="700" fill="#b45309">BH-30 (τ=0.3)</text>
      <text x="360.5" y="878" text-anchor="middle" font-size="10" fill="#475569">~66% retained</text>
    </g>
    <g>
      <rect x="444" y="836" width="137" height="64" rx="10" fill="#fff4e0" stroke="#f59e0b" stroke-width="1.5"></rect>
      <text x="512.5" y="862" text-anchor="middle" font-size="12.5" font-weight="700" fill="#b45309">BH-50 (τ=0.5)</text>
      <text x="512.5" y="878" text-anchor="middle" font-size="10" fill="#475569">~48% retained</text>
    </g>

    <!-- Row H: GNN training -->
    <g>
      <rect x="140" y="940" width="440" height="88" rx="12" fill="#ffe9e9" stroke="#ef4444" stroke-width="1.5"></rect>
      <text x="360" y="966" text-anchor="middle" font-size="14.5" font-weight="700" fill="#b91c1c">GNN training</text>
      <text x="360" y="984" text-anchor="middle" font-size="11" fill="#475569">GCN · GraphSAGE · GAT</text>
      <text x="360" y="1000" text-anchor="middle" font-size="11" fill="#475569">× Full / BH-30 / BH-50 → 9 runs</text>
    </g>

    <!-- Row I: master comparison table -->
    <g>
      <rect x="140" y="1068" width="920" height="88" rx="12" fill="#e8fbf0" stroke="#10b981" stroke-width="1.5"></rect>
      <text x="600" y="1094" text-anchor="middle" font-size="14.5" font-weight="700" fill="#047857">Master comparison table</text>
      <text x="600" y="1112" text-anchor="middle" font-size="11" fill="#475569">accuracy · macro-F1 · Cohen's κ (classification)</text>
      <text x="600" y="1128" text-anchor="middle" font-size="11" fill="#475569">MAE · RMSE · R² (regression)</text>
    </g>

    <!-- legend -->
    <g font-size="12" fill="#334155">
      <rect x="75" y="1195" width="12" height="12" rx="3" fill="#eaf1ff" stroke="#2f6fed"></rect>
      <text x="93" y="1205">Data ingestion</text>

      <rect x="225" y="1195" width="12" height="12" rx="3" fill="#f1ecff" stroke="#7c5cff"></rect>
      <text x="243" y="1205">Feature engineering</text>

      <rect x="400" y="1195" width="12" height="12" rx="3" fill="#e6fbf7" stroke="#14b8a6"></rect>
      <text x="418" y="1205">Graph construction</text>

      <rect x="570" y="1195" width="12" height="12" rx="3" fill="#fff4e0" stroke="#f59e0b"></rect>
      <text x="588" y="1205">Sparsification</text>

      <rect x="710" y="1195" width="12" height="12" rx="3" fill="#ffe9e9" stroke="#ef4444"></rect>
      <text x="728" y="1205">GNN modeling</text>

      <rect x="850" y="1195" width="12" height="12" rx="3" fill="#eef1f5" stroke="#64748b"></rect>
      <text x="868" y="1205">Baseline modeling</text>

      <rect x="1005" y="1195" width="12" height="12" rx="3" fill="#e8fbf0" stroke="#10b981"></rect>
      <text x="1023" y="1205">Evaluation</text>
    </g>
  </svg>
</div>
<div style="display:flex; justify-content:flex-end; margin-top:14px;">
  <button id="download-btn" type="button" style="
    background:#2f6fed; color:#ffffff; border:none; border-radius:10px;
    padding:10px 18px; font-size:14px; font-weight:600; cursor:pointer;
    font-family:Inter,ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;">
    Download as image
  </button>
</div>
"""

    _DIAGRAM_JS = """
export default function (component) {
  const { parentElement } = component

  const btn = parentElement.querySelector("#download-btn")
  const svg = parentElement.querySelector("#pipeline-svg")
  if (!btn || !svg) return

  btn.onclick = () => {
    const scale = 2
    const viewBox = svg.viewBox.baseVal
    const width = viewBox.width
    const height = viewBox.height

    const clone = svg.cloneNode(true)
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg")
    clone.setAttribute("width", String(width))
    clone.setAttribute("height", String(height))

    const svgString = new XMLSerializer().serializeToString(clone)
    const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" })
    const url = URL.createObjectURL(svgBlob)

    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement("canvas")
      canvas.width = width * scale
      canvas.height = height * scale
      const ctx = canvas.getContext("2d")
      ctx.fillStyle = "#ffffff"
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.scale(scale, scale)
      ctx.drawImage(img, 0, 0, width, height)
      URL.revokeObjectURL(url)

      canvas.toBlob((pngBlob) => {
        const pngUrl = URL.createObjectURL(pngBlob)
        const a = document.createElement("a")
        a.href = pngUrl
        a.download = "mofquantumnet_pipeline_architecture.png"
        a.click()
        URL.revokeObjectURL(pngUrl)
      }, "image/png")
    }
    img.onerror = () => {
      URL.revokeObjectURL(url)
    }
    img.src = url
  }
}
"""

    _pipeline_diagram = st.components.v2.component(
        "mofquantumnet_pipeline_diagram",
        html=_DIAGRAM_HTML,
        js=_DIAGRAM_JS,
    )
    _pipeline_diagram()

note(
    "Save the downloaded PNG to <code>outputs/pipeline_architecture.png</code> — README.md "
    "already links to it from the pipeline overview section."
)
