/* MOFQuantumNet Console — frontend controller.
 * No build step, no framework — plain fetch() against the FastAPI backend mounted at
 * the same origin (webapp/backend/api.py). Ports planning/mockup_console.html's design
 * as-is; every number here is a real API response, not mock data.
 */

const CATEGORY_ORDER = ["nonporous", "small pore", "medium pore", "large pore"];
const state = {
  ds: "large",
  task: "classification",
  datasets: null,          // /api/datasets response, fetched once
  labCache: {},             // key `${step}:${ds}:${task}` -> rendered already
  activeStep: 1,
};

async function apiGet(path) {
  const r = await fetch(path);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
async function apiPost(path, payload) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function pct(x) { return (x * 100).toFixed(1) + "%"; }

/* ---------- navigation ---------- */

function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  document.getElementById("view-" + name).classList.add("active");
  document.querySelectorAll(".rail-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
}

function showStep(n) {
  state.activeStep = n;
  document.querySelectorAll(".lab-panel").forEach((p) => p.classList.remove("active"));
  document.getElementById("step-" + n).classList.add("active");
  document.querySelectorAll(".step-btn").forEach((b, i) => b.classList.toggle("active", i === n - 1));
  loadStep(n);
}

function setDataset(ds) {
  state.ds = ds;
  document.body.setAttribute("data-ds", ds);
  document.querySelectorAll(".seg button[data-ds]").forEach((b) => b.classList.toggle("active", b.dataset.ds === ds));
  if (ds === "small") setTask("classification");
  resetPredictResult();
  loadStep(state.activeStep);
}

function setTask(t) {
  state.task = t;
  document.body.setAttribute("data-task", t);
  document.querySelectorAll(".seg button[data-task]").forEach((b) => b.classList.toggle("active", b.dataset.task === t));
  resetPredictResult();
  loadStep(state.activeStep);
}

function resetPredictResult() {
  document.getElementById("result-empty").style.display = "flex";
  document.getElementById("result-loading").style.display = "none";
  document.getElementById("result-body").style.display = "none";
}

/* ---------- Predict ---------- */

async function runPredict() {
  document.getElementById("result-empty").style.display = "none";
  document.getElementById("result-body").style.display = "none";
  const loading = document.getElementById("result-loading");
  loading.style.display = "flex";
  document.getElementById("loading-text").textContent =
    state.ds === "large" ? "Comparing against 14,296 known MOFs and running every model…" : "Comparing against 2,004 known MOFs and running every model…";
  document.getElementById("predict-btn").disabled = true;

  const payload = {
    dataset: state.ds,
    task: state.task,
    smiles: document.getElementById("smiles").value.trim(),
  };
  if (state.ds === "large") payload.metal = document.getElementById("metal").value;

  try {
    const result = await apiPost("/api/predict", payload);
    renderPredictResult(result);
  } catch (err) {
    document.getElementById("result-body").innerHTML =
      `<div class="callout danger"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v4M12 17h.01"/><circle cx="12" cy="12" r="9"/></svg><div><strong>Prediction failed.</strong> ${esc(err.message)}</div></div>`;
    document.getElementById("result-body").style.display = "block";
  } finally {
    loading.style.display = "none";
    document.getElementById("predict-btn").disabled = false;
  }
}

function renderPredictResult(result) {
  const board = result.leaderboard;
  const isClassification = result.task === "classification";
  const hasGraphRows = board.some((r) => r.type === "graph");

  let head, rows;
  if (isClassification) {
    head = "<tr><th>#</th><th>Model</th><th>Type</th><th>Predicted category</th><th>Confidence</th><th>Test accuracy</th></tr>";
    rows = board.map((r, i) => {
      const conf = r.confidence ? pct(r.confidence[r.predicted_category] || 0) : "—";
      return `<tr class="${i === 0 ? "picked" : ""}"><td>${r.rank}</td><td>${esc(r.model)}</td>` +
        `<td><span class="tag-${r.type === "baseline" ? "flat" : "graph"}">${r.type}</span></td>` +
        `<td>${esc(r.predicted_category)}</td><td class="num">${conf}</td><td class="num">${pct(r.measured_accuracy)}</td></tr>`;
    }).join("");
  } else {
    head = "<tr><th>#</th><th>Model</th><th>Type</th><th>Predicted PLD</th><th>Typical error</th><th>R²</th></tr>";
    rows = board.map((r, i) =>
      `<tr class="${i === 0 ? "picked" : ""}"><td>${r.rank}</td><td>${esc(r.model)}</td>` +
      `<td><span class="tag-${r.type === "baseline" ? "flat" : "graph"}">${r.type}</span></td>` +
      `<td class="num">${r.predicted_pld.toFixed(2)} Å</td>` +
      `<td class="num">${r.type === "baseline" ? "± " + r.measured_mae.toFixed(2) + " Å" : "± " + r.measured_mae.toFixed(2) + " Å"}</td>` +
      `<td class="num">${r.measured_r2.toFixed(3)}</td></tr>`
    ).join("");
  }

  const top = board[0];
  const context = isClassification
    ? `<strong>${esc(top.model)}</strong> is ranked first — it's right about ${pct(top.measured_accuracy)} of the time on MOFs like this one.`
    : `<strong>${esc(top.model)}</strong> is ranked first — typical error ${top.measured_mae.toFixed(2)} Å, R²=${top.measured_r2.toFixed(3)}.`;

  // Confidence (this specific input) and accuracy (the model's typical performance) are
  // different things — a low-confidence call on an otherwise-decent model is the model
  // being honest about an ambiguous case, not a bug. Flagged explicitly rather than left
  // for the reader to notice the gap and wonder about it unexplained.
  let confidenceNote = "";
  if (isClassification && top.confidence) {
    const topConfidence = top.confidence[top.predicted_category] || 0;
    if (top.measured_accuracy - topConfidence > 0.15) {
      confidenceNote = `<p class="result-context">This specific prediction is only ${pct(topConfidence)} confident — well below ${esc(top.model)}'s usual ${pct(top.measured_accuracy)} accuracy. That means this particular MOF is a genuinely harder, more ambiguous case for the model, not a data error.</p>`;
    }
  }

  const caveatLines = Object.keys(result.caveats || {}).map((k) => {
    const labels = {
      used_median_geometry: "This MOF is missing real pore-geometry data (only known for MOFs whose 3D structure has been simulated) — typical values were used instead, so accuracy here is likely lower than the benchmarked number.",
      used_median_metal_feat: "This dataset has no metal-identity column — typical metal-descriptor values were used instead of a specific metal.",
      unknown_metal: "The given metal wasn't among the 53 seen during training.",
      invalid_smiles: "That SMILES string couldn't be parsed — treated as an empty/unknown linker.",
    };
    return `<div class="callout gold"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 8v5M12 16h.01"/><circle cx="12" cy="12" r="9"/></svg><div>${labels[k] || k}</div></div>`;
  }).join("");

  let insertionNote = "";
  if (hasGraphRows) {
    insertionNote = `<div class="insertion-note">
      <svg viewBox="0 0 60 44" fill="none">
        <circle cx="10" cy="10" r="3" fill="var(--ink-faint)"/><circle cx="14" cy="30" r="3" fill="var(--ink-faint)"/><circle cx="34" cy="6" r="3" fill="var(--ink-faint)"/><circle cx="46" cy="18" r="3" fill="var(--ink-faint)"/><circle cx="36" cy="36" r="3" fill="var(--ink-faint)"/>
        <line x1="10" y1="10" x2="34" y2="6" stroke="var(--line)" stroke-width="1.4"/><line x1="14" y1="30" x2="36" y2="36" stroke="var(--line)" stroke-width="1.4"/><line x1="34" y1="6" x2="46" y2="18" stroke="var(--line)" stroke-width="1.4"/>
        <circle cx="28" cy="22" r="4.5" fill="var(--verdigris)"/>
        <line x1="28" y1="22" x2="10" y2="10" stroke="var(--verdigris)" stroke-width="1.6" stroke-dasharray="2 2"/>
        <line x1="28" y1="22" x2="14" y2="30" stroke="var(--verdigris)" stroke-width="1.6" stroke-dasharray="2 2"/>
        <line x1="28" y1="22" x2="46" y2="18" stroke="var(--verdigris)" stroke-width="1.6" stroke-dasharray="2 2"/>
      </svg>
      <p>The <span style="color:var(--verdigris);font-weight:700;">graph</span> models were inserted into their trained similarity graph via nearest chemical neighbors before predicting (see graph_variant / n_neighbors_found below). <strong>Random Forest and k-NN skip this entirely</strong> — they only look at this MOF's own feature vector.</p>
    </div>`;
  } else if (result.note) {
    insertionNote = `<div class="callout gold"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 8v5M12 16h.01"/><circle cx="12" cy="12" r="9"/></svg><div>${esc(result.note)}</div></div>`;
  }

  document.getElementById("result-body").innerHTML = `
    <p class="sub" style="margin-bottom:2px;">Ranked by each model's own measured test performance — best first.</p>
    <div class="table-wrap"><table class="data"><thead>${head}</thead><tbody>${rows}</tbody></table></div>
    <p class="result-context">${context}</p>
    ${confidenceNote}
    ${caveatLines}
    ${insertionNote}
    <button class="see-lab" onclick="showView('lab'); showStep(7)">See how this was validated <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M13 6l6 6-6 6"/></svg></button>
  `;
  document.getElementById("result-body").style.display = "block";
}

/* ---------- Lab: dispatch ---------- */

function loadStep(n) {
  const loaders = { 1: loadStep1, 2: loadStep2, 3: loadStep3, 4: loadStep4, 5: loadStep5, 6: loadStep6, 7: loadStep7 };
  loaders[n]();
}

function panelLoading(el) {
  el.innerHTML = `<div class="card"><div class="panel-loading"><div class="spinner sm"></div>Loading…</div></div>`;
}
function panelError(el, err) {
  el.innerHTML = `<div class="card"><div class="panel-error">Couldn't load this stage: ${esc(err.message)}</div></div>`;
}

/* ---------- Step 1: Data & EDA ---------- */

function renderHist(values) {
  if (!values || !values.length) return "";
  const max = Math.max(...values);
  return values.map((v) => `<div class="b" style="height:${Math.max(3, Math.round((v / max) * 100))}%"></div>`).join("");
}

async function loadStep1() {
  const el = document.getElementById("step-1");
  panelLoading(el);
  try {
    if (!state.datasets) state.datasets = await apiGet("/api/datasets");
    const d = state.datasets[state.ds];
    const maxCat = Math.max(...Object.values(d.category_counts));
    const barsHtml = CATEGORY_ORDER.map((cat) => {
      const count = d.category_counts[cat] || 0;
      return `<div class="bar-row"><span>${cat}</span><div class="bar-bg"><div class="bar-fill" style="width:${(count / maxCat) * 100}%"></div></div><span class="num">${count.toLocaleString()}</span></div>`;
    }).join("");

    let html = `<div class="card">
      <h3>Loading &amp; cleaning the data</h3>
      <p class="sub">Both source datasets normalized into one schema — refcode, linker SMILES, metal, PLD category, PLD value.</p>
      <div class="stat-grid">
        <div class="stat-tile"><div class="v num">${d.rows.toLocaleString()}</div><div class="l">MOFs loaded</div></div>
        <div class="stat-tile"><div class="v num">${d.columns}</div><div class="l">Columns</div></div>
        <div class="stat-tile"><div class="v num">${d.duplicate_refcodes}</div><div class="l">Duplicate refcodes</div></div>
        <div class="stat-tile"><div class="v num">${d.unique_metals ?? "n/a"}</div><div class="l">Unique metals</div></div>
      </div>
      <div class="bars">${barsHtml}</div>
    </div>`;

    if (d.pld_histogram) {
      html += `<div class="card"><h3>Continuous PLD distribution</h3><p class="sub">Right-skewed — most MOFs cluster at small pore sizes.</p><div class="vhist">${renderHist(d.pld_histogram)}</div></div>`;
    }
    if (d.metal_distribution) {
      const vals = Object.values(d.metal_distribution);
      html += `<div class="card"><h3>Metal-type distribution (${Object.keys(d.metal_distribution).length} unique metals)</h3><p class="sub">Copper and zinc dominate; most others appear only a handful of times.</p><div class="vhist">${renderHist(vals)}</div></div>`;
    }
    el.innerHTML = html;
  } catch (err) { panelError(el, err); }
}

/* ---------- Step 2: Feature Engineering ---------- */

async function loadStep2() {
  const el = document.getElementById("step-2");
  panelLoading(el);
  try {
    const f = await apiGet(`/api/lab/${state.ds}/features`);
    const schemeLabel = f.scheme === "fingerprint"
      ? "Fingerprint scheme: 1,024-bit Morgan fingerprint + 2 pore-geometry features + one-hot over real metals."
      : "Compact scheme: 6 metal descriptors + linker molecular weight, min-max scaled.";
    el.innerHTML = `<div class="card">
      <h3>Turning chemistry into numbers</h3>
      <p class="sub">${schemeLabel}</p>
      <div class="stat-grid">
        <div class="stat-tile"><div class="v num">${f.dimensions}</div><div class="l">Feature dimensions</div></div>
        <div class="stat-tile"><div class="v num">${f.invalid_smiles_count ?? 0}</div><div class="l">Malformed SMILES (fallback used)</div></div>
      </div>
      <div class="callout danger">
        <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v4M12 17h.01M10.3 3.9L2.5 17a2 2 0 0 0 1.7 3h15.6a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>
        <div><strong>Bug found &amp; fixed — target leakage.</strong> The reference code fed the actual pore size in as an input feature while also asking the model to predict it. Removed entirely here.</div>
      </div>
    </div>`;
  } catch (err) { panelError(el, err); }
}

/* ---------- Network graph rendering (shared by steps 3 & 4) ---------- */

const CAT_INDEX = { "nonporous": 0, "small pore": 1, "medium pore": 2, "large pore": 3 };

function normalizePositions(nodes) {
  const xs = nodes.map((n) => n.x), ys = nodes.map((n) => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = maxX - minX || 1, spanY = maxY - minY || 1;
  return nodes.map((n) => ({ ...n, sx: 20 + ((n.x - minX) / spanX) * 360, sy: 20 + ((n.y - minY) / spanY) * 200 }));
}

function renderNetworkSvg(svgId, nodes, edges, options) {
  const svg = document.getElementById(svgId);
  if (!svg) return;
  options = options || {};
  const survived = options.survived; // Map id -> bool, or undefined = all shown
  const fixedIds = options.fixedIds || new Set();
  const byId = {};
  nodes.forEach((n, i) => (byId[n.id] = { ...n, idx: i }));

  const parts = [];
  edges.forEach(([a, b]) => {
    const na = nodes[a], nb = nodes[b];
    if (!na || !nb) return;
    if (survived && (!survived.get(na.id) || !survived.get(nb.id))) return;
    parts.push(`<line class="edge" data-a="${a}" data-b="${b}" x1="${na.sx.toFixed(1)}" y1="${na.sy.toFixed(1)}" x2="${nb.sx.toFixed(1)}" y2="${nb.sy.toFixed(1)}"></line>`);
  });
  nodes.forEach((n, i) => {
    const alive = !survived || survived.get(n.id);
    const isFixed = fixedIds.has(n.id);
    const cat = CAT_INDEX[n.category] ?? 0;
    if (!alive) {
      parts.push(`<circle class="node ghost" cx="${n.sx.toFixed(1)}" cy="${n.sy.toFixed(1)}" r="4.5"></circle>`);
    } else {
      parts.push(`<circle class="node${isFixed ? " fixed" : ""}" data-i="${i}" cx="${n.sx.toFixed(1)}" cy="${n.sy.toFixed(1)}" r="${isFixed ? 5.5 : 4.5}" fill="var(--cat${cat})"></circle>`);
    }
  });
  svg.innerHTML = parts.join("");
  svg.querySelectorAll(".node:not(.ghost)").forEach((node) => {
    node.addEventListener("mouseenter", function () {
      const i = this.getAttribute("data-i");
      svg.querySelectorAll(".edge").forEach((edge) => {
        const on = edge.getAttribute("data-a") === i || edge.getAttribute("data-b") === i;
        edge.style.stroke = on ? "var(--copper)" : "var(--line)";
        edge.style.strokeWidth = on ? "1.8" : "1";
      });
    });
    node.addEventListener("mouseleave", () => {
      svg.querySelectorAll(".edge").forEach((edge) => { edge.style.stroke = "var(--line)"; edge.style.strokeWidth = "1"; });
    });
  });
}

const legendHtml = `<div class="legend">
  <span class="legend-dot"><i class="sw" style="background:var(--cat0)"></i>nonporous</span>
  <span class="legend-dot"><i class="sw" style="background:var(--cat1)"></i>small pore</span>
  <span class="legend-dot"><i class="sw" style="background:var(--cat2)"></i>medium pore</span>
  <span class="legend-dot"><i class="sw" style="background:var(--cat3)"></i>large pore</span>
</div>`;

/* ---------- Step 3: Graph Construction ---------- */

async function loadStep3() {
  const el = document.getElementById("step-3");
  panelLoading(el);
  try {
    const [g, sample] = await Promise.all([
      apiGet(`/api/lab/${state.ds}/graph`),
      apiGet(`/api/lab/${state.ds}/graph-sample`),
    ]);
    const rows = g.topology.map((row) => `<tr class="${row.config === g.best_graph ? "picked" : ""}">
      <td>${esc(row.config)}</td><td class="num">${row.edges.toLocaleString()}</td>
      <td class="num">${pct(row.isolated_node_rate)}</td><td class="num">${row.mean_degree.toFixed(2)}</td>
      <td class="num">${row.modularity.toFixed(3)}</td></tr>`).join("");

    el.innerHTML = `<div class="card">
        <h3>Connecting similar MOFs</h3>
        <p class="sub">Four candidate similarity graphs compared by isolated-node rate and modularity — computed live.</p>
        <div class="table-wrap"><table class="data"><thead><tr><th>Config</th><th>Edges</th><th>Isolated</th><th>Mean degree</th><th>Modularity</th></tr></thead><tbody>${rows}</tbody></table></div>
        <div class="callout gold"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 8v5M12 16h.01"/><circle cx="12" cy="12" r="9"/></svg><div><strong>${esc(g.best_graph)}</strong> selected — lowest isolation, highest modularity. See Findings for whether this is also the best-<em>performing</em> choice.</div></div>
      </div>
      <div class="card">
        <h3>Degree distribution — ${esc(g.best_graph)}</h3>
        <p class="sub">Full graph, computed live from the real edge list.</p>
        <div class="vhist dense">${renderHist(g.degree_histogram)}</div>
      </div>
      <div class="card">
        <h3>Interactive graph view</h3>
        <p class="sub">A real, connected sample of the selected graph — node positions from spring_layout, colored by each MOF's actual PLD category. Hover a node to trace its neighbors.</p>
        <div class="viz-toolbar">${legendHtml}</div>
        <div class="net-wrap">
          <div class="cap">${sample.n_sampled} nodes sampled from ${sample.n_total.toLocaleString()} total, ${esc(sample.best_graph)}</div>
          <svg id="net-full" class="net-svg" viewBox="0 0 400 240"></svg>
        </div>
      </div>`;

    renderNetworkSvg("net-full", normalizePositions(sample.nodes), sample.edges);
  } catch (err) { panelError(el, err); }
}

/* ---------- Step 4: Black Hole Pruning (+ retrain form) ---------- */

async function loadStep4() {
  const el = document.getElementById("step-4");
  panelLoading(el);
  try {
    const [p30, p50, sample] = await Promise.all([
      apiGet(`/api/lab/${state.ds}/pruning?tau=0.3`),
      apiGet(`/api/lab/${state.ds}/pruning?tau=0.5`),
      apiGet(`/api/lab/${state.ds}/graph-sample`),
    ]);
    const pruneSample = await apiGet(`/api/lab/${state.ds}/pruning-sample?tau=0.3`);

    el.innerHTML = `<div class="card">
        <h3>Trimming the map</h3>
        <p class="sub">Gravity score = degree + betweenness + edge-weight-sum, weighted equally (0.33/0.33/0.33) by default.</p>
        <div class="stat-grid">
          <div class="stat-tile"><div class="v num">${p30.metrics.node_retention_pct}%</div><div class="l">Nodes kept, τ=0.3</div></div>
          <div class="stat-tile"><div class="v num">${p30.metrics.edge_retention_pct}%</div><div class="l">Edges kept, τ=0.3</div></div>
          <div class="stat-tile"><div class="v num">${p50.metrics.node_retention_pct}%</div><div class="l">Nodes kept, τ=0.5</div></div>
          <div class="stat-tile"><div class="v num">${p50.metrics.edge_retention_pct}%</div><div class="l">Edges kept, τ=0.5</div></div>
        </div>
        <div class="callout good"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg><div>Trimming to roughly half the map costs almost nothing in downstream accuracy — see Findings.</div></div>
      </div>
      <div class="card">
        <h3>Before vs. after pruning</h3>
        <p class="sub">Same node sample, same layout — a node either survives or disappears, it never moves. Verdigris ring = fixed test node, always kept.</p>
        <div class="net-pair">
          <div><h5>Before (full graph)</h5><div class="net-wrap"><svg id="net-before" class="net-svg" viewBox="0 0 400 240"></svg></div></div>
          <div><h5>After (τ = 0.3)</h5><div class="net-wrap"><svg id="net-after" class="net-svg" viewBox="0 0 400 240"></svg></div></div>
        </div>
      </div>
      <div class="card">
        <h3>Retrain with different configuration</h3>
        <p class="sub">Adjust gravity weights and pruning threshold, then retrain. Real background job — takes ~4-10s (baselines, small dataset) up to real minutes for GNNs on the large dataset.</p>
        <div class="retrain-grid">
          <div class="slider-field"><label>Degree weight (α) <span id="rw-a-val">0.33</span></label><input type="range" id="rw-a" min="0" max="1" step="0.01" value="0.33" oninput="document.getElementById('rw-a-val').textContent=this.value"></div>
          <div class="slider-field"><label>Betweenness weight (β) <span id="rw-b-val">0.33</span></label><input type="range" id="rw-b" min="0" max="1" step="0.01" value="0.33" oninput="document.getElementById('rw-b-val').textContent=this.value"></div>
          <div class="slider-field"><label>Edge-weight-sum (γ) <span id="rw-c-val">0.33</span></label><input type="range" id="rw-c" min="0" max="1" step="0.01" value="0.33" oninput="document.getElementById('rw-c-val').textContent=this.value"></div>
        </div>
        <div class="field" style="max-width:220px;">
          <label>Pruning threshold (τ, baselines only)</label>
          <input type="range" id="rw-tau" min="0" max="0.9" step="0.05" value="0.3" oninput="document.getElementById('rw-tau-val').textContent=this.value">
          <div style="font-size:12px;color:var(--ink-muted);margin-top:2px;">τ = <span id="rw-tau-val">0.3</span></div>
        </div>
        <div class="field" style="max-width:260px;">
          <label>Retrain</label>
          <select id="rw-target">
            <option value="baselines">Baselines (Random Forest, k-NN) — fast</option>
            <option value="gnns">Graph models (GCN, GraphSAGE, GAT) — slow</option>
            <option value="both">Both</option>
          </select>
        </div>
        <div class="retrain-actions">
          <button class="btn-primary" style="width:auto;margin-top:0;" onclick="startRetrain()" id="retrain-btn">Retrain now</button>
        </div>
        <div class="job-status" id="retrain-status"></div>
      </div>`;

    const positioned = normalizePositions(sample.nodes);
    renderNetworkSvg("net-before", positioned, sample.edges);
    const survivedMap = new Map(pruneSample.nodes.map((n) => [n.id, n.survived]));
    const fixedIds = new Set(pruneSample.nodes.filter((n) => n.fixed).map((n) => n.id));
    renderNetworkSvg("net-after", positioned, sample.edges, { survived: survivedMap, fixedIds });
  } catch (err) { panelError(el, err); }
}

async function startRetrain() {
  const btn = document.getElementById("retrain-btn");
  const statusEl = document.getElementById("retrain-status");
  btn.disabled = true;
  const payload = {
    task: state.task,
    target: document.getElementById("rw-target").value,
    gravity_degree_weight: parseFloat(document.getElementById("rw-a").value),
    gravity_betweenness_weight: parseFloat(document.getElementById("rw-b").value),
    gravity_edge_weight_sum_weight: parseFloat(document.getElementById("rw-c").value),
    pruning_threshold: parseFloat(document.getElementById("rw-tau").value),
  };
  try {
    const res = await apiPost(`/api/lab/${state.ds}/retrain`, payload);
    statusEl.innerHTML = res.jobs.map((j) => `<div class="job-row" id="job-${j.job_id}"><div class="spinner sm"></div>Training ${esc(j.kind)}…</div>`).join("");
    res.jobs.forEach((j) => pollJob(j.job_id, j.kind));
  } catch (err) {
    statusEl.innerHTML = `<div class="job-row failed">Failed to start: ${esc(err.message)}</div>`;
    btn.disabled = false;
  }
}

async function pollJob(jobId, kind) {
  const row = document.getElementById(`job-${jobId}`);
  try {
    const job = await apiGet(`/api/jobs/${jobId}`);
    if (job.status === "running") {
      setTimeout(() => pollJob(jobId, kind), 2000);
      return;
    }
    document.getElementById("retrain-btn").disabled = false;
    if (job.status === "done") {
      row.className = "job-row done";
      row.innerHTML = `✓ ${esc(kind)} retrained.`;
      // invalidate anything downstream that could have changed
      loadStep(4);
      if (state.activeStep === 5) loadStep5();
      if (state.activeStep === 6) loadStep6();
      if (state.activeStep === 7) loadStep7();
    } else {
      row.className = "job-row failed";
      row.innerHTML = `✗ ${esc(kind)} failed: ${esc(job.error || "unknown error")}`;
    }
  } catch (err) {
    row.className = "job-row failed";
    row.innerHTML = `✗ ${esc(err.message)}`;
    document.getElementById("retrain-btn").disabled = false;
  }
}

/* ---------- Step 5: GNN Training ---------- */

function lossCurveSvg(history) {
  if (!history || !history.train_loss || !history.train_loss.length) return "";
  const build = (arr) => {
    const max = Math.max(...arr), min = Math.min(...arr);
    const span = max - min || 1;
    const n = arr.length;
    return arr.map((v, i) => `${(i / (n - 1 || 1) * 152 + 4).toFixed(1)},${(76 - ((v - min) / span) * 72).toFixed(1)}`).join(" ");
  };
  return `<svg viewBox="0 0 160 80" preserveAspectRatio="none">
    <polyline points="${build(history.train_loss)}" fill="none" stroke="var(--copper)" stroke-width="2"/>
    <polyline points="${build(history.val_loss)}" fill="none" stroke="var(--verdigris)" stroke-width="2"/>
  </svg>`;
}

async function loadStep5() {
  const el = document.getElementById("step-5");
  panelLoading(el);
  try {
    const t = await apiGet(`/api/lab/${state.ds}/training?task=${state.task}`);
    const progressRows = Object.entries(t.served_models).map(([name, info]) =>
      `<div class="progress-item done"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg>${esc(name)} — ${esc(info.variant)} — served, ${(info.metrics.accuracy ?? info.metrics.r2).toFixed(3)} ${info.metrics.accuracy !== undefined ? "accuracy" : "R²"}</div>`
    ).join("");

    let lossHtml = "";
    if (t.all_runs && t.all_runs.length) {
      lossHtml = `<div class="card"><h3>Training / validation loss curves</h3>
        <p class="sub">The variant each model actually serves predictions with. Early stopping (patience 20) watches a genuine held-out validation split, not the test set.</p>
        <div class="loss-grid">${Object.entries(t.served_models).map(([name, info]) => {
          const run = t.all_runs.find((r) => r.model === name && r.variant === info.variant);
          return `<div class="loss-card"><h5>${esc(name)} — ${esc(info.variant)}</h5>${lossCurveSvg(run && run.history)}<div class="loss-legend"><span><i style="background:var(--copper)"></i>train</span><span><i style="background:var(--verdigris)"></i>val</span></div></div>`;
        }).join("")}</div>
      </div>`;
    }

    el.innerHTML = `<div class="card">
        <h3>Training the graph models</h3>
        <p class="sub">GCN, GraphSAGE, GAT — each trained on the full graph, BH-30, and BH-50; the best-scoring variant per architecture is what Predict serves.</p>
        <p class="sub" style="margin-top:16px;font-weight:600;color:var(--ink);">Served models — ${esc(state.ds)}/${esc(state.task)}</p>
        <div class="progress-list">${progressRows}</div>
      </div>
      ${lossHtml}`;
  } catch (err) {
    if (String(err.message).includes("not trained")) {
      el.innerHTML = `<div class="card"><h3>Training the graph models</h3>
        <div class="callout gold"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 8v5M12 16h.01"/><circle cx="12" cy="12" r="9"/></svg>
        <div>${esc(err.message)} Use the retrain form in the Black Hole Pruning step, or run the CLI directly.</div></div></div>`;
    } else panelError(el, err);
  }
}

/* ---------- Step 6: Baseline Check ---------- */

async function loadStep6() {
  const el = document.getElementById("step-6");
  panelLoading(el);
  try {
    const b = await apiGet(`/api/lab/${state.ds}/baselines?task=${state.task}`);
    const isClassification = state.task === "classification";
    const metricKey = isClassification ? "accuracy" : "r2";
    const rf = b.master_table.find((r) => r.configuration.includes("Random Forest"));
    const knn = b.master_table.find((r) => r.configuration.includes("k-NN"));
    const gnnRows = b.master_table.filter((r) => r.type === "GNN");
    const bestGnn = gnnRows.length ? gnnRows.reduce((a, c) => (c[metricKey] > a[metricKey] ? c : a)) : null;

    const tiles = [
      rf ? `<div class="stat-tile"><div class="v num">${isClassification ? pct(rf.accuracy) : rf.r2.toFixed(3)}</div><div class="l">Random Forest ${isClassification ? "accuracy" : "R²"}</div></div>` : "",
      knn ? `<div class="stat-tile"><div class="v num">${isClassification ? pct(knn.accuracy) : knn.r2.toFixed(3)}</div><div class="l">k-NN ${isClassification ? "accuracy" : "R²"}</div></div>` : "",
      bestGnn ? `<div class="stat-tile"><div class="v num">${isClassification ? pct(bestGnn.accuracy) : bestGnn.r2.toFixed(3)}</div><div class="l">Best GNN ${isClassification ? "accuracy" : "R²"}</div></div>` : `<div class="stat-tile"><div class="v num">—</div><div class="l">GNNs not trained yet</div></div>`,
    ].join("");

    const rows = b.master_table.map((r) => `<tr class="${r.rank === 1 ? "picked" : ""}"><td>${r.rank}</td><td>${esc(r.configuration)}</td><td><span class="tag-${r.type === "Baseline" ? "flat" : "graph"}">${r.type}</span></td><td class="num">${isClassification ? pct(r.accuracy) : r.r2.toFixed(3)}</td></tr>`).join("");

    el.innerHTML = `<div class="card">
      <h3>The reality check</h3>
      <p class="sub">Random Forest and k-NN, trained on the same features with no graph at all, evaluated on the identical test set.</p>
      <div class="stat-grid">${tiles}</div>
      <div class="table-wrap"><table class="data"><thead><tr><th>#</th><th>Configuration</th><th>Type</th><th>${isClassification ? "Accuracy" : "R²"}</th></tr></thead><tbody>${rows}</tbody></table></div>
    </div>`;
  } catch (err) { panelError(el, err); }
}

/* ---------- Step 7: Findings ---------- */

async function loadStep7() {
  const el = document.getElementById("step-7");
  panelLoading(el);
  try {
    const f = await apiGet(`/api/lab/${state.ds}/findings?task=${state.task}`);
    const isClassification = state.task === "classification";
    const metricKey = isClassification ? "accuracy" : "r2";
    const fmt = (v) => (isClassification ? pct(v) : v.toFixed(3));

    let lines = `<div class="score-line win"><span class="who">${esc(f.best_baseline.model)} (baseline)</span><span class="val num">${fmt(f.best_baseline[metricKey])}</span></div>`;
    if (f.best_gnn) {
      lines += `<div class="score-line"><span class="who">${esc(f.best_gnn.model)} (${esc(f.best_gnn.variant)})</span><span class="val num">${fmt(f.best_gnn[metricKey])}</span></div>`;
    }

    el.innerHTML = `<div class="card">
      <h3>The headline finding</h3>
      <p class="sub">${esc(state.ds)} dataset · ${esc(state.task)} — computed live from whatever's currently trained.</p>
      <div class="scoreboard"><div class="score-card"><h4>${esc(state.ds)} · ${esc(state.task)}</h4>${lines}</div></div>
      ${f.best_gnn ? `<p class="result-context">${f.baseline_beats_gnn ? "The non-graph baseline beats the best graph model here" : "The best graph model beats the non-graph baseline here"} — the similarity graph is built from the same metal/linker similarity already sitting in each MOF's own feature vector, so message passing mostly re-derives information the model already had.</p>` : `<p class="result-context">Train the GNN models (Black Hole Pruning step) to see the full comparison.</p>`}
    </div>`;
  } catch (err) { panelError(el, err); }
}

/* ---------- boot ---------- */

document.body.setAttribute("data-ds", state.ds);
document.body.setAttribute("data-task", state.task);
loadStep(1);
