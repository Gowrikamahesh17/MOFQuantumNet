"""Interactive control panel for the MOFQuantumNet pipeline.

Run with: .venv/bin/streamlit run app.py

Each tab maps to one development phase from planning/DEVELOPMENT_TODO.md.
Only Phase 1 (Data & EDA) is functional so far; later tabs are placeholders
that light up as each phase is implemented, so this app always reflects the
pipeline's real current state rather than promising features that don't exist.

Per the 2026-07-21 meeting with Prof. Jalali: dataset selection cascades into
feature scheme (derived, not independently chosen) and task options (only the
large dataset offers classification vs. regression); Black Hole's gravity
weights and pruning threshold are configurable sliders, not fixed constants.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import DATASET_CHOICES, TASK_CHOICES, PipelineConfig  # noqa: E402
from data_ingestion import load_dataset  # noqa: E402
from feature_engineering import build_features  # noqa: E402
from graph_construction import (  # noqa: E402
    build_similarity_graphs,
    compute_topology_metrics,
    plot_degree_distributions,
    select_best_graph,
)
from ui_theme import bar_row, card_title, eyebrow, inject_theme, note, pill, tile  # noqa: E402

st.set_page_config(page_title="MOFQuantumNet Pipeline", layout="wide", page_icon="🕸️")
inject_theme()


# Expensive steps (feature engineering, O(n^2)-ish graph construction) are cached so
# switching tabs or moving a Black Hole slider doesn't force recomputation.
@st.cache_data(show_spinner=False)
def cached_load_dataset(name: str):
    return load_dataset(name)


@st.cache_data(show_spinner=False)
def cached_build_features(df: pd.DataFrame, scheme: str):
    return build_features(df, scheme)


@st.cache_data(show_spinner=False)
def cached_build_graphs(df: pd.DataFrame, dataset: str):
    return build_similarity_graphs(df, dataset)

# --- Sidebar: the root switch (dataset) + what cascades from it ---
st.sidebar.markdown("### Pipeline configuration")
dataset_choice = st.sidebar.radio(
    "Dataset",
    DATASET_CHOICES,
    format_func=lambda d: "Small — 2,000 MOF (MOFGalaxyNet)" if d == "small" else "Large — 14,296 MOF (BlackHole/MOFCSD)",
    help="Confirmed with Prof. Jalali 2026-07-21: both datasets stay in scope, configurable — this is the one root switch everything else cascades from.",
)

feature_scheme = "compact" if dataset_choice == "small" else "fingerprint"
if dataset_choice == "small":
    task_choice = "classification"
    st.sidebar.caption("Task: **Classification only** — small dataset has no continuous PLD to regress on.")
else:
    task_choice = st.sidebar.radio(
        "Task",
        TASK_CHOICES,
        format_func=lambda t: "Classification (4 PLD buckets)" if t == "classification" else "Regression (exact PLD, Å)",
        help="Only offered for the large dataset, which has continuous PLD.",
    )
st.sidebar.caption(
    f"Feature scheme: **{'Compact (7-dim)' if feature_scheme == 'compact' else 'Fingerprint (Morgan + geometry + metal one-hot)'}** "
    "(auto-selected by dataset, not independently configurable)."
)

config = PipelineConfig(dataset=dataset_choice, task=task_choice)
st.sidebar.caption("Switching here only affects tabs marked ✅ — later phases pick this up as they're built.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Phase progress**")
st.sidebar.markdown(
    pill("Phase 0 done", "green") + pill("Phase 1 done", "green") + pill("Phase 2 done", "green")
    + pill("Phase 3 done", "green") + pill("Phase 4–7 pending", "orange"),
    unsafe_allow_html=True,
)

# --- Hero ---
with st.container(key="hero"):
    eyebrow("Case Study 2 · Graph Neural Networks for MOF Property Prediction")
    st.markdown("# MOFQuantumNet — Pipeline Control Panel")
    st.markdown(
        '<p style="color:var(--muted); font-size:16px; max-width:900px;">'
        "Reimplementing MOFGalaxyNet + Black Hole in PyTorch Geometric — graph construction, "
        "gravity-based sparsification, GNN training, and baseline comparison for MOF Pore "
        "Limiting Diameter prediction.</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading dataset..."):
        df = cached_load_dataset(config.dataset)

    h1, h2, h3, h4, h5 = st.columns(5)
    with h1:
        tile(config.dataset.capitalize(), "Active dataset")
    with h2:
        tile(f"{df.shape[0]:,}", "MOFs loaded")
    with h3:
        tile(config.task.capitalize(), "Task")
    with h4:
        tile("4 / 7", "Phases complete")
    with h5:
        n_metals = df["metal"].nunique() if df["metal"].notna().any() else "N/A"
        tile(n_metals, "Unique metals")

tabs = st.tabs([
    "1. Data & EDA ✅",
    "2. Feature Engineering ✅",
    "3. Graph Construction ✅",
    "4. Black Hole Sparsification",
    "5. GNN Training",
    "6. Baseline Comparison",
    "7. Results Summary",
])

# --- Tab 1: Data & EDA (functional) ---
with tabs[0]:
    with st.container(key="card-eda-summary"):
        card_title("Dataset summary", f"Currently viewing the {config.dataset} dataset")
        c1, c2, c3 = st.columns(3)
        with c1:
            tile(df.shape[0], "Rows (MOFs)")
        with c2:
            tile(df.shape[1], "Columns")
        with c3:
            tile(df["refcode"].duplicated().sum(), "Duplicate refcodes")
        st.markdown("")

    with st.container(key="card-eda-sample"):
        card_title("Sample rows")
        st.dataframe(df.head(10), width="stretch")

    with st.container(key="card-eda-category"):
        card_title("PLD category distribution")
        category_order = ["nonporous", "small pore", "medium pore", "large pore"]
        counts = df["pld_category"].value_counts().reindex(category_order).fillna(0)
        max_count = counts.max()
        colors = ["", "green", "orange", "purple"]
        for (cat, count), color in zip(counts.items(), colors):
            bar_row(cat, count, max_count, color_class=color)
        st.markdown("")

    if config.dataset == "large":
        with st.container(key="card-eda-pld"):
            card_title("Continuous PLD distribution")
            pld_bins = pd.cut(df["pld_value"], bins=30).value_counts().sort_index()
            pld_bins.index = pld_bins.index.astype(str)
            st.bar_chart(pld_bins)

        with st.container(key="card-eda-metal"):
            card_title(f"Metal-type distribution ({df['metal'].nunique()} unique metals)")
            st.bar_chart(df["metal"].value_counts())

        invalid_smiles = (df["linker_smiles"] == "F[Si](F)(F)(F)(F)F").sum()
        if invalid_smiles:
            note(f"{invalid_smiles} rows had the known invalid SMILES <code>F[Si](F)(F)(F)(F)F</code> (auto-replaced with benzene).", "warning")
    else:
        note("Small dataset has a precomputed category label only — no continuous PLD or metal name available. Switch to the large dataset in the sidebar to see those views.")

# --- Tab 2: Feature Engineering (functional) ---
with tabs[1]:
    with st.container(key="card-feature-summary"):
        st.markdown(pill("Phase 2 — Feature Engineering", "green"), unsafe_allow_html=True)
        card_title(
            "Objective",
            "Build the node feature matrix using the scheme derived from the sidebar's dataset choice.",
        )
        with st.spinner("Building feature matrix..."):
            features, feat_meta = cached_build_features(df, feature_scheme)

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            tile(feature_scheme.capitalize(), "Scheme (auto-selected)")
        with fc2:
            tile(features.shape[1], "Dimensions")
        with fc3:
            tile(feat_meta["invalid_smiles_count"], "Invalid SMILES (fallback used)")
        st.markdown("")

    with st.container(key="card-feature-preview"):
        card_title("Feature matrix preview", "First 5 MOFs × first 12 dimensions")
        preview = pd.DataFrame(
            features[:5, :12],
            index=df["refcode"].values[:5],
            columns=[f"dim_{i}" for i in range(min(12, features.shape[1]))],
        )
        st.dataframe(preview, width="stretch")

    if feature_scheme == "compact":
        note(f"Columns: {', '.join(feat_meta['columns'])}. Scaling: {feat_meta['scaling']}.")
    else:
        note(
            f"1024-bit Morgan fingerprint (radius {feat_meta['morgan_radius']}) + "
            f"{', '.join(feat_meta['pore_geometry_columns'])} + one-hot over "
            f"{feat_meta['n_unique_metals']} real metals (generalized from BlackHole's hardcoded 4-metal map)."
        )
        note(
            f"<strong>Pore Limiting Diameter deliberately excluded</strong> from the feature matrix — "
            f"BlackHole's <code>data_utils.py</code> includes it directly, which is target leakage "
            f"(it's exactly what both the classification label and the regression target are derived from).",
            "warning",
        )

    if feat_meta["invalid_smiles_count"]:
        note(
            f"{feat_meta['invalid_smiles_count']} rows have genuinely truncated/malformed SMILES in the "
            f"source CSV itself (e.g. cut off mid-string) — not a parsing bug on our end. Falls back to a "
            f"deterministic zero vector (BlackHole's code uses unseeded random noise here, which isn't "
            f"reproducible run-to-run)."
        )

# --- Tab 3: Graph Construction (functional) ---
with tabs[2]:
    with st.container(key="card-graph-summary"):
        st.markdown(pill("Phase 3 — Graph Construction", "green"), unsafe_allow_html=True)
        card_title("Objective", "Compare fixed-threshold similarity (φ=0.9) against k-NN (k=3,5,10) graph construction.")
        if config.dataset == "small":
            note("Metal similarity: cosine distance over the 6 metal_feat columns (matches Similarity.py).")
        else:
            note(
                "Metal similarity here is a same/different metal indicator (1.0 if two MOFs share a metal, "
                "else 0.0) — the large dataset has no per-row numeric metal descriptor to run cosine "
                "similarity on like the small dataset does, only a categorical metal name."
            )
        with st.spinner("Building similarity graphs (all 4 configurations)..."):
            graphs = cached_build_graphs(df, config.dataset)
            topology = compute_topology_metrics(graphs)
        best = select_best_graph(topology)

    with st.container(key="card-graph-topology"):
        card_title("Topology comparison", f"Best configuration by rule (lowest isolated rate, then highest modularity): {best}")
        display_topology = topology.copy()
        display_topology["isolated_node_rate"] = (display_topology["isolated_node_rate"] * 100).round(2).astype(str) + "%"
        display_topology["mean_degree"] = display_topology["mean_degree"].round(2)
        display_topology["modularity"] = display_topology["modularity"].round(3)
        st.dataframe(display_topology, width="stretch")
        note(
            f"<strong>{best}</strong> selected to carry forward into Phase 4 (Black Hole) and Phase 5 (GNN training).",
            "warning",
        )

    with st.container(key="card-graph-degrees"):
        card_title("Degree distributions")
        plot_path = plot_degree_distributions(graphs, config.dataset, output_dir="outputs")
        st.image(plot_path)

# --- Tab 4: Black Hole Sparsification (placeholder, but with live-wired controls) ---
with tabs[3]:
    with st.container(key="card-placeholder-3"):
        st.markdown(pill("Phase 4 — Black Hole Sparsification", "orange"), unsafe_allow_html=True)
        card_title(
            "Objective",
            "Apply gravity-based pruning (degree + betweenness + edge-weight-sum) at a configurable threshold τ.",
        )
        note("Confirmed 2026-07-21: gravity weights and pruning threshold are user-configurable, not fixed constants.")

    with st.container(key="card-gravity-weights"):
        card_title("Gravity score weights", "How much each factor contributes to a node's importance score")
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            w_degree = st.slider("Degree centrality (α)", 0.0, 1.0, 0.3, 0.05)
        with gc2:
            w_betweenness = st.slider("Betweenness centrality (β)", 0.0, 1.0, 0.3, 0.05)
        with gc3:
            w_edge_sum = st.slider("Edge-weight-sum (γ)", 0.0, 1.0, 0.4, 0.05)

        config = PipelineConfig(
            dataset=dataset_choice,
            task=task_choice,
            gravity_degree_weight=w_degree,
            gravity_betweenness_weight=w_betweenness,
            gravity_edge_weight_sum_weight=w_edge_sum,
        )
        norm_degree, norm_betweenness, norm_edge_sum = config.gravity_weights_normalized
        st.caption(
            f"Normalized (used internally, always sums to 1): "
            f"α={norm_degree:.2f} · β={norm_betweenness:.2f} · γ={norm_edge_sum:.2f}"
        )

    with st.container(key="card-pruning-threshold"):
        card_title("Pruning threshold (τ)", "Fraction of nodes removed per community")
        tau = st.slider("τ", 0.0, 0.9, 0.3, 0.05)
        config.pruning_threshold = tau
        st.caption("Originally planned reference points: τ=0.3 (BH-30) and τ=0.5 (BH-50) — now freely adjustable.")
        note("These controls are fully wired and validated (see src/config.py), but don't yet affect real output — Phase 3 (graph construction) must exist before gravity scores can be computed over an actual graph.")

# --- Placeholder tabs for phases not yet implemented, with real phase context ---
placeholder_content = {
    4: ("Phase 5 — GNN Training", "orange",
        "Train 2-layer GCN and GraphSAGE on the full graph, BH-30, and BH-50 variants.",
        ["Not yet implemented"]),
    5: ("Phase 6 — Baseline Comparison", "orange",
        "Train Random Forest and k-NN classifier on the flat feature matrix as non-graph baselines.",
        ["Not yet implemented"]),
    6: ("Phase 7 — Results Summary", "orange",
        "Final comparison table, report, slide deck, and thesis-extension outline.",
        ["Not yet implemented"]),
}
for idx, (title, badge, objective, notes) in placeholder_content.items():
    with tabs[idx]:
        with st.container(key=f"card-placeholder-{idx}"):
            st.markdown(pill(title, badge), unsafe_allow_html=True)
            card_title("Objective", objective)
            for n in notes:
                note(n)
