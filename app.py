"""Interactive control panel for the MOFQuantumNet pipeline.

Run with: .venv/bin/streamlit run app.py

Each tab maps to one pipeline stage from README.md's "Pipeline overview".

Dataset selection cascades into feature scheme (derived, not independently
chosen) and task options (only the large dataset offers classification vs.
regression); Black Hole's gravity weights and pruning threshold are
configurable sliders, not fixed constants.
"""

import os
import sys

import networkx as nx
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from baseline_models import build_master_comparison_table, run_baseline_comparison  # noqa: E402
from black_hole_sparsification import apply_black_hole_sparsification  # noqa: E402
from config import DATASET_CHOICES, TASK_CHOICES, PipelineConfig  # noqa: E402
from data_ingestion import load_dataset  # noqa: E402
from feature_engineering import build_features  # noqa: E402
from gnn_training import plot_loss_curves, run_gnn_comparison  # noqa: E402
from graph_construction import (  # noqa: E402
    build_similarity_graphs,
    compute_topology_metrics,
    plot_degree_distributions,
    sample_subgraph_for_viz,
    select_best_graph,
)
from graph_viz import render_interactive_graph  # noqa: E402
from ui_theme import bar_row, card_title, eyebrow, inject_theme, note, pill, tile  # noqa: E402

GRAPH_VIZ_MAX_NODES = 90

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


@st.cache_data(show_spinner=False)
def cached_black_hole(df: pd.DataFrame, dataset: str, best_graph_name: str, weights: tuple, threshold: float):
    graphs = cached_build_graphs(df, dataset)
    return apply_black_hole_sparsification(graphs[best_graph_name], df, weights, threshold)


# A readable node-link drawing needs a small, connected sample, not the full graph
# (2,000-14,296 nodes). Layout is computed once on that sample and cached so the
# "before" (Tab 3) and "after" (Tab 4) interactive views land on identical
# coordinates -- a node that disappears after pruning visibly vanishes from the
# same spot rather than the whole picture re-shuffling.
@st.cache_data(show_spinner=False)
def cached_graph_sample_layout(df: pd.DataFrame, dataset: str, best_graph_name: str, max_nodes: int = GRAPH_VIZ_MAX_NODES):
    graphs = cached_build_graphs(df, dataset)
    graph = graphs[best_graph_name]
    node_ids = sample_subgraph_for_viz(graph, max_nodes=max_nodes)
    sub = graph.subgraph(node_ids)
    pos = nx.spring_layout(sub, seed=42, weight="weight")
    return node_ids, pos


# Phase 5 takes ~35-40s (small dataset) but ~9 MINUTES (large dataset — two Black Hole
# runs plus 9 full training loops: GCN + GraphSAGE + GAT, the last of which is
# noticeably heavier per epoch due to 12-head attention). Every tab's body runs on every
# Streamlit rerun, so without a button gate, simply moving an unrelated slider on the
# large dataset would block the entire app for minutes. st.cache_data alone isn't enough
# — it only helps on a *repeated* identical call, not the very first one, which would
# still run unconditionally.
@st.cache_data(show_spinner=False)
def cached_gnn_comparison(df: pd.DataFrame, dataset: str, best_graph_name: str, task: str, weights: tuple):
    graphs = cached_build_graphs(df, dataset)
    features, _ = cached_build_features(df, "compact" if dataset == "small" else "fingerprint")
    return run_gnn_comparison(df, features, graphs[best_graph_name], task, weights)


# Phase 6 is fast (~2s incremental, on top of already-cached features/graph/BH) —
# no button gate needed, unlike Phase 5.
@st.cache_data(show_spinner=False)
def cached_baseline_comparison(df: pd.DataFrame, dataset: str, best_graph_name: str, task: str, weights: tuple):
    graphs = cached_build_graphs(df, dataset)
    features, _ = cached_build_features(df, "compact" if dataset == "small" else "fingerprint")
    bh_result = cached_black_hole(df, dataset, best_graph_name, weights, 0.3)
    return run_baseline_comparison(df, features, task, bh_result["fixed_test_nodes"])

# --- Sidebar: the root switch (dataset) + what cascades from it ---
st.sidebar.markdown("### Pipeline configuration")
dataset_choice = st.sidebar.radio(
    "Dataset",
    DATASET_CHOICES,
    format_func=lambda d: "Small — 2,000 MOF (MOFGalaxyNet)" if d == "small" else "Large — 14,296 MOF (BlackHole/MOFCSD)",
    help="Both datasets stay in scope — this is the one root switch everything else cascades from.",
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
        tile("6.5 / 7", "Phases complete")
    with h5:
        n_metals = df["metal"].nunique() if df["metal"].notna().any() else "N/A"
        tile(n_metals, "Unique metals")

tabs = st.tabs([
    "1. Data & EDA ✅",
    "2. Feature Engineering ✅",
    "3. Graph Construction ✅",
    "4. Black Hole Sparsification ✅",
    "5. GNN Training ✅",
    "6. Baseline Comparison ✅",
    "7. Results Summary ✅",
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

    with st.container(key="card-graph-interactive"):
        card_title(
            "Interactive graph view",
            f"A readable, connected sample of the selected graph ({best}) — drag nodes, scroll to zoom, drag the "
            "background to pan, hover a node for its refcode/category. Too many nodes to draw all "
            f"{graphs[best].number_of_nodes():,} at once, so this is a {GRAPH_VIZ_MAX_NODES}-node neighborhood sample.",
        )
        sample_node_ids, sample_pos = cached_graph_sample_layout(df, config.dataset, best)
        render_interactive_graph(
            graphs[best],
            sample_node_ids,
            sample_pos,
            df["refcode"].values,
            df["pld_category"].values,
            key=f"graph-viz-before-{config.dataset}-{best}",
            subtitle=f"{len(sample_node_ids)} nodes sampled from {graphs[best].number_of_nodes():,} total",
        )

# --- Tab 4: Black Hole Sparsification (functional) ---
with tabs[3]:
    with st.container(key="card-placeholder-3"):
        st.markdown(pill("Phase 4 — Black Hole Sparsification", "green"), unsafe_allow_html=True)
        card_title(
            "Objective",
            "Apply gravity-based pruning (degree + betweenness + edge-weight-sum) at a configurable threshold τ.",
        )
        note("Gravity weights and pruning threshold are user-configurable, not fixed constants.")

    with st.container(key="card-gravity-weights"):
        card_title(
            "Gravity score weights",
            "How much each factor contributes to a node's importance score. Default 0.33/0.33/0.33 "
            "matches the paper's own main configuration (BlackHole.pdf, Section 2.3) — the code's "
            "hardcoded default of 0.3/0.3/0.4 doesn't actually match what the paper reports using.",
        )

        def _weight_control(label: str, key_prefix: str, default: float) -> float:
            """Slider + number input, kept fully independent — no auto-rebalancing between the
            three weights. Setting one to 0.50 leaves the other two exactly where they were, so
            you can freely fix any two values (e.g. 0.50 and 0.15) and see the third stay put too."""
            slider_key, number_key = f"{key_prefix}_slider", f"{key_prefix}_number"
            if slider_key not in st.session_state:
                st.session_state[slider_key] = default
                st.session_state[number_key] = default

            def _from_slider():
                st.session_state[number_key] = st.session_state[slider_key]

            def _from_number():
                v = max(0.0, min(1.0, st.session_state[number_key]))
                st.session_state[number_key] = v
                st.session_state[slider_key] = v

            sc, nc = st.columns([3, 1])
            with sc:
                st.slider(label, 0.0, 1.0, step=0.01, key=slider_key, on_change=_from_slider)
            with nc:
                st.number_input(label, 0.0, 1.0, step=0.01, key=number_key, on_change=_from_number,
                                 label_visibility="collapsed")
            return st.session_state[slider_key]

        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            w_degree = _weight_control("Degree centrality (α)", "gravity_alpha", 0.33)
        with gc2:
            w_betweenness = _weight_control("Betweenness centrality (β)", "gravity_beta", 0.33)
        with gc3:
            w_edge_sum = _weight_control("Edge-weight-sum (γ)", "gravity_gamma", 0.33)

        config = PipelineConfig(
            dataset=dataset_choice,
            task=task_choice,
            gravity_degree_weight=w_degree,
            gravity_betweenness_weight=w_betweenness,
            gravity_edge_weight_sum_weight=w_edge_sum,
        )
        raw_sum = w_degree + w_betweenness + w_edge_sum
        norm_a, norm_b, norm_g = config.gravity_weights_normalized
        if abs(raw_sum - 1.0) < 0.005:
            st.caption(f"α={w_degree:.2f} + β={w_betweenness:.2f} + γ={w_edge_sum:.2f} = {raw_sum:.2f} ✓")
        else:
            st.caption(
                f"α={w_degree:.2f} + β={w_betweenness:.2f} + γ={w_edge_sum:.2f} = {raw_sum:.2f} — "
                f"doesn't need to be exactly 1, it's normalized before use: "
                f"α={norm_a:.2f} · β={norm_b:.2f} · γ={norm_g:.2f}"
            )

    with st.container(key="card-pruning-threshold"):
        card_title("Pruning threshold (τ)", "Fraction of nodes removed per community")
        tau = st.slider("τ", 0.0, 0.9, 0.3, 0.05)
        config.pruning_threshold = tau
        st.caption("Originally planned reference points: τ=0.3 (BH-30) and τ=0.5 (BH-50) — now freely adjustable.")

    with st.container(key="card-bh-results"):
        card_title("Sparsification result", f"Applied to Phase 3's selected graph ({best}) at the settings above")
        with st.spinner("Running gravity computation + pruning..."):
            bh_result = cached_black_hole(
                df, config.dataset, best,
                config.gravity_weights_normalized, config.pruning_threshold,
            )
        m = bh_result["metrics"]

        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            tile(f"{m['nodes_after']:,} / {m['nodes_before']:,}", f"Nodes retained ({m['node_retention_pct']}%)")
        with bc2:
            tile(f"{m['edges_after']:,} / {m['edges_before']:,}", f"Edges retained ({m['edge_retention_pct']}%)")
        with bc3:
            tile(m["isolated_nodes_after"], "Isolated nodes after pruning")
        with bc4:
            tile(m["num_fixed_test_nodes"], "Fixed test nodes (always kept)")
        st.markdown("")

        bc5, bc6, bc7 = st.columns(3)
        with bc5:
            tile(f"{m['density_before']:.5f} → {m['density_after']:.5f}", "Graph density")
        with bc6:
            tile(m["num_communities"], "Louvain communities")
        with bc7:
            tile(f"{m['elapsed_seconds']}s", "Computation time")

        note(
            f"Peak memory during this run: {m['peak_memory_mb']:.0f} MB. This is a whole-process RSS snapshot "
            f"(includes PyTorch, RDKit, everything already loaded), not an isolated measurement of just this "
            f"step — <strong>not directly comparable</strong> to the ~114 MB the Black Hole paper reports for "
            f"its own 50%-pruning run, which was measured differently.",
            "warning",
        )
        note(
            "Isolated nodes above are a natural consequence of the original algorithm: it guarantees fixed "
            "test nodes keep at least one edge, but not every retained non-test node — a node can survive "
            "pruning and still lose all its edges if its neighbors were pruned. Not a bug, just how the "
            "reference algorithm works."
        )

    with st.container(key="card-bh-graph-compare"):
        card_title(
            "Before vs. after pruning",
            "Same node-sample, same layout, both sides — a node's position doesn't move between the two; "
            "it either survives or disappears.",
        )
        sample_node_ids, sample_pos = cached_graph_sample_layout(df, config.dataset, best)
        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown("**Before pruning**")
            render_interactive_graph(
                graphs[best],
                sample_node_ids,
                sample_pos,
                df["refcode"].values,
                df["pld_category"].values,
                key=f"graph-viz-bh-before-{config.dataset}-{best}",
                subtitle=f"{len(sample_node_ids)} nodes",
                height=420,
            )
        with gc2:
            st.markdown(f"**After pruning (τ={tau})**")
            after_ids = [n for n in sample_node_ids if n in bh_result["graph"].nodes()]
            render_interactive_graph(
                bh_result["graph"],
                sample_node_ids,
                sample_pos,
                df["refcode"].values,
                df["pld_category"].values,
                key=f"graph-viz-bh-after-{config.dataset}-{best}-{tau}",
                subtitle=f"{len(after_ids)} of {len(sample_node_ids)} sampled nodes survived",
                height=420,
            )

# --- Tab 5: GNN Training (functional, but button-gated — see cached_gnn_comparison above) ---
with tabs[4]:
    with st.container(key="card-gnn-objective"):
        st.markdown(pill("Phase 5 — GNN Training", "green"), unsafe_allow_html=True)
        card_title(
            "Objective",
            "Train GCN, GraphSAGE, and GAT on the Phase-3 best graph, BH-30 (τ=0.3), and BH-50 (τ=0.5) — 9 runs total.",
        )
        note(
            "<strong>Real bug found and fixed:</strong> the reference code's <code>train()</code> sets "
            "<code>val_mask = data.test_mask</code> — early stopping is driven by the same set used for "
            "final evaluation, which is test-set leakage. Fixed with a genuine 3-way split: Phase 4's fixed "
            "test nodes stay untouched until final evaluation; a separate validation split (carved out of "
            "the remaining nodes) drives early stopping instead.",
            "warning",
        )
        est_time = "~35-40s" if config.dataset == "small" else "~9 minutes"
        run_clicked = st.button(f"Run GNN training comparison ({est_time})", key="run_gnn")

    result_key = (config.dataset, best, config.task, config.gravity_weights_normalized)
    if run_clicked:
        with st.spinner(f"Training GCN + GraphSAGE on 3 graph variants ({est_time})..."):
            st.session_state["gnn_results"] = cached_gnn_comparison(
                df, config.dataset, best, config.task, config.gravity_weights_normalized
            )
            st.session_state["gnn_results_key"] = result_key

    if st.session_state.get("gnn_results_key") == result_key:
        gnn_results = st.session_state["gnn_results"]

        with st.container(key="card-gnn-table"):
            card_title("Results comparison")
            rows = []
            for (variant, model_name), r in gnn_results.items():
                row = {"Graph variant": variant, "Model": model_name}
                row.update(r["metrics"] if config.task == "regression" else
                            {k: v for k, v in r["metrics"].items() if k != "confusion_matrix"})
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), width="stretch")

        with st.container(key="card-gnn-loss"):
            card_title("Training / validation loss curves")
            plot_path = plot_loss_curves(gnn_results, config.dataset, output_dir="outputs")
            st.image(plot_path)
    else:
        note("Click the button above to run this phase — results aren't recomputed automatically since training takes real time.")

# --- Tab 6: Baseline Comparison (functional) ---
with tabs[5]:
    with st.container(key="card-baseline-objective"):
        st.markdown(pill("Phase 6 — Baseline Comparison", "green"), unsafe_allow_html=True)
        card_title(
            "Objective",
            "Train Random Forest and k-NN on the flat feature matrix — no graph structure at all — "
            "evaluated on the exact same fixed test nodes as the GNNs, for a genuine apples-to-apples comparison.",
        )
        with st.spinner("Training baselines..."):
            baseline_results = cached_baseline_comparison(df, config.dataset, best, config.task, config.gravity_weights_normalized)

        cols = st.columns(len(baseline_results))
        for col, (name, r) in zip(cols, baseline_results.items()):
            with col:
                m = r["metrics"]
                if config.task == "classification":
                    tile(f"{m['accuracy']:.3f}", f"{name} — accuracy (κ={m['cohen_kappa']:.2f})")
                else:
                    tile(f"R²={m['r2']:.3f}", f"{name} — MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}")

    with st.container(key="card-baseline-master-table"):
        card_title("Master comparison — GNNs vs. baselines")
        gnn_key = (config.dataset, best, config.task, config.gravity_weights_normalized)
        if st.session_state.get("gnn_results_key") == gnn_key:
            master_table = build_master_comparison_table(st.session_state["gnn_results"], baseline_results, config.task)
            st.dataframe(master_table, width="stretch")

            top_config = master_table.iloc[0]["Configuration"]
            top_type = master_table.iloc[0]["Type"]
            if top_type == "Baseline":
                note(
                    f"<strong>The best result here is a non-graph baseline</strong> ({top_config}), not a GNN. "
                    f"Consistent across both classification and regression on the large dataset: Random Forest "
                    f"reaches R²=0.89 (MAE=0.53 Å) vs. the GNNs' best of R²=0.23 (MAE=1.64 Å). Likely explanation: "
                    f"the 2 pore-geometry features (Largest Cavity Diameter, Largest Free Sphere) are strongly "
                    f"correlated with the PLD target, and tree ensembles exploit that directly — while the graph "
                    f"is built from the *same* linker/metal similarity already encoded in the node features, so "
                    f"message passing mostly re-derives information the model already has rather than adding new "
                    f"signal. A real, honest finding, not a bug — and a natural thesis-extension question (would "
                    f"a graph built from information *not* already in the node features let GNNs add value?).",
                    "warning",
                )
            else:
                note(f"Best result: {top_config} ({top_type}).")
        else:
            note("Run Phase 5 (GNN Training tab) first to see the full master comparison table including GNN results — showing baselines only for now.")
            baseline_only_table = build_master_comparison_table({}, baseline_results, config.task)
            st.dataframe(baseline_only_table, width="stretch")

# --- Tab 7: Results Summary ---
with tabs[6]:
    with st.container(key="card-summary-objective"):
        st.markdown(pill("Phase 7 — Analysis & Reporting", "green"), unsafe_allow_html=True)
        card_title("Objective", "Final comparison table, written report, and thesis-extension outline.")
        note(
            "Full report (background, methodology, results, discussion, limitations, thesis-extension "
            "directions): <code>planning/CASE_STUDY_REPORT.md</code>. Every number in it is cited from "
            "saved artifacts in <code>data/processed/</code>, not re-derived from memory."
        )

    with st.container(key="card-summary-tradeoff"):
        card_title(
            "Key finding 1 — topology-only graph selection doesn't always pick the best-performing graph",
            "Phase 3 selected knn_3 by connectivity/modularity alone. Phase 7 tested downstream accuracy for all 4 candidates.",
        )
        note(
            "On the small dataset, knn_3 genuinely is best. On the <strong>large dataset, knn_5 and knn_10 "
            "both beat it</strong> on accuracy and Cohen's κ — reproduced across two separate runs. "
            "See <code>data/processed/graph_tradeoff_*.csv</code>."
        )

    with st.container(key="card-summary-baseline"):
        card_title(
            "Key finding 2 — non-graph baselines beat every GNN configuration",
            "The single most important result of this case study.",
        )
        note(
            "Random Forest reaches <strong>R²=0.89 (MAE=0.53 Å)</strong> on regression vs. the best GNN's "
            "R²=0.23 (MAE=1.64 Å) — and leads on classification accuracy too, on both datasets. Likely cause: "
            "the graph is built from the same linker/metal similarity already present in the node features, "
            "so message passing mostly re-derives information the model already has. See Phase 6 tab and "
            "<code>planning/CASE_STUDY_REPORT.md</code> Section 4.4 for the full reasoning.",
            "warning",
        )

    with st.container(key="card-summary-deferred"):
        card_title("Not yet included")
        note(
            "<strong>Notebook consolidation</strong> — the pipeline is script-based throughout "
            "(<code>src/*.py</code>, no Jupyter dependency). A single consolidated notebook covering "
            "the full pipeline hasn't been built."
        )
        note(
            "<strong>Results slide deck</strong> — a presentation-ready summary of the results in "
            "<code>planning/CASE_STUDY_REPORT.md</code> hasn't been built."
        )
