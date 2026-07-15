"""Interactive control panel for the MOFQuantumNet pipeline.

Run with: .venv/bin/streamlit run app.py

Each tab maps to one development phase from planning/DEVELOPMENT_TODO.md.
Only Phase 1 (Data & EDA) is functional so far; later tabs are placeholders
that light up as each phase is implemented, so this app always reflects the
pipeline's real current state rather than promising features that don't exist.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import DATASET_CHOICES, FEATURE_SCHEME_CHOICES, PipelineConfig  # noqa: E402
from data_ingestion import derive_pld_category, load_dataset  # noqa: E402

st.set_page_config(page_title="MOFQuantumNet Pipeline", layout="wide")

st.title("MOFQuantumNet — Pipeline Control Panel")
st.caption("Graph Neural Networks for MOF Property Prediction — Case Study 2")

# --- Sidebar: the two switches pending professor confirmation ---
st.sidebar.header("Pipeline configuration")
dataset_choice = st.sidebar.radio(
    "Dataset",
    DATASET_CHOICES,
    format_func=lambda d: "Small (2,000 MOF, MOFGalaxyNet)" if d == "small" else "Large (14,296 MOF, BlackHole/MOFCSD)",
    help="Open question for Prof. Jalali — see planning/MEETING_PREP.md, Part 1.B",
)
feature_scheme_choice = st.sidebar.radio(
    "Feature vector scheme",
    FEATURE_SCHEME_CHOICES,
    format_func=lambda f: "Compact 7-dim (paper-original)" if f == "compact" else "Fingerprint 1031-dim (BlackHole code)",
    help="Tied to the dataset choice above — not yet wired into Phase 2 (unbuilt)",
)
config = PipelineConfig(dataset=dataset_choice, feature_scheme=feature_scheme_choice)
st.sidebar.info("Switching here only affects the tabs marked ✅ below — later phases pick this up as they're built.")

tabs = st.tabs([
    "1. Data & EDA ✅",
    "2. Feature Engineering",
    "3. Graph Construction",
    "4. Black Hole Sparsification",
    "5. GNN Training",
    "6. Baseline Comparison",
    "7. Results Summary",
])

# --- Tab 1: Data & EDA (functional) ---
with tabs[0]:
    st.subheader(f"Dataset: {config.dataset}")

    with st.spinner("Loading dataset..."):
        df = load_dataset(config.dataset)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows (MOFs)", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Unique metals", df["metal"].nunique() if df["metal"].notna().any() else "N/A (small dataset)")

    st.markdown("**Sample rows**")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("**PLD category distribution**")
    category_order = ["nonporous", "small pore", "medium pore", "large pore"]
    counts = df["pld_category"].value_counts().reindex(category_order).fillna(0)
    st.bar_chart(counts)

    if config.dataset == "large":
        st.markdown("**Continuous PLD distribution**")
        st.bar_chart(pd.cut(df["pld_value"], bins=30).value_counts().sort_index())

        st.markdown("**Metal-type distribution**")
        st.bar_chart(df["metal"].value_counts())

        invalid_smiles = (df["linker_smiles"] == "F[Si](F)(F)(F)(F)F").sum()
        if invalid_smiles:
            st.warning(f"{invalid_smiles} rows had the known invalid SMILES `F[Si](F)(F)(F)(F)F` (auto-replaced with benzene).")
    else:
        st.caption("Small dataset has a precomputed category label only — no continuous PLD or metal name available.")

# --- Placeholder tabs for phases not yet implemented ---
placeholders = {
    tabs[1]: ("Phase 2 — Feature Engineering", "Blocked on the feature-vector scheme decision (compact vs. fingerprint)."),
    tabs[2]: ("Phase 3 — Graph Construction", "Threshold (φ=0.9) vs. k-NN (k=3,5,10) comparison — not yet implemented."),
    tabs[3]: ("Phase 4 — Black Hole Sparsification", "Gravity-based pruning at τ=0.3/0.5 — not yet implemented."),
    tabs[4]: ("Phase 5 — GNN Training", "GCN / GraphSAGE training and metrics — not yet implemented."),
    tabs[5]: ("Phase 6 — Baseline Comparison", "Random Forest / k-NN classifier baselines — not yet implemented."),
    tabs[6]: ("Phase 7 — Results Summary", "Final comparison table and report artifacts — not yet implemented."),
}
for tab, (title, note) in placeholders.items():
    with tab:
        st.subheader(title)
        st.info(f"Not yet implemented. {note}")
