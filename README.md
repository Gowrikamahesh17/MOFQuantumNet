# MOFQuantumNet

Building on [MOFGalaxyNet](https://github.com/MehrdadJalali-AI/MOFGalaxyNet) and the [Black Hole Strategy](https://github.com/MehrdadJalali-AI/BlackHole) (Jalali et al.), this project modernizes graph-based MOF (Metal-Organic Framework) property prediction using PyTorch Geometric.

Case Study 2 for the Applied Data Science and Analytics program (SRH Hochschule Heidelberg), supervised by Prof. Dr. Mehrdad Jalali. Topic: **Graph Neural Networks for MOF Property Prediction**.

The project reimplements the MOFGalaxyNet/Black Hole pipeline in PyTorch Geometric, runs a controlled comparison of graph construction strategies (fixed similarity threshold vs. k-NN), integrates Black Hole gravity-based sparsification, and benchmarks GNNs (GCN, GraphSAGE) against non-graph baselines (Random Forest, k-NN) for predicting a MOF's Pore Limiting Diameter (PLD).

## Status

Development is organized into 7 phases, tracked in [planning/DEVELOPMENT_TODO.md](planning/DEVELOPMENT_TODO.md).

- ✅ **Phase 0 — Environment setup**: `venv` + pip-installed PyTorch Geometric, RDKit, and supporting libraries (no conda). Verified on a MacBook Air M4 (CPU; MPS available as an optional speed-up).
- ✅ **Phase 1 — Data ingestion & EDA**: loaded and validated both datasets; reproduced the original paper's PLD class histogram almost exactly. `src/data_ingestion.py`.
- ✅ **Phase 2 — Feature engineering**: compact 7-dim (small dataset) and generalized fingerprint 1,079-dim (large dataset) node features. Found and fixed a target-leakage bug in the reference code along the way (see below). `src/feature_engineering.py`.
- ✅ **Phase 3 — Graph construction**: fixed-threshold (φ=0.9) vs. k-NN (k=3,5,10), vectorized for both dataset sizes (~10s for 14,296 nodes). `src/graph_construction.py`.
- 🟡 **Phase 4 — Black Hole sparsification**: UI controls (gravity weights, pruning threshold) built and validated; the actual sparsification computation is the next piece of work.
- ⬜ Phases 5–7 (GNN training, baseline comparison, reporting) — not yet started.

Full phase-by-phase breakdown, objectives, and dependencies: [planning/DEVELOPMENT_TODO.md](planning/DEVELOPMENT_TODO.md).
Meeting prep and presentation materials: [planning/MEETING_PREP.md](planning/MEETING_PREP.md).

### Outcome of the 2026-07-21 meeting with Prof. Jalali

Both datasets stay in scope, selected via one config/UI switch — not an either/or pick. Everything else cascades from that single choice:

| Dataset | Feature scheme | Task |
|---|---|---|
| Small (2,000 MOF) | Compact 7-dim | Classification only (no continuous PLD to regress on) |
| Large (14,296 MOF) | Fingerprint 1,079-dim | Classification **or** regression, user's choice |

Black Hole's gravity weights (α/β/γ) and pruning threshold (τ) are configurable sliders in the app, not fixed constants.

## Setup

No conda — a plain `venv` is used throughout.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements are unpinned (see `requirements.txt`) — PyTorch, PyTorch Geometric, RDKit, pandas, NetworkX, scikit-learn, psutil, tqdm, matplotlib, and Streamlit.

For UI development/verification only (not needed to run the pipeline itself): `pip install -r requirements-dev.txt && playwright install chromium` — see [Interactive control panel](#interactive-control-panel) below.

## Usage

Each phase can be run standalone as a script, or explored interactively.

```bash
.venv/bin/python src/data_ingestion.py        # Phase 1: EDA, plots to outputs/
.venv/bin/python src/feature_engineering.py   # Phase 2: builds + saves feature matrices to data/processed/
.venv/bin/python src/graph_construction.py    # Phase 3: builds all 4 graphs, saves edge lists + topology table
```

### Interactive control panel

```bash
.venv/bin/streamlit run app.py
```

A dark, tabbed control panel — one tab per phase. The sidebar lets you switch dataset (small/large) and, for the large dataset, task (classification/regression); everything cascades from there. Tabs 1–3 (Data & EDA, Feature Engineering, Graph Construction) are fully functional with live computed output; Tab 4 (Black Hole) has working, validated sliders not yet wired to a computation; the rest are marked "not yet implemented" so the app never overstates what's done.

To verify UI changes visually (Streamlit renders client-side, so `curl` only returns an empty shell): `.claude/skills/developing-with-streamlit/capture_screenshots.py` launches the app headlessly with Playwright and screenshots every tab — see that skill's `SKILL.md` for details.

## Project structure

```
app.py                    Streamlit control panel
src/
  config.py               PipelineConfig — dataset is the one root switch; feature_scheme
                           is derived from it; task, gravity weights, and pruning threshold
                           are the other configurable fields
  data_ingestion.py        Phase 1: loading, cleaning, EDA plots for both datasets
  feature_engineering.py   Phase 2: compact / fingerprint node feature schemes
  graph_construction.py    Phase 3: threshold vs. k-NN graph construction + topology metrics
  ui_theme.py              Dark "glassmorphism" theme + reusable Streamlit components
  logging_setup.py         Shared logging configuration
data/
  raw/                     Input datasets (from MOFGalaxyNet-main/Data, BlackHole-main)
  processed/               Saved feature matrices, graph edge lists, topology tables
outputs/                   Generated plots, logs, UI screenshots
planning/
  DEVELOPMENT_TODO.md      Phase-by-phase dev checklist, findings, and decisions log
  MEETING_PREP.md          Professor meeting prep: open questions + slide-deck prompts
.claude/skills/
  developing-with-streamlit/  Streamlit best-practices skill, incl. headless screenshot verification
reference_content/         Professor's reference papers/code (gitignored, not part of this repo's own codebase)
```

## Two datasets, one interface

Two usable datasets were found in the reference material (contrary to the BlackHole README, which states its data files are "not included" — they are present locally):

- **Small** (`SMILES_METAL_2000_NoPLD.csv`, 2,000 MOFs): 6 anonymous numeric metal features + SMILES + precomputed PLD category label. No continuous PLD, no metal name.
- **Large** (`MOFCSD.csv`, 14,296 MOFs): continuous PLD, real metal names (53 unique), linker SMILES, pore geometry.

`src/data_ingestion.py::load_dataset(name)` returns both in the same canonical schema; `src/config.py` is the single switch point for dataset (and everything that cascades from it).

## Notable findings along the way

- **Target leakage in the reference code**: `BlackHole/data_utils.py` includes the raw Pore Limiting Diameter value as an input feature — but that's exactly what both the classification label and the regression target are derived from. Fixed in `src/feature_engineering.py` by excluding it regardless of task.
- **Metal one-hot encoding gap**: the reference code hardcodes a 4-slot map (Cu/Zn/Fe/Co), silently mis-encoding any other metal as Cu. The real data has 53 unique metals — generalized to a full dynamic one-hot.
- **k-NN vs. fixed threshold**: reproduces the original paper's own noted limitation directly — the φ=0.9 threshold leaves 20.5% (small) / 11.3% (large) of nodes isolated, while every k-NN configuration guarantees 0% isolated nodes by construction.
- **Non-reproducible fallback**: the reference code's invalid-SMILES handling uses unseeded random noise; replaced with a deterministic zero vector.
