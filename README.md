# MOFQuantumNet

Building on [MOFGalaxyNet](https://github.com/MehrdadJalali-AI/MOFGalaxyNet) and the [Black Hole Strategy](https://github.com/MehrdadJalali-AI/BlackHole) (Jalali et al.), this project modernizes graph-based MOF (Metal-Organic Framework) property prediction using PyTorch Geometric.

Case Study 2 for the Applied Data Science and Analytics program (SRH Hochschule Heidelberg), supervised by Prof. Dr. Mehrdad Jalali. Topic: **Graph Neural Networks for MOF Property Prediction**.

The project reimplements the MOFGalaxyNet/Black Hole pipeline in PyTorch Geometric, runs a controlled comparison of graph construction strategies (fixed similarity threshold vs. k-NN), integrates Black Hole gravity-based sparsification, and benchmarks GNNs (GCN, GraphSAGE, GAT) against non-graph baselines (Random Forest, k-NN) for predicting a MOF's Pore Limiting Diameter (PLD).

## Status

Development is organized into 7 phases, tracked in [planning/DEVELOPMENT_TODO.md](planning/DEVELOPMENT_TODO.md).

- ✅ **Phase 0 — Environment setup**: `venv` + pip-installed PyTorch Geometric, RDKit, and supporting libraries (no conda). Verified on a MacBook Air M4 (CPU; MPS available as an optional speed-up).
- ✅ **Phase 1 — Data ingestion & EDA**: loaded and validated both datasets; reproduced the original paper's PLD class histogram almost exactly. `src/data_ingestion.py`.
- ✅ **Phase 2 — Feature engineering**: compact 7-dim (small dataset) and generalized fingerprint 1,079-dim (large dataset) node features. Found and fixed a target-leakage bug in the reference code along the way. `src/feature_engineering.py`.
- ✅ **Phase 3 — Graph construction**: fixed-threshold (φ=0.9) vs. k-NN (k=3,5,10), vectorized for both dataset sizes (~10s for 14,296 nodes). `src/graph_construction.py`.
- ✅ **Phase 4 — Black Hole sparsification**: gravity-based pruning, fully wired to live UI sliders (weights + threshold). `src/black_hole_sparsification.py`.
- ✅ **Phase 5 — GNN training**: GCN, GraphSAGE, and GAT trained on the full graph, BH-30, and BH-50 (9 runs). Found and fixed a test-set leakage bug in the reference training loop. `src/gnn_training.py`.
- ⬜ Phases 6–7 (baseline comparison, reporting) — not yet started.

Full phase-by-phase breakdown, objectives, dependencies, and a running log of every finding/decision: [planning/DEVELOPMENT_TODO.md](planning/DEVELOPMENT_TODO.md).
Meeting prep and presentation materials: [planning/MEETING_PREP.md](planning/MEETING_PREP.md).

### Outcome of the 2026-07-21 meeting with Prof. Jalali

Both datasets stay in scope, selected via one config/UI switch — not an either/or pick. Everything else cascades from that single choice:

| Dataset | Feature scheme | Task |
|---|---|---|
| Small (2,000 MOF) | Compact 7-dim | Classification only (no continuous PLD to regress on) |
| Large (14,296 MOF) | Fingerprint 1,079-dim | Classification **or** regression, user's choice |

Black Hole's gravity weights (α/β/γ) and pruning threshold (τ) are configurable in the app — independent slider + number-input pairs (no auto-rebalancing, so any two values can be fixed manually), default **0.33/0.33/0.33** matching the actual Black Hole paper's stated configuration (not the reference code's own default of 0.3/0.3/0.4 — verified by reading `BlackHole.pdf` directly, not just secondary sources).

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
.venv/bin/python src/data_ingestion.py             # Phase 1: EDA, plots to outputs/
.venv/bin/python src/feature_engineering.py        # Phase 2: builds + saves feature matrices to data/processed/
.venv/bin/python src/graph_construction.py         # Phase 3: builds all 4 graphs, saves edge lists + topology table
.venv/bin/python src/black_hole_sparsification.py  # Phase 4: sparsifies the best graph at tau=0.3 and tau=0.5
.venv/bin/python src/gnn_training.py                # Phase 5: trains GCN/GraphSAGE/GAT on all 3 graph variants
```

### Interactive control panel

```bash
.venv/bin/streamlit run app.py
```

A dark, tabbed control panel — one tab per phase. The sidebar lets you switch dataset (small/large) and, for the large dataset, task (classification/regression); everything cascades from there. Tabs 1–4 (Data & EDA, Feature Engineering, Graph Construction, Black Hole Sparsification) run live on every interaction; Tab 5 (GNN Training) is button-gated instead — training takes ~35-40s (small dataset) to ~9 minutes (large dataset), and since every tab's body runs on every Streamlit rerun, it would otherwise block the entire app on any unrelated slider tweak. The rest are marked "not yet implemented" so the app never overstates what's done.

To verify UI changes visually (Streamlit renders client-side, so `curl` only returns an empty shell): `.claude/skills/developing-with-streamlit/capture_screenshots.py` launches the app headlessly with Playwright and screenshots every tab — see that skill's `SKILL.md` for details, including several Streamlit/Playwright gotchas discovered along the way.

## Project structure

```
app.py                          Streamlit control panel
src/
  config.py                     PipelineConfig — dataset is the one root switch; feature_scheme
                                 is derived from it; task, gravity weights, and pruning threshold
                                 are the other configurable fields
  data_ingestion.py              Phase 1: loading, cleaning, EDA plots for both datasets
  feature_engineering.py         Phase 2: compact / fingerprint node feature schemes
  graph_construction.py          Phase 3: threshold vs. k-NN graph construction + topology metrics
  black_hole_sparsification.py   Phase 4: gravity scoring + PLD-stratified node/edge pruning
  gnn_training.py                Phase 5: GCN / GraphSAGE / GAT + train/eval with a proper 3-way split
  ui_theme.py                    Dark "glassmorphism" theme + reusable Streamlit components
  logging_setup.py               Shared logging configuration
data/
  raw/                           Input datasets (from MOFGalaxyNet-main/Data, BlackHole-main)
  processed/                     Saved feature matrices, graph edge lists, topology tables, BH results
outputs/                         Generated plots, logs, UI screenshots
planning/
  DEVELOPMENT_TODO.md            Phase-by-phase dev checklist, findings, and decisions log
  MEETING_PREP.md                Professor meeting prep: open questions + slide-deck prompts
.claude/skills/
  developing-with-streamlit/     Streamlit best-practices skill, incl. headless screenshot verification
reference_content/               Professor's reference papers/code (gitignored, not part of this repo's own codebase)
```

## Two datasets, one interface

Two usable datasets were found in the reference material (contrary to the BlackHole README, which states its data files are "not included" — they are present locally):

- **Small** (`SMILES_METAL_2000_NoPLD.csv`, 2,000 MOFs): 6 anonymous numeric metal features + SMILES + precomputed PLD category label. No continuous PLD, no metal name.
- **Large** (`MOFCSD.csv`, 14,296 MOFs): continuous PLD, real metal names (53 unique), linker SMILES, pore geometry.

`src/data_ingestion.py::load_dataset(name)` returns both in the same canonical schema; `src/config.py` is the single switch point for dataset (and everything that cascades from it).

## Notable findings along the way

- **Target leakage in the reference code (Phase 2)**: `BlackHole/data_utils.py` includes the raw Pore Limiting Diameter value as an input feature — but that's exactly what both the classification label and the regression target are derived from. Fixed in `src/feature_engineering.py` by excluding it regardless of task.
- **Test-set leakage in the reference training loop (Phase 5)**: `graphsage_model.py`'s `train()` sets `val_mask = data.test_mask` — early stopping is driven by the same set used for final evaluation. Fixed with a genuine 3-way split: Phase 4's fixed test nodes stay untouched until final evaluation; a separate validation split drives early stopping instead.
- **Metal one-hot encoding gap**: the reference code hardcodes a 4-slot map (Cu/Zn/Fe/Co), silently mis-encoding any other metal as Cu. The real data has 53 unique metals — generalized to a full dynamic one-hot.
- **Gravity weight default resolved by reading the actual paper**: the Expose said 0.33/0.33/0.33, the code defaults to 0.3/0.3/0.4 — reading `BlackHole.pdf` directly (Section 2.3) confirmed the paper's own main configuration is equal weighting; the code's default doesn't match its own paper.
- **k-NN vs. fixed threshold**: reproduces the original paper's own noted limitation directly — the φ=0.9 threshold leaves 20.5% (small) / 11.3% (large) of nodes isolated, while every k-NN configuration guarantees 0% isolated nodes by construction.
- **Non-reproducible fallback**: the reference code's invalid-SMILES handling uses unseeded random noise; replaced with a deterministic zero vector.
- **Our accuracy is lower than the paper's reported 0.783 — deliberately, not a regression**: removing the two leakage bugs above makes the task genuinely harder. We also evaluate on the full test set (the paper restricts evaluation to each graph's largest connected component) and run once instead of the paper's 10-run average. All differences are documented in `DEVELOPMENT_TODO.md`.
