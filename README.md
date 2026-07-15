# MOFQuantumNet

Building on [MOFGalaxyNet](https://github.com/MehrdadJalali-AI/MOFGalaxyNet) and the [Black Hole Strategy](https://github.com/MehrdadJalali-AI/BlackHole) (Jalali et al.), this project modernizes graph-based MOF (Metal-Organic Framework) property prediction using PyTorch Geometric.

Case Study 2 for the Applied Data Science and Analytics program (SRH Hochschule Heidelberg), supervised by Prof. Dr. Mehrdad Jalali. Topic: **Graph Neural Networks for MOF Property Prediction**.

The project reimplements the MOFGalaxyNet/Black Hole pipeline in PyTorch Geometric, runs a controlled comparison of graph construction strategies (fixed similarity threshold vs. k-NN), integrates Black Hole gravity-based sparsification, and benchmarks GNNs (GCN, GraphSAGE) against non-graph baselines (Random Forest, k-NN) for predicting a MOF's Pore Limiting Diameter (PLD) category.

## Status

Development is organized into 7 phases, tracked in [planning/DEVELOPMENT_TODO.md](planning/DEVELOPMENT_TODO.md).

- ✅ **Phase 0 — Environment setup**: `venv` + pip-installed PyTorch Geometric, RDKit, and supporting libraries (no conda). Verified on a MacBook Air M4 (CPU; MPS available as an optional speed-up).
- ✅ **Phase 1 — Data ingestion & EDA**: loaded and validated both candidate datasets; reproduced the original paper's PLD class histogram almost exactly. See `src/data_ingestion.py` and the generated plots in `outputs/`.
- ⬜ Phases 2–7 (feature engineering, graph construction, Black Hole sparsification, GNN training, baseline comparison, reporting) — not yet started.

Full phase-by-phase breakdown, objectives, and dependencies: [planning/DEVELOPMENT_TODO.md](planning/DEVELOPMENT_TODO.md).
Meeting prep, open questions for the professor, and presentation outline: [planning/MEETING_PREP.md](planning/MEETING_PREP.md).

## Setup

No conda — a plain `venv` is used throughout.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements are unpinned (see `requirements.txt`) — PyTorch, PyTorch Geometric, RDKit, pandas, NetworkX, scikit-learn, psutil, tqdm, matplotlib, and Streamlit.

## Usage

Run the Phase 1 data ingestion & EDA script directly:

```bash
.venv/bin/python src/data_ingestion.py
```

Or launch the interactive control panel, which lets you switch between datasets and feature schemes from the sidebar:

```bash
.venv/bin/streamlit run app.py
```

Only the "Data & EDA" tab is functional so far — the remaining tabs are visible placeholders that light up as each phase is implemented.

## Project structure

```
app.py                  Streamlit control panel
src/
  config.py             Single switch point: dataset ("small"/"large") and feature scheme
  data_ingestion.py      Phase 1: loading, cleaning, EDA plots for both datasets
  logging_setup.py       Shared logging configuration
data/raw/                2,000-MOF dataset (from MOFGalaxyNet-main/Data)
outputs/                  Generated plots, logs
planning/
  DEVELOPMENT_TODO.md     Phase-by-phase dev checklist
  MEETING_PREP.md         Professor meeting prep: open questions + slide-deck prompts
reference_content/        Professor's reference papers/code (gitignored, not part of this repo's own codebase)
```

## Two datasets, one interface

Two usable datasets were found in the reference material (contrary to the BlackHole README, which states its data files are "not included" — they are present locally):

- **Small** (`SMILES_METAL_2000_NoPLD.csv`, 2,000 MOFs): compact metal descriptor + SMILES, precomputed PLD category label, no continuous PLD value. Currently the default.
- **Large** (`MOFCSD.csv`, 14,296 MOFs): richer, continuous PLD (enables regression too), real metal names, linker SMILES.

`src/data_ingestion.py::load_dataset(name)` returns both in the same canonical schema, and `src/config.py` is the single place to switch — so this choice (still open with the professor) doesn't require a rewrite either way.
