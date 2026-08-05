# MOFQuantumNet

Graph Neural Networks for predicting a Metal-Organic Framework's (MOF) Pore Limiting
Diameter (PLD) — the width of the narrowest constriction a molecule must pass through to
enter the pore. Builds on [MOFGalaxyNet](https://github.com/MehrdadJalali-AI/MOFGalaxyNet)
and the [Black Hole Strategy](https://github.com/MehrdadJalali-AI/BlackHole), reimplemented
in PyTorch Geometric with a generalized, dataset-agnostic pipeline and a Streamlit control
panel for interactively exploring every stage.

Case Study project for the Applied Data Science and Analytics program at SRH Hochschule
Heidelberg. Topic: **Graph Neural Networks for MOF Property Prediction**.

## Pipeline overview

![Pipeline architecture](outputs/pipeline_architecture.png)
<!-- Placeholder: run `streamlit run pipeline_diagram.py` and use the "Download as image"
     button to generate this file, then save it to outputs/pipeline_architecture.png. -->

Each MOF is turned into a graph node; edges connect chemically similar MOFs (by linker
molecule and metal center). A Graph Neural Network passes messages along those edges to
predict PLD, either as one of four pore-size categories (classification) or as a continuous
value in Ångströms (regression).

1. **Data ingestion** (`src/data_ingestion.py`) — loads a MOF dataset into one canonical
   schema (`refcode`, `linker_smiles`, `metal`, `pld_category`, `pld_value`), regardless of
   which of the two source datasets is selected.
2. **Feature engineering** (`src/feature_engineering.py`) — turns each MOF's raw columns
   into a numeric feature vector per node: either a compact 7-dim descriptor or a 1,079-dim
   Morgan-fingerprint-based vector, depending on the dataset.
3. **Graph construction** (`src/graph_construction.py`) — builds 4 candidate similarity
   graphs (1 fixed-threshold, 3 k-NN variants) from linker Tanimoto similarity + metal
   similarity, and picks the one with the best connectivity/community structure.
4. **Black Hole sparsification** (`src/black_hole_sparsification.py`) — scores every node
   by a "gravity" metric (degree + betweenness centrality + edge-weight sum) and prunes the
   graph down to a configurable fraction of nodes/edges, stratified by PLD category so no
   class is wiped out.
5. **GNN training** (`src/gnn_training.py`) — trains GCN, GraphSAGE, and GAT on the full
   graph and on two sparsified variants, with a proper train/validation/test split.
6. **Baseline comparison** (`src/baseline_models.py`) — trains Random Forest and k-NN on
   the same flat feature vectors (no graph) as a sanity check on whether the graph structure
   is actually adding predictive value.
7. **Graph-selection trade-off check** (`src/graph_tradeoff_analysis.py`) — verifies
   Step 3's connectivity-based graph pick against actual downstream model accuracy, since
   the two aren't guaranteed to agree.

All 7 stages are wired into the Streamlit control panel (`app.py`) as tabs, cascading from
one dataset switch in the sidebar.

## Two datasets, one interface

| Dataset | Rows | Node features | Task |
|---|---|---|---|
| Small (`SMILES_METAL_2000_NoPLD.csv`) | 2,000 MOFs | Compact 7-dim (6 metal descriptors + linker molecular weight) | Classification only (no continuous PLD in this file) |
| Large (`MOFCSD.csv`) | 14,296 MOFs | Fingerprint 1,079-dim (1,024-bit Morgan fingerprint + 2 pore-geometry features + 53-way metal one-hot) | Classification or regression |

`src/data_ingestion.py::load_dataset(name)` returns both in the same canonical schema, and
`src/config.py::PipelineConfig` is the single switch point — everything else (feature
scheme, which tasks are available) cascades from the `dataset` choice.

## Setup

No conda — a plain `venv` is used throughout.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements are unpinned (see `requirements.txt`) — PyTorch, PyTorch Geometric, RDKit,
pandas, NetworkX, scikit-learn, psutil, tqdm, matplotlib, and Streamlit.

For development only (not needed to run the pipeline itself): `pip install -r
requirements-dev.txt`. Covers UI verification (`playwright install chromium`) and the test
suite (`pytest`, `hypothesis`, `mutmut`, `pytest-cov`).

## Usage

Each stage can be run standalone as a script, or explored interactively.

```bash
.venv/bin/python src/data_ingestion.py              # EDA, plots to outputs/
.venv/bin/python src/feature_engineering.py         # builds + saves feature matrices to data/processed/
.venv/bin/python src/graph_construction.py          # builds all 4 graphs, saves edge lists + topology table
.venv/bin/python src/black_hole_sparsification.py   # sparsifies the best graph at tau=0.3 and tau=0.5
.venv/bin/python src/gnn_training.py                # trains GCN/GraphSAGE/GAT on all 3 graph variants
.venv/bin/python src/baseline_models.py             # RF + k-NN baselines + master comparison table
.venv/bin/python src/graph_tradeoff_analysis.py     # checks the graph-construction pick against downstream accuracy
```

### Interactive control panel

```bash
.venv/bin/streamlit run app.py
```

A dark, tabbed control panel — one tab per pipeline stage. The sidebar switches dataset
(small/large) and, for the large dataset, task (classification/regression); everything
cascades from there. Most tabs run live on every interaction; the GNN Training tab is
button-gated instead, since a full run takes ~35-40s (small dataset) to ~9 minutes (large
dataset), and every tab's body re-runs on every Streamlit interaction.

To verify UI changes visually (Streamlit renders client-side, so `curl` only returns an
empty shell): `.claude/skills/developing-with-streamlit/capture_screenshots.py` launches
the app headlessly with Playwright and screenshots every tab.

### Pipeline architecture diagram

```bash
.venv/bin/streamlit run pipeline_diagram.py
```

Renders the diagram at the top of this README and includes a "Download as image" button to
export it as a PNG.

## Testing

```bash
pytest tests/ -q                                                     # everything (235 tests), ~10s
pytest tests/ -m "not slow" --cov=src --cov-report=term-missing -q   # skip the 1 real-training smoke test, with coverage
mutmut run && mutmut results                                         # mutation testing (config.py + black_hole_sparsification.py)
```

Four layers under `tests/`: `unit/` (per-function correctness), `property/` (Hypothesis —
invariants across the full input space), `torture/` (extreme/degenerate inputs),
`acceptance/` (documented claims re-verified against the real data files). Full results,
mutation-testing detail, coverage breakdown, and every bug found:
[planning/QA_REPORT.md](planning/QA_REPORT.md).

## Project structure

```
app.py                            Streamlit control panel
pipeline_diagram.py                Standalone architecture-diagram page (downloadable as an image)
src/
  config.py                       PipelineConfig — dataset is the one root switch; feature_scheme
                                   is derived from it; task, gravity weights, and pruning threshold
                                   are the other configurable fields
  data_ingestion.py                Loading, cleaning, EDA plots for both datasets
  feature_engineering.py           Compact / fingerprint node feature schemes
  graph_construction.py            Threshold vs. k-NN graph construction + topology metrics
  black_hole_sparsification.py     Gravity scoring + PLD-stratified node/edge pruning
  gnn_training.py                  GCN / GraphSAGE / GAT + train/eval with a proper 3-way split
  baseline_models.py               Random Forest / k-NN baselines + master comparison table
  graph_tradeoff_analysis.py       Graph-construction connectivity-vs-accuracy trade-off check
  ui_theme.py                      Dark theme + reusable Streamlit components
  logging_setup.py                 Shared logging configuration
data/
  raw/                             Input datasets
  processed/                       Saved feature matrices, graph edge lists, topology tables, results
outputs/                           Generated plots, logs, UI screenshots, architecture diagram image
tests/
  conftest.py                      Shared fixtures (synthetic dataset builders, tiny_graph, etc.)
  unit/                            Per-function correctness tests, one file per src/ module
  property/                        Hypothesis property-based tests — invariants across the full input space
  torture/                         Extreme/degenerate-input tests
  acceptance/                      Documented claims re-verified against the real data files
planning/
  DEVELOPMENT_TODO.md              Development checklist, findings, and decisions log
  MEETING_PREP.md                  Open questions and presentation materials
  CASE_STUDY_REPORT.md             Final written report — background, methodology, results, discussion, thesis extension
  QA_REPORT.md                     Full test/mutation/coverage results and every bug found
.claude/skills/
  developing-with-streamlit/       Streamlit best-practices skill, incl. headless screenshot verification
```

## Key implementation decisions

- **Target leakage removed**: the source pipeline includes the raw Pore Limiting Diameter
  value as an input feature — but that's exactly what both the classification label and the
  regression target are derived from. Excluded here regardless of task
  (`src/feature_engineering.py`).
- **Proper train/validation/test split**: the source training loop reuses the test set for
  early-stopping validation. `src/gnn_training.py` keeps a genuine 3-way split — the fixed
  test nodes from Black Hole sparsification stay untouched until final evaluation; a
  separate validation split drives early stopping.
- **Generalized metal encoding**: the source pipeline hardcodes a 4-slot metal one-hot
  (Cu/Zn/Fe/Co), silently mis-encoding anything else as Cu. The large dataset has 53 unique
  metals — encoded with a full dynamic one-hot instead.
- **Equal gravity weighting by default**: Black Hole's gravity score combines degree
  centrality, betweenness centrality, and edge-weight sum. The default here is
  0.33/0.33/0.33 (equal weighting), matching the source paper's stated main configuration
  — all three weights are independently adjustable sliders in the UI.
- **k-NN vs. fixed similarity threshold**: a fixed threshold (φ=0.9) leaves 20.5% (small
  dataset) / 11.3% (large dataset) of nodes with no edges at all; every k-NN configuration
  guarantees 0% isolated nodes by construction, since every node gets its k nearest
  neighbors regardless of absolute similarity.
- **Deterministic invalid-SMILES handling**: an unparseable linker SMILES string produces a
  fixed all-zero feature vector, not random noise — repeated runs on the same data are
  reproducible.
- **Topology-only graph selection was checked against real accuracy**: picking a graph by
  connectivity/modularity alone doesn't always pick the best-*performing* graph —
  `src/graph_tradeoff_analysis.py` trains one model per graph candidate to check. Correct on
  the small dataset; on the large dataset, two of the k-NN variants beat the
  connectivity-selected pick on accuracy.
- **Non-graph baselines outperform every GNN configuration** on both tasks, on both
  datasets. The graph is built from the same linker/metal similarity that's already encoded
  in the node features, so message passing mostly re-derives information the model already
  has rather than adding new signal. Full reasoning and numbers:
  [planning/CASE_STUDY_REPORT.md](planning/CASE_STUDY_REPORT.md).
