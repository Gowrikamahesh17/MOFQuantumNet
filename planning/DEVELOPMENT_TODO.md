# MOFQuantumNet — Development To-Do List

Development-perspective checklist for Case Study 2: *Graph Neural Networks for MOF Property Prediction*.
Organized by phase, sequential. Check items off as completed.

Reference repos: `reference_content/BlackHole-main/` (PyTorch, modern), `reference_content/MOFGalaxyNet-main/` (TensorFlow/StellarGraph, legacy — logic only, not runtime).

---

## Phase 0 — Environment & Repo Setup

**Hardware note:** MacBook Air M4 is sufficient — no GPU required. Models here are small (2-layer GCN/GraphSAGE, hidden dim 64) on ~2,000 nodes, far smaller than the 12,561-node dataset BlackHole's own README benchmarks at ~10 min on Apple Silicon *CPU*. Train on CPU by default; MPS (Apple's GPU backend) is an optional speed experiment, not a requirement — some PyG scatter/sparse ops have incomplete MPS support, so CPU is the safe default for reproducibility.

- [x] Create a plain `venv` environment (no conda) — `.venv` created and used directly via `.venv/bin/pip` / `.venv/bin/python`
- [x] Upgrade packaging tools: pip 26.1.2, setuptools 83.0.0, wheel 0.47.0
- [x] Install PyTorch (CPU/MPS build, no CUDA needed on Apple Silicon) — torch 2.13.0, torchvision 0.28.0, torchaudio 2.11.0
- [x] Install PyTorch Geometric — torch_geometric 2.8.0; `GCNConv`, `GATConv`, `SAGEConv` all import and run without needing `torch_scatter`/`torch_sparse`/`torch_cluster` on this version
- [x] Install RDKit via pip — rdkit 2026.03.3, no conda needed, arm64 wheel
- [x] Install remaining deps — pandas 3.0.3, numpy 2.5.1, networkx 3.6.1, scikit-learn 1.9.0, psutil 7.2.2, tqdm 4.68.4
- [x] Generate `requirements.txt` (unpinned, top-level packages only — see repo root)
- [x] Verify all imports clean; `torch.backends.mps.is_available()` → `True` (MPS available as an optional speed-up, CPU remains default)
- [x] Set up project skeleton: `src/`, `data/raw/`, `data/processed/`, `notebooks/`, `outputs/`
- [x] Copy dataset files from `MOFGalaxyNet-main/Data/` into `data/raw/` (`SMILES_METAL_2000_NoPLD.csv`, `MOF_Features.csv`, `EdgesList_0.9.csv`, `EdgesList-0.7.csv`, `EdgesList-0.2.csv`)
- [x] Initialize a results log — `src/logging_setup.py::get_logger()` mirrors the `bh_evaluation.log` pattern from `main.py`, writes to `outputs/pipeline.log`

## Phase 1 — Data Ingestion & EDA

**Discovery (important — changes the Phase 2 dataset-choice question):** the BlackHole README claims `MOFGalaxyNet.csv` and `MOFCSD.csv` are "not included," but both are present locally with real data — `MOFCSD.csv` = 14,296 MOFs with genuine continuous PLD, linker SMILES, metal, and pore geometry (no NaNs); `MOFGalaxyNet.csv` = the full 829,300-edge weighted graph. So we actually have **two complete, usable datasets**: the small 2,000-MOF set (paper-original, precomputed category label, no continuous PLD, no metal name) and this larger 14,296-MOF set (continuous PLD, real metal names, supports both classification and regression). See updated open question in `MEETING_PREP.md`.

- [x] Load `SMILES_METAL_2000_NoPLD.csv` (no header row on disk — columns inferred from `Similarity.py`'s indexing: 6 metal features, index, refcode, SMILES, PLD-category code) and `MOFCSD.csv` (the actual larger dataset, since `MOFGalaxyNet.csv`/`MOFCSD.csv` turned out to be present) — see `src/data_ingestion.py`
- [x] Port `derive_pld_category()` logic from `data_utils.py` (bins: <2.4 nonporous, 2.4–4.0 small, 4.0–8.0 medium, >8.0 large) — applied to the large dataset's continuous PLD column
- [x] Reproduce original class histogram — small dataset's precomputed label matches almost exactly: **1062/425/271/246** vs. paper's reported **1062/422/271/244** (off by 2–3 MOFs, negligible)
- [x] Handle known data issues: `MOFCSD.csv` has 3 rows with the invalid SMILES `F[Si](F)(F)(F)(F)F` → replaced with benzene fallback; **zero NaNs** in `MOFCSD.csv` (cleaner than expected); confirmed 53 unique metals in the real data vs. BlackHole's `data_utils.py` hardcoded 4-metal one-hot (Cu/Zn/Fe/Co) — a real gap in that code, flagged as an open question
- [x] EDA script (`src/data_ingestion.py`) generates: class-balance comparison (`outputs/phase1_class_balance.png`), metal-type distribution across all 53 metals (`outputs/phase1_metal_distribution.png`), continuous PLD histogram with category bin edges overlaid (`outputs/phase1_pld_histogram.png`) — kept as a script rather than a `.ipynb` for now since Jupyter isn't in the current dependency set; portable into notebook cells in Phase 7
- [x] Documented discrepancies: (1) small-dataset counts off by 2–3 vs. paper, (2) large dataset's derived category distribution is far more imbalanced (nonporous 7106 / medium 3540 / small 2840 / large 810) than the small dataset's, worth a class-weighting discussion in Phase 5

## Phase 2 — Feature Engineering

- [ ] **Decide feature scheme** (flag for professor sync — see `MEETING_PREP.md`): 7-dim descriptor set (paper) vs. 1031-dim Morgan-fingerprint vector (`BlackHole` code)
- [ ] If 7-dim: replicate SMILES-fingerprint + AN/AW/AR/ME/P/EA vector, normalize with `sklearn.preprocessing.normalize` (per `Normalization.py`)
- [ ] If 1031-dim: port `get_morgan_fingerprint()` and `load_summary_data()` from `data_utils.py` (radius=2, 1024 bits + 3 pore features + 4-dim metal one-hot)
- [ ] Validate output feature matrix shape (N × d) — no NaNs, no shape mismatches
- [ ] Save feature matrix as a reusable artifact (`.npy`/`.csv`) for both graph and baseline models

## Phase 3 — Graph Construction Experiment (core contribution)

- [ ] Implement **Branch A**: fixed-threshold similarity graph (φ = 0.9), porting the weighted-similarity formula from `Similarity.py` (Tanimoto linker similarity + cosine metal-property distance, α = 0.1)
- [ ] Implement **Branch B**: k-NN adjacency builder, parametrized by k ∈ {3, 5, 10}
- [ ] Vectorize/optimize the similarity computation (original `Similarity.py` uses raw nested loops — replace with RDKit bulk fingerprint ops or `scipy.spatial.distance` batch calls for performance)
- [ ] For each of the 4 resulting graphs, compute: isolated-node rate, mean degree, edge count, Louvain modularity (`networkx.algorithms.community.louvain_communities`)
- [ ] Build a topology comparison table across all 4 configurations
- [ ] Visualize degree distributions side-by-side
- [ ] Select the best-performing graph to carry forward into Phase 4/5

## Phase 4 — Black Hole Sparsification Integration

- [ ] Port `calculate_gravity_per_community()` from `bh_sparsification.py` (degree centrality + betweenness centrality + edge-weight-sum, MinMax-normalized per community)
- [ ] **Confirm gravity weights with professor** before running (code default `0.3/0.3/0.4` vs. Expose-stated `0.33/0.33/0.33`)
- [ ] Port `black_hole_strategy_per_community()` — PLD-stratified node retention per community
- [ ] Port `prune_edges()` — ensures fixed test nodes retain ≥1 edge post-pruning
- [ ] Run sparsification at τ = 0.3 (BH-30) and τ = 0.5 (BH-50) on the Phase-3 winning graph
- [ ] Log peak memory usage and graph density at each pruning level (target: reproduce ~114 MB at 50%)
- [ ] Save sparsified edge lists + node feature files per threshold

## Phase 5 — GNN Model Training

- [ ] Port `GCN` class from `graphsage_model.py` (2-layer `GCNConv`, ReLU, dropout 0.3)
- [ ] Port `GraphSAGE` class (custom sparse mean-aggregation implementation) for robustness comparison
- [ ] Port `train()` / `test()` functions — Adam (lr=0.005, weight_decay=1e-3), early stopping (patience=20), 200 epochs max
- [ ] Decide task framing: classification (4-class PLD) vs. regression (continuous PLD) — or both
- [ ] Train GCN + GraphSAGE on: (i) full best graph, (ii) BH-30, (iii) BH-50 — 6 runs total
- [ ] Plot training/validation loss curves for each run
- [ ] Collect test accuracy, confusion matrix, Cohen's Kappa (classification) or MAE/RMSE/R² (regression) per run

## Phase 6 — Baseline Comparison

- [ ] Port fixed-test-node sampling logic (top 4% per community, degree > 2) from `calculate_gravity_per_community()` for a reproducible, identical split across all models
- [ ] Train Random Forest (scikit-learn, ~100 trees) on flat feature matrix
- [ ] Train k-NN classifier (k=5) on flat feature matrix
- [ ] Evaluate both with same metrics as Phase 5 (accuracy, macro F1, confusion matrix)
- [ ] Assemble master comparison table: Threshold-GCN, kNN-GCN, BH30-GCN, BH50-GCN, GraphSAGE, Random Forest, kNN classifier

## Phase 7 — Analysis, Reporting & Thesis Outline

- [ ] Identify best connectivity–accuracy trade-off across graph construction strategies
- [ ] Quantify Black Hole's memory/time savings vs. full graph
- [ ] Draw conclusions on graph topology's contribution vs. raw node features
- [ ] Consolidate all notebooks into one reproducible pipeline notebook
- [ ] Write **Background** section (what MOFs are, why useful in materials science) — explicit roadmap item (a) from the professor's email, has no dedicated dev phase since it's a writing task; tracked here so it isn't dropped
- [ ] Write final case study report (methodology, results tables, figures, discussion, references)
- [ ] Prepare final slide deck summarizing contributions and findings
- [ ] Write thesis-extension section (multi-property prediction, improved similarity via SOAP/3D structure, BH weight optimization, MOF recommendation system)

---

## Roadmap coverage check (vs. professor's email, 2026-06-03)

| His roadmap item | Covered by | Status |
|---|---|---|
| (a) Background — what MOFs are, why useful | Phase 7 report + Slide 2 of the meeting deck | Tracked (writing task, no dev phase) |
| (b) Dataset + target property | Phase 1 | Done — both datasets loaded, EDA complete |
| (c) Graph construction (similarity/k-NN) | Phase 3 | Exceeds ask — both threshold *and* k-NN implemented for comparison |
| (d) GNN model (GCN or GraphSAGE) | Phase 5 | Exceeds ask — both GCN *and* GraphSAGE |
| (e) Evaluation vs. simple baseline | Phase 6 | Exceeds ask — both Random Forest *and* k-NN |
| (f) Thesis extension | Phase 7 | Tracked — four candidate directions already drafted in the Expose |

No gaps — every roadmap item has a home. The only non-dev item (Background) is now explicitly tracked in Phase 7 so it isn't lost.

## Generic, switchable design (dataset + feature scheme)

- [x] `src/config.py` — single source of truth for the two open decisions (`dataset`: "small"/"large", `feature_scheme`: "compact"/"fingerprint"). Change one line here (or a UI toggle, see below) instead of rewriting code once the professor confirms.
- [x] `src/data_ingestion.py::load_dataset(name)` — canonical loader; both datasets return the same column schema (`refcode`, `linker_smiles`, `metal`, `pld_category`, `pld_value`), so Phase 2+ can be written against one interface regardless of which dataset is active.
- [ ] **Note for Phase 2/3 planning:** switching datasets isn't purely a config change — graph construction is O(n²) pairwise similarity. Small (2,000 MOF) → ~4M pairs; large (14,296 MOF) → ~204M pairs (~51x). Write the similarity/k-NN code with vectorized RDKit bulk ops from the start so it scales to either dataset without a rewrite.
- [ ] Phase 2's two feature schemes (compact 7-dim / fingerprint 1031-dim) should be implemented behind the same `feature_scheme` switch when Phase 2 starts — not stubbed yet, since Phase 2 hasn't begun.

## Interactive Control Panel (Streamlit)

- [x] `app.py` (repo root) — run with `.venv/bin/streamlit run app.py`. Sidebar lets you switch dataset and feature scheme live; currently wired to Phase 1 (Data & EDA tab, fully functional: summary stats, class balance, metal distribution, PLD histogram, invalid-SMILES warning). Remaining tabs (Feature Engineering, Graph Construction, Black Hole, GNN Training, Baselines, Results) are visible but explicitly marked "not yet implemented" — each lights up as its phase is built, so the app never overstates what's done.

---

## Cross-cutting / Housekeeping

- [ ] Version control: commit after each phase with clear messages
- [ ] Keep a running log of deviations from the Expose's original plan (and why)
- [ ] Track open questions for the professor in `MEETING_PREP.md` as they arise
- [ ] Back up raw data and trained model checkpoints outside git (large files)
