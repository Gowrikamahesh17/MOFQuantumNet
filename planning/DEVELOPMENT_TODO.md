# MOFQuantumNet — Development Log

Combined development log for Case Study 2: *Graph Neural Networks for MOF Property
Prediction*. Originally two separate files — `DEVELOPMENT_TODO.md` (the core `src/`
pipeline) and `WEBAPP_DEVELOPMENT_TODO.md` (the `webapp/` console) — merged into one on
request, since both describe a now-mostly-finished project and having two decision logs
was more files than the project needed. Kept as three parts rather than interleaved
chronologically, since they were genuinely built as separate phases of work with
different scopes.

- **Part 1 — Original pipeline** (`src/`): the 7-phase data/ML pipeline, plus environment
  setup and QA. This is the project's technical core, and everything else is built on it.
- **Part 2 — Web console** (`webapp/`): the Predict + Lab console, built after Part 1 was
  functionally complete, wrapping `src/` behind a FastAPI backend and a real frontend.
- **Part 3 — Streamlit removal & final cleanup**: the last pass — removing the Streamlit
  app once the web console fully superseded it, and general repo housekeeping.

---

# Part 1 — Original Pipeline (`src/`)

Reference repos: `reference_code/BlackHole-main/` (PyTorch, modern), `reference_code/MOFGalaxyNet-main/` (TensorFlow/StellarGraph, legacy — logic only, not runtime).

---

## Meeting outcome (2026-07-21, Prof. Jalali) — resolves several open questions below

- **Both datasets stay in scope**, selectable via one config/UI switch (as already planned) — not an either/or decision.
- **Feature dimensionality (7-dim vs. 1031-dim) is not an independent choice** — it cascades automatically from the dataset switch. No separate UI control for it.
- **Task framing cascades too**: small dataset → classification only (forced, no continuous PLD exists to regress on); large dataset → user gets a real classification-vs-regression choice.
- **Black Hole gravity weights (α/β/γ) must be configurable via the UI** (sliders), not a fixed constant — this also resolves the "which default is correct" question from the professor sync, since it's now adjustable either way.
- Reuse of his existing code as a reference/base remains explicitly fine.
- Implementation approach agreed for now: **UI/config scaffolding first** — wire up the controls correctly (including a pruning-threshold τ slider alongside the gravity weights), without yet building the Phase 2–4 computation the sliders will eventually feed. See the updated Phase 2/4 entries and the "Generic, switchable design" section below.

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

**Discovery (important — changes the Phase 2 dataset-choice question):** the BlackHole README claims `MOFGalaxyNet.csv` and `MOFCSD.csv` are "not included," but both are present locally with real data — `MOFCSD.csv` = 14,296 MOFs with genuine continuous PLD, linker SMILES, metal, and pore geometry (no NaNs); `MOFGalaxyNet.csv` = the full 829,300-edge weighted graph. So we actually have **two complete, usable datasets**: the small 2,000-MOF set (paper-original, precomputed category label, no continuous PLD, no metal name) and this larger 14,296-MOF set (continuous PLD, real metal names, supports both classification and regression).

- [x] Load `SMILES_METAL_2000_NoPLD.csv` (no header row on disk — columns inferred from `Similarity.py`'s indexing: 6 metal features, index, refcode, SMILES, PLD-category code) and `MOFCSD.csv` (the actual larger dataset, since `MOFGalaxyNet.csv`/`MOFCSD.csv` turned out to be present) — see `src/data_ingestion.py`
- [x] Port `derive_pld_category()` logic from `data_utils.py` (bins: <2.4 nonporous, 2.4–4.0 small, 4.0–8.0 medium, >8.0 large) — applied to the large dataset's continuous PLD column
- [x] Reproduce original class histogram — small dataset's precomputed label matches almost exactly: **1062/425/271/246** vs. paper's reported **1062/422/271/244** (off by 2–3 MOFs, negligible)
- [x] Handle known data issues: `MOFCSD.csv` has 3 rows with the invalid SMILES `F[Si](F)(F)(F)(F)F` → replaced with benzene fallback; **zero NaNs** in `MOFCSD.csv` (cleaner than expected); confirmed 53 unique metals in the real data vs. BlackHole's `data_utils.py` hardcoded 4-metal one-hot (Cu/Zn/Fe/Co) — a real gap in that code, flagged as an open question
- [x] EDA script (`src/data_ingestion.py`) generates: class-balance comparison (`outputs/phase1_class_balance.png`), metal-type distribution across all 53 metals (`outputs/phase1_metal_distribution.png`), continuous PLD histogram with category bin edges overlaid (`outputs/phase1_pld_histogram.png`) — kept as a script rather than a `.ipynb` for now since Jupyter isn't in the current dependency set
- [x] Documented discrepancies: (1) small-dataset counts off by 2–3 vs. paper, (2) large dataset's derived category distribution is far more imbalanced (nonporous 7106 / medium 3540 / small 2840 / large 810) than the small dataset's, worth a class-weighting discussion in Phase 5

## Phase 2 — Feature Engineering ✅ DONE

- [x] **Decide feature scheme** — resolved 2026-07-21: not an independent choice, cascades from the dataset switch (small → compact 7-dim, large → fingerprint). Implemented as a derived property (`PipelineConfig.feature_scheme`) in `src/config.py`
- [x] **Compact scheme (small dataset)** — `src/feature_engineering.py::build_compact_features()`. Uses the 6 existing `metal_feat_0..5` columns + 1 new linker descriptor (molecular weight via RDKit `Descriptors.MolWt`), then column-wise `MinMaxScaler` over all 7 dims. **Deviation from plan:** originally intended to replicate `Normalization.py`'s row-wise L2 `preprocessing.normalize`, but tested the hypothesis that `metal_feat_0..5` were part of a unit-norm 7-dim vector empirically (sum-of-squares across all 2,004 rows) — it doesn't hold (ranges 0.23–2.67, not ~1), so row-wise L2 would have been an unjustified distortion. Used standard column-wise Min-Max scaling instead.
- [x] **Fingerprint scheme (large dataset)** — `src/feature_engineering.py::build_fingerprint_features()`. Ported `get_morgan_fingerprint()` logic using the modern non-deprecated RDKit API (`rdFingerprintGenerator.GetMorganGenerator`, radius=2, 1024 bits) instead of the deprecated `AllChem.GetMorganFingerprintAsBitVect` the original code uses.
- [x] **Real bug found and fixed — target leakage:** BlackHole's `data_utils.py` includes the raw `Pore Limiting Diameter` value directly as an input feature. Since PLD is exactly what both the classification label and the regression target are derived from, this is direct leakage, not a legitimate feature. Excluded it from the feature matrix regardless of task; kept only `Largest Cavity Diameter` and `Largest Free Sphere` (correlated with PLD but not identical to it) — final dimension is **1,079** (1024 fingerprint + 2 geometry + 53 metal one-hot), not the originally-planned 1031.
- [x] **Metal one-hot generalized** — replaced BlackHole's hardcoded 4-slot map (Cu/Zn/Fe/Co, silently mis-encoding anything else as Cu) with a full one-hot built dynamically over all 53 real metals actually present in the data.
- [x] **Reproducibility fix:** BlackHole's invalid-SMILES fallback returns `np.random.randn(n_bits)` with no fixed seed — different every run. Used a deterministic all-zero vector instead, with the affected rows logged.
- [x] **Data-quality finding (source data, not our code):** 16 rows in the small dataset and 1 additional pattern (2 rows) in the large dataset have genuinely truncated/malformed SMILES strings *in the raw CSV itself*. Not a parsing bug on our end; handled via the deterministic fallback above.
- [x] Validate output feature matrix shape (N × d) — no NaNs, no infs, row count matches input — enforced in `build_features()`
- [x] Save feature matrix as a reusable artifact — `data/processed/features_{scheme}.npy` + `_index.csv` (refcodes) + `_metadata.json` (scheme details, metal-to-index mapping)

## Phase 3 — Graph Construction Experiment (core contribution) ✅ DONE

- [x] Implement **Branch A**: fixed-threshold similarity graph (φ = 0.9) — `src/graph_construction.py::build_similarity_graphs()`, porting the weighted-similarity formula from `Similarity.py` (α = 0.1 metal weight, 0.9 linker weight)
- [x] Implement **Branch B**: k-NN adjacency builder, k ∈ {3, 5, 10} — same function, computed in the same pass as Branch A (shared similarity computation, no wasted recomputation)
- [x] **Documented adaptation:** `Similarity.py`'s metal-similarity term is cosine distance over 6 numeric metal-property columns, which only exist in the small dataset. For the large dataset (categorical `metal` name only, no numeric descriptor), used a same/different-metal binary indicator instead (1.0 if two MOFs share a metal, 0.0 otherwise) — captures "shared coordination chemistry" without fabricating a periodic-table properties dataset that isn't part of the given data.
- [x] Vectorized the similarity computation — original `Similarity.py` computes pairwise Tanimoto with one RDKit call per pair in a raw nested Python loop (fine at 2,000×2,000, would be far too slow at 14,296×14,296 ≈ 204M pairs). Reimplemented as bit-matrix multiplication (`intersection = fingerprints @ fingerprints.T`, `union` via broadcast row sums), processed in row blocks (1,500 rows/block) to keep memory bounded instead of ever materializing the full N×N matrix. **Measured runtime: ~10s for the full 14,296-node dataset, ~2s for the 2,004-node dataset** (all 4 graph configs, both branches, in one pass).
- [x] For each of the 4 resulting graphs, compute: isolated-node rate, mean degree, edge count, Louvain modularity — `compute_topology_metrics()`
- [x] Build a topology comparison table across all 4 configurations — `compute_topology_metrics()` returns it directly as a DataFrame; also saved to `data/processed/phase3_topology_{dataset}.csv`
- [x] Visualize degree distributions side-by-side — `plot_degree_distributions()`, saved to `outputs/phase3_degree_distributions_{dataset}.png`
- [x] Select the best-performing graph to carry forward — `select_best_graph()`: among configs tied for lowest isolated-node rate, picks highest modularity. **Result on both datasets: `knn_3` wins** — the fixed threshold (φ=0.9) leaves 20.5% (small) / 11.3% (large) of nodes isolated, directly reproducing the original paper's own noted limitation; every k-NN config guarantees 0% isolated nodes by construction, and k=3 has the highest modularity (0.977 small / 0.986 large) among those.

## Phase 4 — Black Hole Sparsification Integration ✅ DONE

- [x] **Gravity weights confirmed configurable** (resolved 2026-07-21, no longer a fixed-default debate) — `src/config.py::PipelineConfig` now holds `gravity_degree_weight`/`gravity_betweenness_weight`/`gravity_edge_weight_sum_weight`
- [x] Ported `calculate_gravity_per_community()` from `bh_sparsification.py` — `src/black_hole_sparsification.py`, reading `config.gravity_weights_normalized` instead of a hardcoded tuple. **Fixed a cosmetic bug along the way:** the original reuses the loop variable `idx` for both the outer per-community loop and inner per-node loop, so its "no eligible test nodes" warning would log the wrong community index (doesn't affect actual gravity scores/selection, both keyed by node id) — avoided with distinct variable names.
- [x] Ported `black_hole_strategy_per_community()` — PLD-stratified node retention per community, using our own `pld_category` column (works for both datasets). **Adaptation:** the original mutates its input graph in place; ours copies first, since our Phase-3 graph is a single cached object reused across every threshold/weight combination the sliders produce — mutating it in place would corrupt subsequent calls.
- [x] Ported `prune_edges()` — ensures fixed test nodes retain ≥1 edge post-pruning
- [x] Runs at a configured τ on the Phase-3 winning graph (`knn_3` for both datasets) — **measured**: τ=0.3 retains 66.3% of nodes / 38.6% of edges (small), 67.9% / 42.5% (large); τ=0.5 retains 47.9% / 16.1% (small), 48.2% / 18.5% (large) — consistent with the `max(0.2N, (1-τ)N)` node-retention formula
- [x] Logs peak memory (via `psutil`, whole-process RSS) and graph density before/after. **Honest caveat, not a false match:** our reading (~400MB small / ~1.3–1.8GB large) is a whole-Python-process snapshot including PyTorch/RDKit/everything already loaded — not an isolated per-step measurement, so it is **not directly comparable** to the paper's reported ~114MB at 50% pruning, which was measured differently.
- [x] **Finding, not a bug:** pruned graphs retain some fully-isolated nodes (95 small / 924 large at τ=0.3) — inherent to the original algorithm, which only guarantees fixed *test* nodes keep an edge, not every retained node.
- [x] Saved sparsified edge lists + metrics — `data/processed/bh_graph_{dataset}_tau{τ}.csv` + `bh_metrics_{dataset}_tau{τ}.json`

## Phase 5 — GNN Model Training ✅ DONE

- [x] Ported `GCN` class from `graphsage_model.py` (2-layer `GCNConv`, ReLU, dropout 0.3) — `src/gnn_training.py`
- [x] Ported `GraphSAGE` class (custom sparse mean-aggregation implementation, not PyG's built-in `SAGEConv` — matches the reference code's own hand-rolled approach) for robustness comparison.
- [x] **Ported `GAT` too (added 2026-07-24).** An earlier note here claimed the professor had confirmed a 2-model comparison (GCN+GraphSAGE) was sufficient — that was **false**; it was a question drafted for a meeting-prep note that the meeting never actually covered, incorrectly written up afterward as if it had been confirmed. Corrected once caught. Since the actual `BlackHole.pdf` paper's own evaluation (Section 3.4) tests all three models, GAT is now included, matching the paper's full comparison. 12-head attention + `LayerNorm`, ported as-is from `graphsage_model.py`; `edge_weight` is accepted for signature consistency but genuinely ignored (matching the original — `GATConv` learns attention weights instead).
- [x] Ported `train()` / `test()` functions — Adam (lr=0.005, weight_decay=1e-3), early stopping (patience=20), 200 epochs max
- [x] **Real bug found and fixed — test-set leakage:** the reference code's `train()` sets `val_mask = data.test_mask` — early stopping (which epoch's weights get kept) is driven by the *same* set used for final evaluation, which inflates the reported test metric. Fixed with a genuine 3-way split: Phase 4's `fixed_test_nodes` stay completely untouched until final evaluation; a separate validation subset (15% of the remaining nodes) drives early stopping instead. `make_splits()`.
- [x] **Decide task framing** — resolved 2026-07-21: cascades from dataset, not an independent choice. Small dataset → classification only (forced in `PipelineConfig.__post_init__`, since no continuous PLD exists); large dataset → classification or regression.
- [x] Train GCN + GraphSAGE + GAT on: (i) full best graph, (ii) BH-30, (iii) BH-50 — **9 runs total**. `run_gnn_comparison()` reuses the *same* `fixed_test_nodes` (computed once, deterministic given the same graph+weights) across all 3 graph variants, so the test set is identical for a fair comparison even though BH-30/BH-50 have fewer nodes.
- [x] **Measured results (classification, with GAT, α=β=γ=0.33):** small dataset accuracy 0.543–0.609 (Cohen's Kappa 0.15–0.29) across all 9 runs — GAT shows a degenerate majority-class-collapse pattern on the BH-pruned small-dataset graphs specifically, a known sensitivity of attention-based GNNs on small/sparse graphs, not a bug in the port; large dataset accuracy 0.663–0.725 (Kappa 0.50–0.60), no collapse — BH-30/BH-50 still slightly *outperform* the full graph, consistent with Black Hole's own claim. **Regression (large dataset only):** GCN on full graph — MAE 1.64 Å, RMSE 2.70 Å, R² 0.23.
- [x] Plot training/validation loss curves for each run — `plot_loss_curves()`, a 3×3 grid (3 models × 3 graph variants), saved to `outputs/phase5_loss_curves_{dataset}.png`
- [x] Collect test accuracy, confusion matrix, Cohen's Kappa (classification) or MAE/RMSE/R² (regression) per run — `evaluate_model()`
- [x] **Gravity weight default corrected (2026-07-24) after actually reading `BlackHole.pdf`:** changed from the code's `0.3/0.3/0.4` to the paper's own stated `0.33/0.33/0.33` (Section 2.3, p.7 — "we assign equal weights to all three components").

## Phase 6 — Baseline Comparison ✅ DONE

- [x] Reused the fixed-test-node split (top 4% per community, degree > 2) via `gnn_training.py::make_splits()` — same split as the GNNs, not a separate sampling — `src/baseline_models.py`
- [x] Train Random Forest (scikit-learn, 100 trees) on flat feature matrix — classifier or regressor depending on task
- [x] Train k-NN (k=5) on flat feature matrix — classifier or regressor depending on task
- [x] Evaluate both with the same metrics as Phase 5 (accuracy, macro F1, cohen's kappa, confusion matrix for classification; MAE/RMSE/R² for regression)
- [x] Assemble master comparison table — `build_master_comparison_table()`, combines Phase 5's GNN results with Phase 6's baseline results, sorted by accuracy (classification) or R² (regression)
- [x] **Major, honest finding: the non-graph baselines decisively beat every GNN configuration.** Small dataset: Random Forest 0.667 accuracy / κ=0.46 vs. GNNs' best 0.609 (GAT); large dataset classification: RF 0.851 / κ=0.78 vs. GNNs' best 0.725; large dataset **regression: RF R²=0.893, MAE=0.53 Å vs. GCN's R²=0.23, MAE=1.64 Å** — a large, consistent gap in both tasks. Likely explanation, not just "GNN is worse": the large dataset's 2 pore-geometry features (Largest Cavity Diameter, Largest Free Sphere) are strongly correlated with PLD and directly available to the tree ensembles, while the graph is built from the *same* linker/metal similarity already encoded in the node features — message passing mostly re-derives information already present rather than adding new signal.

## Phase 7 — Analysis, Reporting & Thesis Outline ✅ DONE (notebook consolidation deliberately skipped)

- [x] **Identify best connectivity–accuracy trade-off across graph construction strategies** — closed a real gap: Phase 3's `select_best_graph()` picks by topology alone (isolated rate, modularity), never validated against downstream accuracy. `src/graph_tradeoff_analysis.py` trains one GCN per graph candidate to check. **Finding:** `knn_3` (Phase 3's pick) is genuinely best on the small dataset, but on the large dataset `knn_5`/`knn_10` both beat it on accuracy and κ — reproduced across two runs. Also found training isn't seeded (run-to-run variance observed directly on the small dataset). Results saved to `data/processed/graph_tradeoff_{dataset}.csv`.
- [x] **Quantify Black Hole's memory/time savings vs. full graph** — node retention 66–68% (τ=0.3) / 48% (τ=0.5) across both datasets; edge retention drops faster (39–43% / 16–19%).
- [x] **Draw conclusions on graph topology's contribution vs. raw node features** — Phase 6's central finding: non-graph baselines decisively beat every GNN configuration on both tasks, both datasets.
- [ ] Consolidate all notebooks into one reproducible pipeline notebook — **deliberately not done**: the pipeline has been script-based throughout (no Jupyter dependency added); a separate scope decision left for explicit request rather than assumed.
- [x] Write **Background** section (what MOFs are, why useful in materials science) — `planning/CASE_STUDY_REPORT.md` Section 1.
- [x] Write final case study report (methodology, results tables, figures, discussion, references) — `planning/CASE_STUDY_REPORT.md`, all numbers cited directly from saved artifacts in `data/processed/`, not re-derived from memory.
- [x] `planning/PPT_PROMPT.md` — slide-by-slide prompts for a results-summary deck, generated on request.
- [x] Write thesis-extension section (multi-property prediction, improved similarity via SOAP/3D structure, BH weight optimization, MOF recommendation system) — `planning/CASE_STUDY_REPORT.md` Section 7.

## Phase 8 — Testing & Quality Assurance ✅ DONE

Full write-up: `planning/QA_REPORT.md`. Summary here for the phase log.

- [x] Test infrastructure — `pytest`, `pytest-cov`, `hypothesis`, `mutmut` in `requirements-dev.txt`; `pyproject.toml` config (`[tool.pytest.ini_options]`, `[tool.coverage.run]`, `[tool.mutmut]`); `tests/{unit,property,torture,acceptance}/` layout with shared fixtures in `tests/conftest.py`
- [x] Unit tests (191 tests) — every `src/` module with meaningful branching logic
- [x] Property-based tests (Hypothesis, 11 groups) — invariants across the full input space: `derive_pld_category` always valid/monotonic, gravity weights always normalize to sum 1, small dataset always forces classification, class weights always sum to 1, splits always partition cleanly, sparsification never increases node/edge counts or mutates the input graph
- [x] Torture tests (26 tests) — extreme/degenerate inputs (empty dataframes, single-node graphs, all-identical linkers, all-invalid SMILES, zero-edge graphs, extreme feature magnitudes). **Found and fixed 3 real production bugs** this uncovered: `build_similarity_graphs` crashing below 11 rows (`np.argpartition` kth-out-of-bounds), `compute_fingerprint_matrix` crashing on an empty Series, and empty-string SMILES silently parsing as a "valid" 0-atom molecule
- [x] Acceptance tests (19 tests, 1 marked `slow`) — verify documented claims (row counts, category histograms, dimensions, retention percentages, isolated-node rates) against the *real* data files, not synthetic fixtures
- [x] Mutation testing (`mutmut`, scoped to `config.py` + `black_hole_sparsification.py`) — 435 mutants, improved from 298 killed (68.5%) to 367 killed (84.4%) by adding 6 targeted tests for genuine gaps mutation testing surfaced (e.g. an un-seeded `louvain_communities` call that would have broken run-to-run determinism, and per-community centrality silently degrading to whole-graph centrality). **Caveat:** this mutmut version generates zero mutants for methods inside a `@dataclass`-decorated class, so `config.py`'s own logic has 0% direct mutation coverage despite 100% line coverage — a tool limitation, not a code or test gap.
- [x] Coverage: 67% of `src/` overall (`pytest-cov`); gaps concentrated in `if __name__ == "__main__":` demo blocks and `matplotlib` plotting functions.
- [x] Fixed a real bug this phase's own tooling surfaced: `logging_setup.py::get_logger()` assumed `outputs/` already existed relative to cwd; mutmut running from its own scratch `mutants/` directory (which lacks that folder) crashed with `FileNotFoundError`. Fixed with `os.makedirs(log_dir, exist_ok=True)`.

---

## Roadmap coverage check (vs. professor's email, 2026-06-03)

| His roadmap item | Covered by | Status |
|---|---|---|
| (a) Background — what MOFs are, why useful | Phase 7 report | Done |
| (b) Dataset + target property | Phase 1 | Done — both datasets loaded, EDA complete |
| (c) Graph construction (similarity/k-NN) | Phase 3 | Exceeds ask — both threshold *and* k-NN implemented for comparison |
| (d) GNN model (GCN or GraphSAGE) | Phase 5 | Exceeds ask — GCN, GraphSAGE, *and* GAT |
| (e) Evaluation vs. simple baseline | Phase 6 | Exceeds ask — both Random Forest *and* k-NN |
| (f) Thesis extension | Phase 7 | Four candidate directions drafted |

No gaps — every roadmap item has a home.

## Generic, switchable design (dataset → feature scheme → task, + Black Hole params)

- [x] `src/config.py::PipelineConfig` — single source of truth:
  - `dataset` ("small"/"large") — the one root switch
  - `feature_scheme` — **derived property**, not a field (cascades from `dataset`, can't be set independently)
  - `task` ("classification"/"regression") — forced to "classification" in `__post_init__` when `dataset == "small"`
  - `gravity_degree_weight` / `gravity_betweenness_weight` / `gravity_edge_weight_sum_weight` — configurable via the webapp's retrain form (Part 2)
  - `gravity_weights_normalized` — derived property, always sums to 1 regardless of raw values
  - `pruning_threshold` (τ) — validated to `[0.0, 0.9]`
- [x] `src/data_ingestion.py::load_dataset(name)` — canonical loader; both datasets return the same column schema (`refcode`, `linker_smiles`, `metal`, `pld_category`, `pld_value`).

---

# Part 2 — Web Console (`webapp/`)

Successor UI to the original Streamlit app — one web app with two surfaces, **Predict**
(run every trained model on a MOF, ranked by measured accuracy) and **Lab** (the full
7-stage pipeline, reskinned, plus a retrain form). Design was prototyped through several
mockup iterations before being built for real; the mockup file itself was deleted once
superseded by the real, backend-connected `webapp/frontend/`.

Architecture decision (settled): **one FastAPI process serves both the JSON API and the
static frontend** — single `uvicorn` command, single URL. `src/*.py` stays
framework-agnostic and mostly unchanged; `webapp/backend/` wraps it.

---

## Phase 0 — Scaffolding

- [x] Decide the stack: FastAPI + vanilla JS frontend (no build step), single process serving both
- [x] `fastapi`, `uvicorn`, `joblib` added to `requirements.txt`; `httpx` (needed by FastAPI's `TestClient`) added to `requirements-dev.txt`
- [x] `webapp/` directory layout: `webapp/backend/` (Python), `webapp/frontend/` (static HTML/CSS/JS), `webapp/models/` (trained artifacts, gitignored — regenerable via the training CLIs)
- [x] `webapp/models/` added to `.gitignore`
- [x] **Real pre-existing bug found and fixed while wiring this up:** `src/data_ingestion.py::load_large_dataset()`'s default path and `tests/acceptance/test_acceptance.py`'s `LARGE_DATA_PATH` both pointed at `reference_content/BlackHole-main/MOFCSD.csv` — a directory name that doesn't exist in this repo (it's actually `reference_code/`). The acceptance tests' skip-on-missing-directory guard silently masked this for every large-dataset test rather than erroring. Fixed both paths.
- [x] **Real fragility found and fixed:** the large dataset (`MOFCSD.csv`) was being read live from `reference_code/BlackHole-main/` — gitignored professor reference material with no guarantee it stays in the repo — while the small dataset was properly copied into `data/raw/` as this project's own tracked input. Copied `MOFCSD.csv` into `data/raw/` too and repointed `load_large_dataset()`'s default there.

## Phase 1 — Model persistence (baselines first)

Baselines before GNNs, deliberately — per the case study's own finding, they're both the
more accurate *and* the simpler path (no graph-insertion step needed for a brand-new MOF).

- [x] `webapp/backend/model_store.py` — trains Random Forest + k-NN for a given (dataset, task) using the *exact* same fixed-test-node split as `src/gnn_training.py::make_splits()` (so numbers match the case study report), saves the fitted model via `joblib` plus a metadata JSON
- [x] `webapp/backend/train_models.py` — CLI: `python -m webapp.backend.train_models --all` (re)builds every baseline artifact for both datasets
- [x] All 3 valid (dataset, task) combinations actually trained end to end: small/classification (RF 0.690 acc / κ 0.52, 4s), large/classification (RF 0.850 / κ 0.78, 89s), large/regression (RF R²=0.876, MAE=0.53, 203s) — small variance from the report's numbers is expected (training isn't seeded, per Phase 7 above)
- [x] `webapp/backend/featurize.py` — turns a raw (metal, SMILES, ± geometry) description into the exact feature vector shape each persisted model expects
- [x] **Real gap found, not papered over — small dataset has no metal identity at all.** `data_ingestion.SMALL_DATASET_COLUMNS` confirms the small dataset only ever had 6 anonymous `metal_feat_0..5` numbers, no metal symbol column. `featurize_compact()` accepts an optional explicit `metal_feat` (6 floats) for power users; otherwise it falls back to training-set medians and the API flags `used_median_metal_feat: true`.
- [x] **Real gap found, not papered over — large dataset's features include real pore geometry.** The fingerprint scheme was trained on each MOF's actual Largest Cavity Diameter / Largest Free Sphere — values from simulating the real 3D structure, which a hypothetical new MOF doesn't have. `featurize_fingerprint()` accepts them as optional overrides; otherwise imputes the training-set median and flags `used_median_geometry: true`.
- [x] Verified via real HTTP calls (`fastapi.testclient.TestClient`): health check, dataset stats, and predictions for all 3 trained combinations, including caveat flags and error responses — see `tests/webapp/test_api.py`

## Phase 2 — Predict API + Lab data endpoints

- [x] `webapp/backend/api.py` — FastAPI app, mounts `webapp/frontend/` as static files
- [x] `GET /api/health`, `GET /api/datasets` (+ `/{dataset}`) — dataset metadata computed live, cached in-process
- [x] `POST /api/predict` — feature-engineers the input MOF, loads persisted models, returns a ranked leaderboard sorted by each model's own measured accuracy/R²
- [x] `GET /api/lab/{dataset}/graph` — real topology comparison table, verified live: large dataset returns `knn_3` selected, ~33K edges / 0% isolated / modularity 0.986 (matches the report closely)
- [x] `GET /api/lab/{dataset}/pruning` — real Black Hole retention stats at a given τ, verified live: large τ=0.3 → 67.99% node retention (report: 67.9%), small τ=0.5 → 48.4% (report: 47.9%)

## Phase 3 — GNN persistence + inductive insertion

The harder half, scoped separately on purpose.

- [x] `src/gnn_training.py::run_gnn_comparison` extended with `return_models=True` — each of the 9 (variant, model) results now also carries the trained model object + `dim_in`/`dim_h`/`dim_out`
- [x] `webapp/backend/gnn_store.py` — trains GCN/GraphSAGE/GAT (which internally trains all 3 graph variants each), persists **whichever variant scored best on the measured test metric for that specific architecture** — the same criterion the case study report uses for "best GNN config". Not necessarily the same variant across all three models — expected/correct. The graph itself isn't serialized — `lab.py::get_graph_variant()` reconstructs any of the 3 variants deterministically on demand.
- [x] `webapp/backend/inductive.py` — computes a new MOF's similarity to every existing node using the *exact* formula `graph_construction.py` uses to build the graph, attaches it via 3 nearest surviving neighbors in whichever variant that model was trained on, augments the feature matrix + edge list, runs one real forward pass
- [x] Wired into `POST /api/predict` — all 5 leaderboard rows are real once both baseline and GNN artifacts exist; each graph row reports `graph_variant` and `n_neighbors_found`
- [x] **Verified for real**: small dataset — full 5-model leaderboard correctly ranked (Random Forest 0.690 > k-NN 0.683 > GAT 0.606 > GCN 0.570 > GraphSAGE 0.563) on a live, never-before-seen SMILES input, matching the case study's own baselines-beat-GNNs finding
- [x] **Real Windows bug found and fixed**: the training CLIs crashed *after* successfully saving artifacts, because printing a graph variant name containing "τ" to a cp1252 Windows console raises `UnicodeEncodeError`. Fixed with `sys.stdout.reconfigure(encoding="utf-8")`.

## Phase 4 — Remaining Lab endpoints

- [x] `GET /api/lab/{dataset}/training` — serves the real 9-run GCN/GraphSAGE/GAT comparison (`all_runs` metrics + per-epoch loss history, plus `served_models`). Deliberately read-only: never triggers training itself — 503s with the exact CLI command if artifacts aren't there yet.
- [x] `GET /api/lab/{dataset}/baselines` — real master comparison table
- [x] `GET /api/lab/{dataset}/findings` — real scoreboard computed from whatever's actually trained (`best_baseline`, `best_gnn`, `baseline_beats_gnn`)
- [x] Verified live end to end for both datasets (`tests/webapp/test_lab_api.py`)

## Phase 5 — Frontend wiring

- [x] `webapp/frontend/index.html` + `app.js` — real, backend-connected frontend (no build step, plain `fetch()`)
- [x] Predict page: real leaderboard from `POST /api/predict` — rows, ranks, confidence, caveats all rendered from the live API response
- [x] Lab pages: every stepper panel fetches its real endpoint lazily, on first view or a dataset/task switch
- [x] Network-graph visualization is **real**: `GET /api/lab/{dataset}/graph-sample` and `/pruning-sample` reuse `graph_construction.sample_subgraph_for_viz` + a real `networkx.spring_layout`, colored by each sampled node's actual PLD category. The before/after Black Hole view reuses the exact same cached sample + positions for both panels.
- [x] Two more gaps closed while wiring the frontend: `GET /api/lab/{dataset}/features` (feature-engineering summary) and a `degree_histogram` field on `GET /api/lab/{dataset}/graph`.
- [x] **Retrain button, added per explicit request.** The Black Hole Pruning step has a real form — gravity-weight sliders, a pruning-threshold slider, a target selector (baselines / GNNs / both) — POSTing to `POST /api/lab/{dataset}/retrain`. Training runs in a background thread (`webapp/backend/jobs.py`) so the request returns immediately with a job id; the frontend polls `GET /api/jobs/{job_id}` every 2s. On completion, it auto-reloads whichever Lab step is active.
- [x] Verified live end to end (predict flow, all 7 Lab steps, a real retrain that completed in ~10s and updated the served metadata correctly).

## Phase 6 — Tests, parity, cleanup

- [x] `tests/webapp/` — 26 tests across `test_api.py` + `test_lab_api.py`
- [x] Parity check: numbers surfaced through the console cross-checked against `planning/CASE_STUDY_REPORT.md` throughout Phases 1–4
- [x] Planning docs consolidated: `planning/MEETING_PREP.md` and `planning/mockup_console.html` deleted (obsolete/superseded); `PROJECT_OVERVIEW.md` moved into `planning/`; this file and the original `WEBAPP_DEVELOPMENT_TODO.md` merged into one.

## Resolved product decisions

- **Small dataset + Predict, arbitrary-MOF mode.** Resolved as: accept the MOF anyway, using dataset-median `metal_feat` values, with `used_median_metal_feat: true` surfaced both in the API response and in the frontend copy.
- **Large dataset + Predict, missing pore geometry.** Median-imputed when not supplied, `used_median_geometry: true` surfaced as a plain-language callout.

---

# Part 3 — Streamlit Removal & Final Cleanup

Once the web console (Part 2) covered everything Streamlit did — predicting, and viewing
every pipeline stage live — Streamlit was removed entirely rather than kept as a second,
now-redundant surface. This part records that removal plus a final structural cleanup
pass.

- [x] **Deleted**: `app.py` (the Streamlit control panel), `src/ui_theme.py` (Streamlit-only theme components), `src/graph_viz.py` (Streamlit-only interactive graph component — confirmed nothing else imports it; the webapp's own network visualization is unrelated, built directly on `graph_construction.sample_subgraph_for_viz`), `.streamlit/config.toml`, `outputs/ui_screenshots/` (screenshots of the now-deleted UI).
- [x] `streamlit` removed from `requirements.txt`; `playwright` removed from `requirements-dev.txt` (it existed solely to screenshot the Streamlit app — confirmed via repo-wide search that nothing else uses it).
- [x] `.gitignore`'s `.streamlit/` entries removed.
- [x] **`pipeline_diagram.py` rewritten**, not deleted — it now writes the same SVG diagram directly to `outputs/pipeline_architecture.svg` as a plain script (`python pipeline_diagram.py`), no Streamlit, no new dependency (SVG renders natively in GitHub/browsers, so no PNG rasterization step is needed either). Diagram content is unchanged, only the delivery mechanism.
- [x] Verified via repo-wide search before deleting anything: no test file, and no part of `webapp/`, actually imports Streamlit, `ui_theme`, or `graph_viz` — confirmed the removal was safe, not just assumed.
- [x] Stale references fixed: `src/config.py`'s docstring pointed at the now-deleted `app.py`, repointed to the webapp's retrain form; `planning/PROJECT_OVERVIEW.md`'s description of the interactive dashboard updated to describe the web console instead of Streamlit.
- [x] **`webapp/backend/` structural cleanup**: 6 files each repeated their own `sys.path.insert(...src...)` boilerplate to import from `src/`. Centralized into `webapp/backend/__init__.py` (runs once, on package import) and stripped from the individual files — verified with a clean `__pycache__` + full test rerun.
- [x] **Logging gap closed**: `gnn_store.py`/`model_store.py` already logged to `outputs/pipeline.log`; `api.py` (every predict/retrain request) and `jobs.py` (background job start/success/failure) had none. Added `get_logger` calls to both.
- [x] Full test suite re-verified after all of the above — nothing in `tests/` depended on any of the removed files.
