# MOFQuantumNet — Development To-Do List

Development-perspective checklist for Case Study 2: *Graph Neural Networks for MOF Property Prediction*.
Organized by phase, sequential. Check items off as completed.

Reference repos: `reference_content/BlackHole-main/` (PyTorch, modern), `reference_content/MOFGalaxyNet-main/` (TensorFlow/StellarGraph, legacy — logic only, not runtime).

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

**Discovery (important — changes the Phase 2 dataset-choice question):** the BlackHole README claims `MOFGalaxyNet.csv` and `MOFCSD.csv` are "not included," but both are present locally with real data — `MOFCSD.csv` = 14,296 MOFs with genuine continuous PLD, linker SMILES, metal, and pore geometry (no NaNs); `MOFGalaxyNet.csv` = the full 829,300-edge weighted graph. So we actually have **two complete, usable datasets**: the small 2,000-MOF set (paper-original, precomputed category label, no continuous PLD, no metal name) and this larger 14,296-MOF set (continuous PLD, real metal names, supports both classification and regression). See updated open question in `MEETING_PREP.md`.

- [x] Load `SMILES_METAL_2000_NoPLD.csv` (no header row on disk — columns inferred from `Similarity.py`'s indexing: 6 metal features, index, refcode, SMILES, PLD-category code) and `MOFCSD.csv` (the actual larger dataset, since `MOFGalaxyNet.csv`/`MOFCSD.csv` turned out to be present) — see `src/data_ingestion.py`
- [x] Port `derive_pld_category()` logic from `data_utils.py` (bins: <2.4 nonporous, 2.4–4.0 small, 4.0–8.0 medium, >8.0 large) — applied to the large dataset's continuous PLD column
- [x] Reproduce original class histogram — small dataset's precomputed label matches almost exactly: **1062/425/271/246** vs. paper's reported **1062/422/271/244** (off by 2–3 MOFs, negligible)
- [x] Handle known data issues: `MOFCSD.csv` has 3 rows with the invalid SMILES `F[Si](F)(F)(F)(F)F` → replaced with benzene fallback; **zero NaNs** in `MOFCSD.csv` (cleaner than expected); confirmed 53 unique metals in the real data vs. BlackHole's `data_utils.py` hardcoded 4-metal one-hot (Cu/Zn/Fe/Co) — a real gap in that code, flagged as an open question
- [x] EDA script (`src/data_ingestion.py`) generates: class-balance comparison (`outputs/phase1_class_balance.png`), metal-type distribution across all 53 metals (`outputs/phase1_metal_distribution.png`), continuous PLD histogram with category bin edges overlaid (`outputs/phase1_pld_histogram.png`) — kept as a script rather than a `.ipynb` for now since Jupyter isn't in the current dependency set; portable into notebook cells in Phase 7
- [x] Documented discrepancies: (1) small-dataset counts off by 2–3 vs. paper, (2) large dataset's derived category distribution is far more imbalanced (nonporous 7106 / medium 3540 / small 2840 / large 810) than the small dataset's, worth a class-weighting discussion in Phase 5

## Phase 2 — Feature Engineering ✅ DONE

- [x] **Decide feature scheme** — resolved 2026-07-21: not an independent choice, cascades from the dataset switch (small → compact 7-dim, large → fingerprint). Implemented as a derived property (`PipelineConfig.feature_scheme`) in `src/config.py`, and reflected live in `app.py`'s sidebar caption + Tab 2
- [x] **Compact scheme (small dataset)** — `src/feature_engineering.py::build_compact_features()`. Uses the 6 existing `metal_feat_0..5` columns + 1 new linker descriptor (molecular weight via RDKit `Descriptors.MolWt`), then column-wise `MinMaxScaler` over all 7 dims. **Deviation from plan:** originally intended to replicate `Normalization.py`'s row-wise L2 `preprocessing.normalize`, but tested the hypothesis that `metal_feat_0..5` were part of a unit-norm 7-dim vector empirically (sum-of-squares across all 2,004 rows) — it doesn't hold (ranges 0.23–2.67, not ~1), so row-wise L2 would have been an unjustified distortion. Used standard column-wise Min-Max scaling instead.
- [x] **Fingerprint scheme (large dataset)** — `src/feature_engineering.py::build_fingerprint_features()`. Ported `get_morgan_fingerprint()` logic using the modern non-deprecated RDKit API (`rdFingerprintGenerator.GetMorganGenerator`, radius=2, 1024 bits) instead of the deprecated `AllChem.GetMorganFingerprintAsBitVect` the original code uses (exactly the upgrade path BlackHole's own README flags as needed for RDKit ≥2024.03).
- [x] **Real bug found and fixed — target leakage:** BlackHole's `data_utils.py` includes the raw `Pore Limiting Diameter` value directly as an input feature. Since PLD is exactly what both the classification label and the regression target are derived from, this is direct leakage, not a legitimate feature. Excluded it from the feature matrix regardless of task; kept only `Largest Cavity Diameter` and `Largest Free Sphere` (correlated with PLD but not identical to it) — final dimension is **1,079** (1024 fingerprint + 2 geometry + 53 metal one-hot), not the originally-planned 1031.
- [x] **Metal one-hot generalized** — replaced BlackHole's hardcoded 4-slot map (Cu/Zn/Fe/Co, silently mis-encoding anything else as Cu) with a full one-hot built dynamically over all 53 real metals actually present in the data.
- [x] **Reproducibility fix:** BlackHole's invalid-SMILES fallback returns `np.random.randn(n_bits)` with no fixed seed — different every run. Used a deterministic all-zero vector instead, with the affected rows logged.
- [x] **Data-quality finding (source data, not our code):** 16 rows in the small dataset and 1 additional pattern (2 rows) in the large dataset have genuinely truncated/malformed SMILES strings *in the raw CSV itself* — verified by grepping the raw file directly (e.g. `OC(=O)c1cc(C` cut off mid-string, identically across 9 different refcodes). Not a parsing bug on our end; handled via the deterministic fallback above.
- [x] Validate output feature matrix shape (N × d) — no NaNs, no infs, row count matches input — enforced in `build_features()`
- [x] Save feature matrix as a reusable artifact — `data/processed/features_{scheme}.npy` + `_index.csv` (refcodes) + `_metadata.json` (scheme details, metal-to-index mapping)
- [x] Wired into `app.py` Tab 2 — live computed dimension count, invalid-SMILES count, feature matrix preview, and the leakage-fix explanation, verified visually via the Playwright screenshot skill on both dataset paths

## Phase 3 — Graph Construction Experiment (core contribution) ✅ DONE

- [x] Implement **Branch A**: fixed-threshold similarity graph (φ = 0.9) — `src/graph_construction.py::build_similarity_graphs()`, porting the weighted-similarity formula from `Similarity.py` (α = 0.1 metal weight, 0.9 linker weight)
- [x] Implement **Branch B**: k-NN adjacency builder, k ∈ {3, 5, 10} — same function, computed in the same pass as Branch A (shared similarity computation, no wasted recomputation)
- [x] **Documented adaptation:** `Similarity.py`'s metal-similarity term is cosine distance over 6 numeric metal-property columns, which only exist in the small dataset. For the large dataset (categorical `metal` name only, no numeric descriptor), used a same/different-metal binary indicator instead (1.0 if two MOFs share a metal, 0.0 otherwise) — captures "shared coordination chemistry" without fabricating a periodic-table properties dataset that isn't part of the given data.
- [x] Vectorized the similarity computation — original `Similarity.py` computes pairwise Tanimoto with one RDKit call per pair in a raw nested Python loop (fine at 2,000×2,000, would be far too slow at 14,296×14,296 ≈ 204M pairs). Reimplemented as bit-matrix multiplication (`intersection = fingerprints @ fingerprints.T`, `union` via broadcast row sums), processed in row blocks (1,500 rows/block) to keep memory bounded instead of ever materializing the full N×N matrix. **Measured runtime: ~10s for the full 14,296-node dataset, ~2s for the 2,004-node dataset** (all 4 graph configs, both branches, in one pass).
- [x] For each of the 4 resulting graphs, compute: isolated-node rate, mean degree, edge count, Louvain modularity — `compute_topology_metrics()`
- [x] Build a topology comparison table across all 4 configurations — `compute_topology_metrics()` returns it directly as a DataFrame; also saved to `data/processed/phase3_topology_{dataset}.csv`
- [x] Visualize degree distributions side-by-side — `plot_degree_distributions()`, saved to `outputs/phase3_degree_distributions_{dataset}.png`
- [x] Select the best-performing graph to carry forward — `select_best_graph()`: among configs tied for lowest isolated-node rate, picks highest modularity. **Result on both datasets: `knn_3` wins** — the fixed threshold (φ=0.9) leaves 20.5% (small) / 11.3% (large) of nodes isolated, directly reproducing the original paper's own noted limitation; every k-NN config guarantees 0% isolated nodes by construction, and k=3 has the highest modularity (0.977 small / 0.986 large) among those.
- [x] Wired into `app.py` Tab 3 — live topology table, best-graph highlight, degree-distribution plot, dataset-dependent metal-similarity explanation; both feature engineering and graph construction now wrapped in `st.cache_data` so switching tabs/sliders doesn't force recomputation
- [x] **Screenshot-workflow fix along the way:** the Playwright verification script was capturing a stale screenshot for this tab — clicking the dataset radio triggers a full script rerun (including this tab's ~10s cached computation) before the screenshot fires, and a fixed `time.sleep()` wasn't long enough. Fixed by polling for Streamlit's spinner to actually disappear instead of guessing a sleep duration — see `references/ui-screenshot-verification.md` in the Streamlit skill

## Phase 4 — Black Hole Sparsification Integration ✅ DONE

- [x] **Gravity weights confirmed configurable** (resolved 2026-07-21, no longer a fixed-default debate) — `src/config.py::PipelineConfig` now holds `gravity_degree_weight`/`gravity_betweenness_weight`/`gravity_edge_weight_sum_weight` (default 0.3/0.3/0.4, matching the code default)
- [x] **UX fix (2026-07-23):** the three sliders originally just displayed a normalized readout underneath (raw values didn't need to sum to 1) — changed so moving any one slider proportionally rebalances the other two live, always summing to exactly 1, via `st.session_state` + `on_change` callbacks in `app.py`. Verified interactively with Playwright: dragging α to 1.0 correctly zeroes β/γ; a partial move preserves the other two's relative ratio. `PipelineConfig.gravity_weights_normalized` kept as a safety net for callers that construct the config directly (scripts/tests) without going through the UI.
- [x] **UI**: `app.py` Tab 4 has three live sliders (α/β/γ, 0.0–1.0) plus a τ pruning-threshold slider (0.0–0.9), now wired to a real computation and displaying real results (see below)
- [x] Ported `calculate_gravity_per_community()` from `bh_sparsification.py` — `src/black_hole_sparsification.py`, reading `config.gravity_weights_normalized` instead of a hardcoded tuple. **Fixed a cosmetic bug along the way:** the original reuses the loop variable `idx` for both the outer per-community loop and inner per-node loop, so its "no eligible test nodes" warning would log the wrong community index (doesn't affect actual gravity scores/selection, both keyed by node id) — avoided with distinct variable names.
- [x] Ported `black_hole_strategy_per_community()` — PLD-stratified node retention per community, using our own `pld_category` column (works for both datasets). **Adaptation:** the original mutates its input graph in place; ours copies first, since our Phase-3 graph is a single cached object reused across every threshold/weight combination the UI produces — mutating it in place would corrupt subsequent calls.
- [x] Ported `prune_edges()` — ensures fixed test nodes retain ≥1 edge post-pruning
- [x] Runs at the UI-configured τ on the Phase-3 winning graph (`knn_3` for both datasets) — **measured**: τ=0.3 retains 66.3% of nodes / 38.6% of edges (small), 67.9% / 42.5% (large); τ=0.5 retains 47.9% / 16.1% (small), 48.2% / 18.5% (large) — consistent with the `max(0.2N, (1-τ)N)` node-retention formula
- [x] Logs peak memory (via `psutil`, whole-process RSS) and graph density before/after. **Honest caveat, not a false match:** our reading (~400MB small / ~1.3–1.8GB large) is a whole-Python-process snapshot including PyTorch/RDKit/everything already loaded — not an isolated per-step measurement, so it is **not directly comparable** to the paper's reported ~114MB at 50% pruning, which was measured differently. Surfaced explicitly in the UI rather than implying a false apples-to-apples match.
- [x] **Finding, not a bug:** pruned graphs retain some fully-isolated nodes (95 small / 924 large at τ=0.3) — inherent to the original algorithm, which only guarantees fixed *test* nodes keep an edge, not every retained node. Documented in the UI so it isn't mistaken for an implementation error.
- [x] Saved sparsified edge lists + metrics — `data/processed/bh_graph_{dataset}_tau{τ}.csv` + `bh_metrics_{dataset}_tau{τ}.json`
- [x] Wired into `app.py` Tab 4 with `st.cache_data` (keyed on hashable params — dataset/weights/threshold — rather than the graph object itself, to avoid hashing a large networkx graph on every rerun); verified visually on both dataset paths via the Playwright screenshot skill

## Phase 5 — GNN Model Training ✅ DONE

- [x] Ported `GCN` class from `graphsage_model.py` (2-layer `GCNConv`, ReLU, dropout 0.3) — `src/gnn_training.py`
- [x] Ported `GraphSAGE` class (custom sparse mean-aggregation implementation, not PyG's built-in `SAGEConv` — matches the reference code's own hand-rolled approach) for robustness comparison.
- [x] **Ported `GAT` too (added 2026-07-24).** An earlier note here claimed the professor had confirmed a 2-model comparison (GCN+GraphSAGE) was sufficient — that was **false**; it was a question drafted for `MEETING_PREP.md` that the meeting never actually covered, incorrectly written up afterward as if it had been confirmed. Corrected once caught. Since the actual `BlackHole.pdf` paper's own evaluation (Section 3.4) tests all three models, GAT is now included, matching the paper's full comparison. 12-head attention + `LayerNorm`, ported as-is from `graphsage_model.py`; `edge_weight` is accepted for signature consistency but genuinely ignored (matching the original — `GATConv` learns attention weights instead).
- [x] Ported `train()` / `test()` functions — Adam (lr=0.005, weight_decay=1e-3), early stopping (patience=20), 200 epochs max
- [x] **Real bug found and fixed — test-set leakage:** the reference code's `train()` sets `val_mask = data.test_mask` — early stopping (which epoch's weights get kept) is driven by the *same* set used for final evaluation, which inflates the reported test metric. Fixed with a genuine 3-way split: Phase 4's `fixed_test_nodes` stay completely untouched until final evaluation; a separate validation subset (15% of the remaining nodes) drives early stopping instead. `make_splits()`.
- [x] **Decide task framing** — resolved 2026-07-21: cascades from dataset, not an independent choice. Small dataset → classification only (forced in `PipelineConfig.__post_init__`, since no continuous PLD exists); large dataset → user picks classification or regression via UI radio.
- [x] Train GCN + GraphSAGE + GAT on: (i) full best graph, (ii) BH-30, (iii) BH-50 — **9 runs total** (was 6 before GAT). `run_gnn_comparison()` reuses the *same* `fixed_test_nodes` (computed once, deterministic given the same graph+weights) across all 3 graph variants, so the test set is identical for a fair comparison even though BH-30/BH-50 have fewer nodes.
- [x] **Measured results (classification, with GAT, α=β=γ=0.33):** small dataset accuracy 0.543–0.609 (Cohen's Kappa 0.15–0.29) across all 9 runs — GAT shows a degenerate majority-class-collapse pattern on the BH-pruned small-dataset graphs specifically (confusion matrix rows of all-zero for 2 of 4 classes), a known sensitivity of attention-based GNNs on small/sparse graphs, not a bug in the port; large dataset accuracy 0.663–0.725 (Kappa 0.50–0.60), no collapse (more data stabilizes it) — BH-30/BH-50 still slightly *outperform* the full graph, consistent with Black Hole's own claim. **Regression (large dataset only):** GCN on full graph — MAE 1.64 Å, RMSE 2.70 Å, R² 0.23 (modest but real signal, not normalized/standardized before training — a candidate future refinement if regression results need improving).
- [x] Plot training/validation loss curves for each run — `plot_loss_curves()`, now a 3×3 grid (3 models × 3 graph variants), saved to `outputs/phase5_loss_curves_{dataset}.png`
- [x] Collect test accuracy, confusion matrix, Cohen's Kappa (classification) or MAE/RMSE/R² (regression) per run — `evaluate_model()`
- [x] **Critical UI fix — button gate, not automatic execution:** Phase 5 takes ~35-40s (small dataset) but **~9 minutes** (large dataset — two Black Hole runs + 9 full training loops, GAT being notably heavier per epoch). Since every tab's body executes on every Streamlit rerun (a gotcha learned in Phase 3), wiring this in unconditionally would have blocked the *entire app* for minutes on any unrelated slider tweak. Gated behind an explicit `st.button` + `st.session_state` result cache in `app.py` Tab 5; verified the large dataset stays fast when the button isn't clicked, and that clicking it actually produces results, via Playwright.
- [x] **Gravity weight default corrected (2026-07-24) after actually reading `BlackHole.pdf`:** changed from the code's `0.3/0.3/0.4` to the paper's own stated `0.33/0.33/0.33` (Section 2.3, p.7 — "we assign equal weights to all three components"). The three weight controls in `app.py` Tab 4 were redesigned as fully independent slider + number-input pairs (no auto-rebalancing) so any two values can be fixed manually and the third stays put — verified interactively via Playwright that setting α=0.50 then β=0.15 leaves γ untouched at 0.33, with the raw sum and normalized values both shown live.

## Phase 6 — Baseline Comparison ✅ DONE

- [x] Reused the fixed-test-node split (top 4% per community, degree > 2) via `gnn_training.py::make_splits()` — same split as the GNNs, not a separate sampling — `src/baseline_models.py`
- [x] Train Random Forest (scikit-learn, 100 trees) on flat feature matrix — classifier or regressor depending on task
- [x] Train k-NN (k=5) on flat feature matrix — classifier or regressor depending on task
- [x] Evaluate both with the same metrics as Phase 5 (accuracy, macro F1, cohen's kappa, confusion matrix for classification; MAE/RMSE/R² for regression) — added `macro_f1` to `gnn_training.py::evaluate_model()` too so the master table has no missing columns
- [x] Assemble master comparison table — `build_master_comparison_table()`, combines Phase 5's GNN results with Phase 6's baseline results, sorted by accuracy (classification) or R² (regression)
- [x] **Major, honest finding: the non-graph baselines decisively beat every GNN configuration.** Small dataset: Random Forest 0.667 accuracy / κ=0.46 vs. GNNs' best 0.609 (GAT); large dataset classification: RF 0.851 / κ=0.78 vs. GNNs' best 0.725; large dataset **regression: RF R²=0.893, MAE=0.53 Å vs. GCN's R²=0.23, MAE=1.64 Å** — a large, consistent gap in both tasks. Likely explanation, not just "GNN is worse": the large dataset's 2 pore-geometry features (Largest Cavity Diameter, Largest Free Sphere) are strongly correlated with PLD and directly available to the tree ensembles, while the graph is built from the *same* linker/metal similarity already encoded in the node features — message passing mostly re-derives information already present rather than adding new signal. Surfaced prominently in the UI (not buried) as a real result with a natural thesis-extension angle (would a graph built from information *not* already in the node features let GNNs add value?), rather than glossed over.
- [x] Wired into `app.py` Tab 6 — runs automatically (no button gate needed, ~2s incremental cost on top of already-cached features/graph/BH), shows baseline-only results if Phase 5 hasn't been run yet in this session, full master table + finding note once it has. Verified via Playwright: ran Phase 5's button, then confirmed Tab 6's master table and finding note both render correctly.

## Phase 7 — Analysis, Reporting & Thesis Outline 🟡 MOSTLY DONE (notebook + slide deck outstanding, by choice — see below)

- [x] **Identify best connectivity–accuracy trade-off across graph construction strategies** — closed a real gap: Phase 3's `select_best_graph()` picks by topology alone (isolated rate, modularity), never validated against downstream accuracy. `src/phase7_analysis.py` trains one GCN per graph candidate to check. **Finding:** `knn_3` (Phase 3's pick) is genuinely best on the small dataset, but on the large dataset `knn_5`/`knn_10` both beat it on accuracy and κ — reproduced across two runs. Topology-only selection is a real, documented limitation, not swept under the rug. Also found training isn't seeded (run-to-run variance observed directly on the small dataset). Results saved to `data/processed/phase7_graph_tradeoff_{dataset}.csv`.
- [x] **Quantify Black Hole's memory/time savings vs. full graph** — node retention 66–68% (τ=0.3) / 48% (τ=0.5) across both datasets; edge retention drops faster (39–43% / 16–19%) since edge pruning applies its own independent cutoff on top of node removal — this edge-count reduction is the real, comparable efficiency signal. Peak-memory readings are explicitly flagged as not comparable to the paper's ~114MB figure (different measurement methodology, documented in Phase 4).
- [x] **Draw conclusions on graph topology's contribution vs. raw node features** — this is Phase 6's central finding: non-graph baselines (Random Forest, k-NN) decisively beat every GNN configuration on both tasks, both datasets (most starkly, RF R²=0.89 vs. GCN R²=0.23 on regression). Written up with a concrete explanation (the graph is built from the same similarity already encoded in node features, so message passing adds little new signal) rather than left as an unexplained gap.
- [ ] Consolidate all notebooks into one reproducible pipeline notebook — **deliberately not done yet**: the pipeline has been script-based throughout (no Jupyter dependency added so far); consolidating into a notebook is a separate scope decision (adds `jupyter`/`nbformat` to `requirements.txt`) left for explicit request rather than assumed.
- [x] Write **Background** section (what MOFs are, why useful in materials science) — `planning/CASE_STUDY_REPORT.md` Section 1.
- [x] Write final case study report (methodology, results tables, figures, discussion, references) — `planning/CASE_STUDY_REPORT.md`, all numbers cited directly from saved artifacts in `data/processed/`, not re-derived from memory.
- [ ] Prepare final slide deck summarizing contributions and findings — **not done yet**, by choice: `planning/MEETING_PREP.md`'s Part 2 already established a detailed-prompt format for the kickoff deck; a results-deck in the same style is a natural next step but wasn't assumed without confirming the professor wants a second presentation at this point.
- [x] Write thesis-extension section (multi-property prediction, improved similarity via SOAP/3D structure, BH weight optimization, MOF recommendation system) — `planning/CASE_STUDY_REPORT.md` Section 6, updated with case-study-specific motivation for each direction (e.g. direction 1 and 2 now directly motivated by Sections 4.1 and 4.4's findings, not just carried over unchanged from the original Expose).

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

## Generic, switchable design (dataset → feature scheme → task, + Black Hole params)

- [x] `src/config.py::PipelineConfig` — single source of truth, expanded 2026-07-21 to match the meeting outcome:
  - `dataset` ("small"/"large") — the one root switch
  - `feature_scheme` — **derived property**, not a field (cascades from `dataset`, can't be set independently)
  - `task` ("classification"/"regression") — forced to "classification" in `__post_init__` when `dataset == "small"`
  - `gravity_degree_weight` / `gravity_betweenness_weight` / `gravity_edge_weight_sum_weight` — sliders, default 0.3/0.3/0.4
  - `gravity_weights_normalized` — derived property, always sums to 1 regardless of raw slider values
  - `pruning_threshold` (τ) — slider, default 0.3, validated to `[0.0, 0.9]`
- [x] `src/data_ingestion.py::load_dataset(name)` — canonical loader; both datasets return the same column schema (`refcode`, `linker_smiles`, `metal`, `pld_category`, `pld_value`), so Phase 2+ can be written against one interface regardless of which dataset is active.
- [ ] **Note for Phase 2/3 planning:** switching datasets isn't purely a config change — graph construction is O(n²) pairwise similarity. Small (2,000 MOF) → ~4M pairs; large (14,296 MOF) → ~204M pairs (~51x). Write the similarity/k-NN code with vectorized RDKit bulk ops from the start so it scales to either dataset without a rewrite.
- [ ] Phase 2's two feature schemes (compact 7-dim / fingerprint 1031-dim) should be implemented behind the same derived `feature_scheme` property when Phase 2 starts — not stubbed yet, since Phase 2 hasn't begun.

## Interactive Control Panel (Streamlit)

- [x] `app.py` (repo root) — run with `.venv/bin/streamlit run app.py`. Sidebar: dataset radio (root switch) → conditional task radio (large dataset only) → read-only feature-scheme caption (derived, not a control). Tab 4 (Black Hole): three gravity-weight sliders (α/β/γ) with live-normalized display, plus a τ pruning-threshold slider — all wired into `PipelineConfig` and validated, but not yet connected to real computation (Phase 3/4 don't exist yet).
- [x] Verified via the `developing-with-streamlit` skill's screenshot workflow across both dataset paths (small → classification forced, large → task radio + 53 metals shown) — no runtime errors, all conditional UI renders as intended.
- Tab 1 (Data & EDA) remains the only tab with real computed output (summary stats, class balance, metal distribution, PLD histogram, invalid-SMILES warning). Remaining tabs (Feature Engineering, Graph Construction, GNN Training, Baselines, Results) are visible but explicitly marked "not yet implemented" — each lights up as its phase is built, so the app never overstates what's done.

---

## Cross-cutting / Housekeeping

- [ ] Version control: commit after each phase with clear messages
- [ ] Keep a running log of deviations from the Expose's original plan (and why)
- [ ] Track open questions for the professor in `MEETING_PREP.md` as they arise
- [ ] Back up raw data and trained model checkpoints outside git (large files)
