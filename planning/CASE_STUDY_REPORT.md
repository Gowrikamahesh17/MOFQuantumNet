# Graph Neural Networks for MOF Property Prediction

**Case Study 2 — Applied Data Science and Analytics, SRH Hochschule Heidelberg**
Gowrika Mahesh · Supervisor: Prof. Dr. Mehrdad Jalali

---

## 1. Background

Metal-Organic Frameworks (MOFs) are porous crystalline materials built from metal-ion nodes connected by organic linker molecules. Their extraordinarily tunable pore structure makes them valuable for gas storage, carbon capture, catalysis, and separation — but a MOF's key structural property, its **Pore Limiting Diameter (PLD)**, is normally only known after the material has been synthesized or computationally simulated, not from its raw building blocks (the metal and linker alone).

This project asks: can PLD be predicted directly from a MOF's metal and linker identity, using a graph representation of MOF-to-MOF similarity? This directly extends the supervisor's own published work — [MOFGalaxyNet](https://doi.org/10.1186/s13321-023-00764-2) (Jalali et al., 2023), which introduced the graph-based similarity approach, and [the Black Hole Strategy](reference_code/BlackHole.pdf) (Jalali et al., 2025), which introduced gravity-based graph sparsification to make training more frugal without sacrificing accuracy.

## 2. Datasets

Two datasets were found usable in the supplied reference material — notably, the BlackHole repository's README states its data files are "not included," but both were confirmed present locally with real data:

| | Small — `SMILES_METAL_2000_NoPLD.csv` | Large — `MOFCSD.csv` |
|---|---|---|
| MOFs | 2,004 | 14,296 |
| Metal identity | 6 anonymous numeric descriptors only | Real element symbol, 53 unique metals |
| Continuous PLD | Not present | Present (range 0–71.5 Å, mean 3.3 Å) |
| PLD category label | Precomputed (1,062 / 425 / 271 / 246 across nonporous/small/medium/large) | Derived by binning (7,106 / 3,540 / 2,840 / 810) |
| Enables regression? | No | Yes |

Per the professor's 2026-07-21 confirmation, both datasets stayed in scope, selected via a single config switch (`src/config.py`) — feature scheme and available tasks cascade automatically from that one choice rather than being independent settings.

## 3. Methodology summary

The pipeline was implemented across 7 phases, each in its own module under `src/`. An
interactive control panel was built for live exploration of every phase — initially as a
Streamlit application, later rebuilt as a standalone web console (FastAPI backend + a
plain HTML/CSS/JS frontend) once the project's scope grew to include predicting on
genuinely new, never-seen MOFs; see Section 8.

| Phase | What it does |
|---|---|
| 0 — Environment | `venv` + PyTorch Geometric, RDKit, no conda |
| 1 — Data ingestion & EDA | Load, clean, validate both datasets |
| 2 — Feature engineering | Compact 7-dim (small) / fingerprint 1,079-dim (large) node features |
| 3 — Graph construction | Fixed-threshold (φ=0.9) vs. k-NN (k=3,5,10) similarity graphs |
| 4 — Black Hole sparsification | Gravity-based node/edge pruning at configurable τ |
| 5 — GNN training | GCN, GraphSAGE, GAT on the best graph, BH-30, and BH-50 |
| 6 — Baseline comparison | Random Forest, k-NN on the flat feature matrix (no graph) |
| 7 — Analysis & reporting | This document |

All code is written fresh against the professor's papers and repositories as reference material — no files under `reference_code/` were ever modified.

## 4. Results

### 4.1 Graph construction: topology vs. accuracy

Phase 3 selected a "best" graph using topology alone (lowest isolated-node rate, then highest modularity) — `knn_3` won on both datasets. Phase 7 tested whether that heuristic also produces the best **downstream accuracy**, by training one GCN per candidate graph:

**Small dataset (2,004 MOFs):**

| Config | Accuracy | Cohen's κ | Isolated rate | Mean degree | Modularity |
|---|---|---|---|---|---|
| threshold_0.9 | 0.464 | 0.176 | 20.5% | 14.87 | 0.832 |
| **knn_3** (Phase 3 pick) | **0.572** | **0.234** | 0.0% | 4.25 | 0.977 |
| knn_5 | 0.554 | 0.201 | 0.0% | 7.07 | 0.955 |
| knn_10 | 0.567 | 0.212 | 0.0% | 14.11 | 0.911 |

**Large dataset (14,296 MOFs):**

| Config | Accuracy | Cohen's κ | Isolated rate | Mean degree | Modularity |
|---|---|---|---|---|---|
| threshold_0.9 | 0.697 | 0.530 | 11.3% | 88.90 | 0.793 |
| **knn_3** (Phase 3 pick) | 0.658 | 0.496 | 0.0% | 4.63 | 0.986 |
| knn_5 | 0.697 | 0.549 | 0.0% | 7.60 | 0.973 |
| knn_10 | **0.698** | **0.541** | 0.0% | 15.08 | 0.954 |

**Finding:** on the small dataset, `knn_3` genuinely is the best choice — both highest accuracy and highest κ. On the **large dataset, it is not** — `knn_5` and `knn_10` both beat it on accuracy and κ by a small but consistent margin (reproduced across two separate training runs). This is a real limitation of topology-only graph selection: modularity and isolated-node rate are good proxies for *some* datasets but not guaranteed to correlate with downstream task performance in general. In both datasets, though, every k-NN configuration clearly beats the fixed threshold on the reliability front — the threshold graph's isolated nodes (11–20% of all MOFs) are a real structural weakness the professor's own paper also notes.

*Reproducibility note:* model initialization is not seeded, so single-run numbers carry some run-to-run variance (observed directly: a repeat run put small-dataset `knn_10`'s κ anywhere between 0.00 and 0.21). The qualitative pattern above (knn_3 best for small, knn_5/knn_10 better for large) held across repeats; exact values will shift slightly on any given run.

### 4.2 Black Hole sparsification: memory and connectivity trade-off

| | Small (τ=0.3) | Small (τ=0.5) | Large (τ=0.3) | Large (τ=0.5) |
|---|---|---|---|---|
| Nodes retained | 66.3% | 47.9% | 67.9% | 48.2% |
| Edges retained | 38.6% | 16.1% | 42.5% | 18.5% |
| Isolated nodes after pruning | 95 | 219 | 924 | 1,716 |
| Graph density | 0.00212 → 0.00186 | 0.00212 → 0.00150 | 0.000324 → 0.000298 | 0.000324 → 0.000258 |

Edge count drops considerably faster than node count at both thresholds — e.g. at τ=0.3, ~34% of nodes are removed but ~61% of edges are, since `prune_edges()` applies its own independent weight-based cutoff on top of node removal. This directly reduces the adjacency structure a GNN has to aggregate over, which is the mechanism behind Black Hole's claimed training-cost savings.

*On memory specifically:* peak RSS measured during our runs (≈400 MB small, ≈1.3–1.8 GB large) is a whole-Python-process snapshot (includes PyTorch, RDKit, and everything else already resident), not an isolated per-step measurement — **not directly comparable** to the paper's reported ~114 MB figure, which was measured differently. The genuinely comparable, dataset-independent signal is the node/edge retention percentages above.

### 4.3 GNN training results

Trained GCN, GraphSAGE, and GAT on the Phase-3 best graph (`knn_3`), BH-30, and BH-50 — 9 runs per dataset/task.

**Classification** (accuracy / Cohen's κ):

| | Small dataset | Large dataset |
|---|---|---|
| Best GNN config | GAT, full graph — 0.601 / 0.291 | GCN, BH-30 — 0.725 / 0.595 |
| Range across all 9 runs | 0.543–0.609 / 0.15–0.29 | 0.663–0.725 / 0.50–0.60 |

**Regression** (large dataset only): GCN on full graph — MAE 1.64 Å, RMSE 2.70 Å, R² 0.23.

A real bug in the reference training loop was found and fixed here: `graphsage_model.py`'s `train()` sets `val_mask = data.test_mask`, meaning early stopping is driven by the same set used for final evaluation — test-set leakage. Fixed with a genuine 3-way split (fixed test nodes untouched until final evaluation; a separate validation split drives early stopping instead).

### 4.4 Baseline comparison — the most important result

Random Forest and k-NN, trained on the **same flat feature matrix with no graph structure at all**, evaluated on the identical fixed test set:

| | Small dataset (accuracy / κ) | Large dataset classification (accuracy / κ) | Large dataset regression (R² / MAE) |
|---|---|---|---|
| Random Forest | **0.667 / 0.465** | **0.851 / 0.780** | **0.893 / 0.53 Å** |
| k-NN | 0.638 / 0.381 | 0.813 / 0.724 | 0.851 / 0.61 Å |
| Best GNN (any config) | 0.601 / 0.291 | 0.725 / 0.595 | 0.23 / 1.64 Å |

**The non-graph baselines decisively beat every GNN configuration, on both tasks, on both datasets.** This is the single most important empirical result in this case study, and it is presented here as a genuine finding rather than something to explain away.

**Why this likely happens:** the large dataset's feature vector includes two pore-geometry descriptors (Largest Cavity Diameter, Largest Free Sphere) that are strongly physically correlated with PLD — exactly the kind of tabular signal tree ensembles are very good at exploiting directly. Meanwhile, the graph itself is constructed from the *same* linker/metal similarity that is already encoded in each node's own feature vector — so a GNN's message-passing step is largely re-deriving information the model already has access to locally, rather than injecting genuinely new relational signal. This suggests the specific similarity-based graph construction used here (following MOFGalaxyNet's own approach) does not add predictive value beyond the node features alone, for this particular target property.

### 4.5 Small dataset vs. the published MOFGalaxyNet number — a 30-point gap, investigated

The MOFGalaxyNet paper (`reference_code/s13321-023-00764-2.pdf`) reports **89.62% accuracy at φ=0.9** on this same 2,000-MOF dataset (Table 4 / Fig. 12) — well above our best small-dataset GNN result (GAT, 60.1%, §4.3). Rather than treat this as an implementation gap, the paper and its reference code were re-read directly to find the source of the difference. Four concrete, verifiable causes were found, none of which are bugs in this codebase:

1. **The paper's own headline number doesn't match its own in-text number for the identical setting.** Immediately after Fig. 10 (the φ=0.9 loss/accuracy curves), the text states *"the overall accuracy percentage achieved is 65.17%"* — but the comparison table three paragraphs later (Table 4, Fig. 12) reports **89.62%** for the same φ=0.9 configuration. A 24-point internal discrepancy for one setting, within one paper, means 89.62% is very unlikely to be a plain held-out test accuracy computed the way this codebase computes one.
2. **Test-set leakage in early stopping, confirmed by reading the actual reference code.** `graphsage_model.py:79` (BlackHole repo, the codebase this project builds on) sets `val_mask = data.test_mask` — the early-stopping criterion that decides when training halts is evaluated on the *same* nodes used for final reporting. This project fixed exactly this bug (§4.3, `src/gnn_training.py::make_splits()`), which alone would be expected to lower the honestly-measured number relative to a leaky baseline.
3. **The MOFGalaxyNet repo's own training script doesn't clearly match the paper's dataset.** `reference_code/MOFGalaxyNet-main/GCN1.py` — the only GCN training script present in that repo — loads `Data/EdgesList.csv` and `Data/content.csv` with column names `paper_id`, `term_0..5`, `subject`: this is Keras' official Cora citation-network GCN tutorial with variable names left unchanged, not a MOF-specific script. It's plausible this is a leftover exploratory/template file rather than the actual code that produced Table 4's numbers — the repo doesn't contain an unambiguous final script to audit, so the 89.62% figure can't be independently reproduced from what's provided.
4. **No repeated-run averaging on the small dataset.** The paper text was searched directly for evidence of multi-seed averaging (the kind BlackHole's own later paper does — see below); the only repeated "runs" mentioned for MOFGalaxyNet are three separate threshold values tested (φ=0.2, 0.7, 0.9), not repeated seeds of the same configuration. So 89.62% is a single run's result, not a mean — and combined with point 2, plausibly the single best epoch of a leaky early-stopping run rather than a stable, reproducible number. This project's own results are also single-run (see Limitations below), so this specific gap isn't a "them-vs-us" methodology difference — but it does mean neither number should be read as more reproducible than the other.

Separately, on the note about run averaging: the later **Black Hole** paper (large-dataset methodology) does explicitly average — *"All experiments were repeated for R = 10 bootstrap runs per sparsification level... Each value represents the average over 10 independent runs"* (`BlackHole.pdf`, §3.5). However, the actual shipped code (`reference_code/BlackHole-main/main.py:48`) hardcodes `num_runs = 4`, not 10 — a real discrepancy between the paper's stated methodology and its own released code, independent of anything in this project.

**Bottom line:** the 89.62% figure is not a number this pipeline could or should try to match. It most likely reflects (a) test-set leakage inside early stopping and (b) a single favorably-timed run/epoch, possibly (c) evaluated by a script that isn't even clearly MOF-specific. The 54–61% range this pipeline reports on the small dataset comes from a genuine 3-way split with the test set held out from both training and early stopping — a stricter and more trustworthy number, at the cost of looking worse on paper.

### 4.6 Large dataset vs. the published Black Hole numbers — a smaller but real gap, root-caused

Our large-dataset GNN accuracy (0.663–0.725, best GCN/BH-30) sits below the Black Hole paper's published Table 2 figures (e.g. full graph: GAT 0.781, **GCN 0.833**, GraphSAGE 0.780; BH-30: GAT 0.805, GCN 0.783, GraphSAGE 0.798 — `results_aggregated.csv`, confirmed to match `BlackHole.pdf`'s Table 2/Fig. 8 numbers exactly). This gap has two concrete, verified causes — one of them decisive:

1. **Direct target leakage in the reference feature pipeline — confirmed by reading the actual code.** `reference_code/BlackHole-main/data_utils.py::load_summary_data()`, lines 89–93 and 106, builds each node's feature vector as:
   ```python
   other_features = np.array([
       row['Pore Limiting Diameter'],      # <- the regression target itself
       row['Largest Cavity Diameter'],
       row['Largest Free Sphere']
   ], dtype=np.float32)
   ...
   feature = np.concatenate([fp, other_features, metal_one_hot])   # PLD is now x[:, 1024]
   ```
   The raw, continuous Pore Limiting Diameter — exactly what both the classification label (`category`) and the regression target are derived from — is fed into the model as an input feature. For classification this reduces most of the task to memorizing which bin edge the (visible) PLD value falls into; for regression, the target is *literally present in the input*, i.e. closer to fitting the identity function than to learning structure-property relationships. This is precisely the leakage bug this project's `src/feature_engineering.py` already identifies and excludes (README, "Key implementation decisions"), and it plausibly explains the bulk of the gap — it's consistent with the scale of the regression gap in particular: their GAT-BH-50 R² of 0.68 (`BlackHole.pdf`, §3.5) vs. this project's leakage-free GCN R² of 0.23 (§4.3) is a far larger relative gap than the classification numbers, exactly what you'd expect if the target were hiding in the features for regression but only partially informative (post-binning) for classification.
2. **Substantially different, sparser graph.** Phase 3's topology-based selection (`data/processed/phase3_topology_large.csv`) picked `knn_3` — 14,296 nodes, 33,084 edges, **mean degree 4.6** — over the denser `threshold_0.9` candidate (635,429 edges, mean degree 88.9) because `knn_3` has 0% isolated nodes and the highest modularity (0.986). The paper's own graph (built from `MOFGalaxyNet.csv`, not reconstructed here) has ~10,230–11,230 nodes and **196,771–297,632 edges — mean degree 35–56**, roughly 8–12× denser than `knn_3` even before any Black Hole pruning is applied. A GNN message-passing over a mean-degree-4.6 graph simply has far less neighbor signal to aggregate per node than one over a mean-degree-35+ graph, independent of any leakage. (Phase 7's own graph-tradeoff check, §4.1, already found `knn_3` isn't even the best-*performing* k-NN choice on the large dataset — `knn_5`/`knn_10` beat it — so some of this gap is addressable within this project's existing graph candidates, though none reach the reference graph's density.)

**A third observation, not a root cause but relevant context:** the BlackHole repo itself contains four different aggregated-results CSVs (`aggregated_results.csv`, `aggregated_results3.csv`, `results_aggregated.csv`, `results_aggregated3.csv`) that report wildly different accuracy for the *identical* threshold-0.0 (no pruning) configuration — 0.605–0.623, 0.611–0.618, **0.781–0.833**, and 0.531–0.552 respectively, a roughly 30-point spread across what should be the same experiment. Only `results_aggregated.csv` matches the paper's published Table 2. No `torch.manual_seed` call exists anywhere in the reference training code (`graphsage_model.py`, `main.py` — verified by search), so model initialization and dropout are unseeded across runs; combined with these files likely being snapshots from different points in the project's iteration (possibly before/after the leakage bug above was present in different forms), the published number should be read as one run among a large observed spread, not a tightly reproducible baseline.

**Bottom line:** unlike the small-dataset gap (mostly a methodology/reporting question), the large-dataset gap has one clearly dominant, mechanical cause — the reference pipeline trains on a feature vector that contains its own prediction target — plus a secondary, addressable cause (this project's selected graph is far sparser than the reference paper's). Removing the leak was a deliberate, correct choice (§2, README); the honest expectation is that any leakage-free reproduction would land well below 0.78–0.83, in the range this project reports.

## 5. Web console — Predict + Lab

Beyond the analysis above, the project delivers a working application built on the same
pipeline: a single FastAPI service (`webapp/backend/`) exposing both a JSON API and a
static frontend (`webapp/frontend/`), with two surfaces.

**Predict** accepts a metal and a linker SMILES string — an existing MOF or a
hypothetical one — and returns a leaderboard of every trained model's prediction, ranked
by that model's own measured test accuracy (Section 4.4's numbers, computed live rather
than hardcoded). Serving GCN, GraphSAGE, and GAT on a MOF outside the training set is not
trivial: these models were trained transductively over a fixed similarity graph, so a new
node has no place in it. This is resolved by inductive insertion — the new MOF's
similarity to every existing node is computed with the exact formula from Section 3
(`ALPHA`-weighted linker Tanimoto + metal similarity), it is attached to its nearest
surviving neighbors in the relevant graph variant, and one forward pass is run over the
augmented graph. Every graph-model response reports the number of neighbors used and the
graph variant, so the mechanism is inspectable rather than a black box. Two further gaps
are handled the same way they are analyzed in this report — honestly, not silently: the
small dataset has no metal-identity column at all (Section 2), so predictions on it fall
back to dataset-median descriptor values and flag this explicitly; a hypothetical MOF on
the large dataset likewise has no simulated pore geometry, so those two features are
median-imputed and flagged, rather than presented as equivalent to the benchmarked
accuracy.

**Lab** reproduces all 7 pipeline phases as live, backend-computed views rather than
static figures, plus a retrain control: the three gravity weights and the pruning
threshold can be adjusted and a real training job (baselines and/or GNNs) is queued in the
background and polled to completion, so the sensitivity analysis in Sections 4.1–4.2 can
be reproduced interactively rather than only by re-running scripts.

The console's own test suite (`tests/webapp/`, 26 tests) independently re-verifies this
report's central claim on live data: for a genuinely new, never-seen linker SMILES, the
ranked leaderboard places Random Forest above every GNN configuration on both datasets —
the same ordering as Section 4.4, reproduced outside the original analysis code path.

## 6. Limitations

- **Single-run results, not averaged.** The Black Hole paper reports means over 10 bootstrap runs per configuration (variance <1% per their own analysis) — though note the paper's own shipped code (`main.py`) hardcodes `num_runs = 4`, not 10, a discrepancy between the paper's stated methodology and its release (§4.5). Our results are from single runs either way, so exact numbers carry more noise, though qualitative patterns (e.g. BH-30 outperforming the full graph) were consistent across the repeats we did perform.
- **Evaluation set differs from the paper's.** The paper restricts evaluation to each graph's largest connected component; we evaluate on the full fixed test set, including any nodes left isolated by pruning — a harder, more conservative standard.
- **No target/gravity-weight hyperparameter search.** Model architecture (hidden dim 64, dropout 0.3), optimizer settings, and gravity weights were taken from the reference code/paper rather than independently tuned for this dataset.
- **Regression target not normalized.** Continuous PLD spans 0–71.5 Å with a long right tail; standardizing the target before training is a plausible improvement not yet tried.
- **GAT shows instability on small, sparse graphs.** Observed degenerate majority-class collapse in some small-dataset/BH-pruned runs — a known sensitivity of attention-based GNNs to graph density and hyperparameters, not unique to this implementation.

## 7. Thesis extension directions

Building on this case study's findings, four directions were identified (the first two directly motivated by Section 4.4's finding):

1. **Alternative graph construction** — since similarity-based edges duplicate information already in the node features, a graph built from information the node features *don't* already contain (e.g. 3D structural descriptors via SOAP kernels, from CoRE MOF 2019 or QMOF) might let a GNN add genuine value over flat baselines, unlike the construction used here.
2. **Task-aware graph selection** — Section 4.1 showed topology-only graph selection (modularity, isolation rate) doesn't reliably pick the best-*performing* graph. A selection criterion that incorporates a quick downstream accuracy check, not just structural metrics, is a natural refinement.
3. **Black Hole gravity-weight optimization** — systematically search the (α, β, γ) space (now a first-class configurable feature of this pipeline) for configurations that maximize accuracy on specific PLD classes, rather than the equal-weighting default.
4. **Multi-property prediction & MOF recommendation** — extend to multiple simultaneous target properties (surface area, pore volume, CO₂ uptake) via multi-task GNN heads, and explore using the community structure to recommend structurally similar or functionally complementary MOFs for a target application.

## 8. References

- Jalali, M., Wonanke, A.D.D., & Wöll, C. (2023). MOFGalaxyNet: A social network analysis for predicting guest accessibility in metal-organic frameworks utilizing graph convolutional networks. *Journal of Cheminformatics*, 15, 94.
- Jalali, M., Wonanke, A.D.D., Friederich, P., & Wöll, C. (2025). The Black Hole Strategy: Gravity-Based Representative Sampling for Frugal Graph Learning on Metal-Organic Framework Networks. *Journal of Chemical Information and Modeling*, 65(20).
- Kipf, T.N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.
- Hamilton, W., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. NeurIPS 2017.
- Fey, M., & Lenssen, J.E. (2019). Fast Graph Representation Learning with PyTorch Geometric.
- Full phase-by-phase development log, all findings, and every code-level decision: [planning/DEVELOPMENT_TODO.md](DEVELOPMENT_TODO.md).
