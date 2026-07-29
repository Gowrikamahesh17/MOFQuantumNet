# Graph Neural Networks for MOF Property Prediction

**Case Study 2 — Applied Data Science and Analytics, SRH Hochschule Heidelberg**
Gowrika Mahesh · Supervisor: Prof. Dr. Mehrdad Jalali

---

## 1. Background

Metal-Organic Frameworks (MOFs) are porous crystalline materials built from metal-ion nodes connected by organic linker molecules. Their extraordinarily tunable pore structure makes them valuable for gas storage, carbon capture, catalysis, and separation — but a MOF's key structural property, its **Pore Limiting Diameter (PLD)**, is normally only known after the material has been synthesized or computationally simulated, not from its raw building blocks (the metal and linker alone).

This project asks: can PLD be predicted directly from a MOF's metal and linker identity, using a graph representation of MOF-to-MOF similarity? This directly extends the supervisor's own published work — [MOFGalaxyNet](https://doi.org/10.1186/s13321-023-00764-2) (Jalali et al., 2023), which introduced the graph-based similarity approach, and [the Black Hole Strategy](reference_content/BlackHole.pdf) (Jalali et al., 2025), which introduced gravity-based graph sparsification to make training more frugal without sacrificing accuracy.

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

The pipeline was implemented across 7 phases, each in its own module under `src/`, with an interactive Streamlit control panel (`app.py`) for live exploration:

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

All code is written fresh against the professor's papers and repositories as reference material — no files under `reference_content/` were ever modified.

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

## 5. Limitations

- **Single-run results, not averaged.** The Black Hole paper reports means over 10 bootstrap runs per configuration (variance <1% per their own analysis); our results are from single runs, so exact numbers carry more noise, though qualitative patterns (e.g. BH-30 outperforming the full graph) were consistent across the repeats we did perform.
- **Evaluation set differs from the paper's.** The paper restricts evaluation to each graph's largest connected component; we evaluate on the full fixed test set, including any nodes left isolated by pruning — a harder, more conservative standard.
- **No target/gravity-weight hyperparameter search.** Model architecture (hidden dim 64, dropout 0.3), optimizer settings, and gravity weights were taken from the reference code/paper rather than independently tuned for this dataset.
- **Regression target not normalized.** Continuous PLD spans 0–71.5 Å with a long right tail; standardizing the target before training is a plausible improvement not yet tried.
- **GAT shows instability on small, sparse graphs.** Observed degenerate majority-class collapse in some small-dataset/BH-pruned runs — a known sensitivity of attention-based GNNs to graph density and hyperparameters, not unique to this implementation.

## 6. Thesis extension directions

Building on this case study's findings, four directions were identified (the first two directly motivated by Section 4.4's finding):

1. **Alternative graph construction** — since similarity-based edges duplicate information already in the node features, a graph built from information the node features *don't* already contain (e.g. 3D structural descriptors via SOAP kernels, from CoRE MOF 2019 or QMOF) might let a GNN add genuine value over flat baselines, unlike the construction used here.
2. **Task-aware graph selection** — Section 4.1 showed topology-only graph selection (modularity, isolation rate) doesn't reliably pick the best-*performing* graph. A selection criterion that incorporates a quick downstream accuracy check, not just structural metrics, is a natural refinement.
3. **Black Hole gravity-weight optimization** — systematically search the (α, β, γ) space (now a first-class configurable feature of this pipeline) for configurations that maximize accuracy on specific PLD classes, rather than the equal-weighting default.
4. **Multi-property prediction & MOF recommendation** — extend to multiple simultaneous target properties (surface area, pore volume, CO₂ uptake) via multi-task GNN heads, and explore using the community structure to recommend structurally similar or functionally complementary MOFs for a target application.

## 7. References

- Jalali, M., Wonanke, A.D.D., & Wöll, C. (2023). MOFGalaxyNet: A social network analysis for predicting guest accessibility in metal-organic frameworks utilizing graph convolutional networks. *Journal of Cheminformatics*, 15, 94.
- Jalali, M., Wonanke, A.D.D., Friederich, P., & Wöll, C. (2025). The Black Hole Strategy: Gravity-Based Representative Sampling for Frugal Graph Learning on Metal-Organic Framework Networks. *Journal of Chemical Information and Modeling*, 65(20).
- Kipf, T.N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.
- Hamilton, W., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. NeurIPS 2017.
- Fey, M., & Lenssen, J.E. (2019). Fast Graph Representation Learning with PyTorch Geometric.
- Full phase-by-phase development log, all findings, and every code-level decision: [planning/DEVELOPMENT_TODO.md](DEVELOPMENT_TODO.md).
