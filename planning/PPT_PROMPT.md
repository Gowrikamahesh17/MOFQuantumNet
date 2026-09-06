# MOFQuantumNet — Presentation Slide Prompts

Full-project academic presentation, generated from [`CASE_STUDY_REPORT.md`](CASE_STUDY_REPORT.md).
Each block is a standalone prompt for a slide-generation tool. Style: minimal on-slide
text (facts, numbers, labels — not paragraphs), academic register, one idea per slide.

---

### Slide 1 — Title
```
Title: "Graph Neural Networks for MOF Property Prediction"
Subtitle: Case Study 2 — Applied Data Science and Analytics, SRH Hochschule Heidelberg
Author: Gowrika Mahesh. Supervisor: Prof. Dr. Mehrdad Jalali.
Visual: minimal, a single MOF crystal-lattice motif. No decoration beyond this.
```

### Slide 2 — Problem Statement
```
Title: "Problem Statement"
Bullets only:
- Pore Limiting Diameter (PLD) governs guest accessibility in Metal-Organic Frameworks
- PLD is known only after synthesis or simulation — not from metal + linker identity alone
- Question: can PLD be predicted directly from a MOF's building blocks via a similarity graph?
Visual: none needed.
```

### Slide 3 — Related Work
```
Title: "Related Work"
Table, two rows:
- Jalali et al. (2023), MOFGalaxyNet — graph-based MOF similarity, GCN classification
- Jalali et al. (2025), Black Hole Strategy — gravity-based graph sparsification
Line: This project reimplements and extends both in PyTorch Geometric.
Visual: none.
```

### Slide 4 — Datasets
```
Title: "Datasets"
Table:
              | Small (2,004 MOFs) | Large (14,296 MOFs)
Metal identity | 6 anonymous descriptors | 53 real element symbols
Continuous PLD | absent | present (0–71.5 Å)
Task           | classification only | classification or regression
Visual: table only.
```

### Slide 5 — Methodology Overview
```
Title: "Pipeline Overview"
7 numbered stages, labels only:
1. Data ingestion  2. Feature engineering  3. Graph construction
4. Black Hole sparsification  5. GNN training  6. Baseline comparison  7. Analysis
Visual: horizontal flow diagram, 7 boxes, arrows.
```

### Slide 6 — Feature Engineering
```
Title: "Feature Engineering"
Bullets:
- Small dataset: 7-dim (6 metal descriptors + linker molecular weight)
- Large dataset: 1,079-dim (1,024-bit Morgan fingerprint + 2 geometry features + 53-way metal one-hot)
- Target leakage identified and removed: reference code included raw PLD as a feature
Visual: none.
```

### Slide 7 — Graph Construction
```
Title: "Graph Construction"
Bullets:
- 4 candidate graphs per dataset: fixed threshold (φ=0.9), k-NN (k=3,5,10)
- Selection rule: lowest isolated-node rate, then highest modularity
- Threshold graph leaves 11–20% of nodes isolated; every k-NN variant guarantees 0%
Visual: small node-link diagram, dense vs. sparse graph side by side.
```

### Slide 8 — Black Hole Sparsification
```
Title: "Black Hole Sparsification"
Bullets:
- Gravity score = degree centrality + betweenness centrality + edge-weight sum
- PLD-stratified node/edge pruning at threshold τ
- τ=0.3: ~67% nodes / ~40% edges retained. τ=0.5: ~48% nodes / ~17% edges retained
Visual: simple before/after node-count bar comparison.
```

### Slide 9 — GNN Training
```
Title: "GNN Training"
Bullets:
- Architectures: GCN, GraphSAGE, GAT
- Each trained on 3 graph variants: full graph, BH-30, BH-50 (9 runs per dataset/task)
- Bug found and fixed in reference code: early stopping leaked the test set into validation
- Fixed with a genuine 3-way split
Visual: none.
```

### Slide 10 — Baseline Models
```
Title: "Baseline Models"
Bullets:
- Random Forest and k-NN, trained on the same flat feature vectors, no graph
- Identical fixed test set as the GNNs — direct comparison
Visual: none.
```

### Slide 11 — Result: Graph Selection vs. Downstream Accuracy
```
Title: "Topology-Based Selection Is Not Always Accuracy-Optimal"
Table:
              | Small dataset | Large dataset
Phase-3 pick (knn_3) | best (0.572 acc) | not best
Best on large        | —                | knn_10 (0.698 acc)
Line: Modularity/isolation rate are not guaranteed proxies for downstream performance.
Visual: table only.
```

### Slide 12 — Result: Non-Graph Baselines Outperform GNNs
```
Title: "Headline Finding: Non-Graph Baselines Outperform Every GNN Configuration"
Table (accuracy / R²):
              | Small (acc) | Large (acc) | Large (R², regression)
Random Forest | 0.667       | 0.851       | 0.893
Best GNN      | 0.601       | 0.725       | 0.23
Visual: grouped bar chart from this table, two colors (baseline vs. GNN).
```

### Slide 13 — Why Baselines Win
```
Title: "Why the Graph Adds No Signal Here"
Bullets:
- Large-dataset features already include 2 pore-geometry descriptors correlated with PLD
- The similarity graph is built from the same linker/metal features already in each node
- Message passing re-derives existing information rather than adding new relational signal
Visual: none.
```

### Slide 14 — Validation Against Published Numbers
```
Title: "Reconciling Results With the Source Papers"
Bullets:
- MOFGalaxyNet reports 89.62% on the small dataset; this project reports 60.1%
- Reference paper's own text is internally inconsistent (65.17% vs. 89.62% for the same setting)
- Reference training code confirmed to leak the test set into early stopping
- Black Hole's large-dataset feature pipeline confirmed to include the target (PLD) as an input feature
Line: Both gaps are root-caused to the reference implementations, not this project's code.
Visual: none.
```

### Slide 15 — Web Console: Predict
```
Title: "Web Console — Predict"
Bullets:
- FastAPI backend + browser frontend, single process
- Describe any MOF (metal + linker SMILES) — real or hypothetical
- Every trained model answers, ranked by its own measured test accuracy
- New MOFs are inserted into each GNN's graph via nearest chemical neighbors before predicting
Visual: screenshot or mockup of the ranked leaderboard.
```

### Slide 16 — Web Console: Lab
```
Title: "Web Console — Lab"
Bullets:
- All 7 pipeline phases reproduced as live, backend-computed views
- Retrain control: adjust gravity weights and pruning threshold, retrain in the background
- Same findings reproduced live: baselines rank above GNNs on real, never-seen input
Visual: screenshot or mockup of the Lab view.
```

### Slide 17 — Testing & Quality Assurance
```
Title: "Testing & Quality Assurance"
Table:
Layer          | Count
Unit           | 187
Property-based | 11 groups
Torture        | 26
Acceptance     | 19
Web console    | 26
Total          | 269, all passing
Line: Mutation testing (config.py, black_hole_sparsification.py): 84.4% mutant-kill rate.
Visual: table only.
```

### Slide 18 — Limitations
```
Title: "Limitations"
Bullets:
- Single-run results; training is not seeded
- No hyperparameter search on architecture or gravity weights
- Regression target (PLD) not normalized before training
- GAT shows instability on small, sparse graphs
Visual: none.
```

### Slide 19 — Thesis Extension Directions
```
Title: "Future Work"
Bullets, numbered:
1. Graph construction from information not already in node features (e.g. 3D SOAP descriptors)
2. Accuracy-aware graph selection, not topology alone
3. Systematic Black Hole gravity-weight optimization
4. Multi-property prediction and MOF recommendation
Visual: none.
```

### Slide 20 — Conclusion
```
Title: "Conclusion"
Bullets:
- Full pipeline reimplemented in PyTorch Geometric, generalized across two datasets
- Two real bugs identified and fixed in the reference implementations
- Central finding: non-graph baselines outperform every GNN configuration tested
- Delivered as a working, tested web application, not only a research script
Visual: none.
```

### Slide 21 — References
```
Title: "References"
List (as in CASE_STUDY_REPORT.md Section 8):
- Jalali, Wonanke, Wöll (2023). MOFGalaxyNet. Journal of Cheminformatics, 15, 94.
- Jalali, Wonanke, Friederich, Wöll (2025). The Black Hole Strategy. J. Chem. Inf. Model., 65(20).
- Kipf & Welling (2017). Semi-Supervised Classification with GCNs. ICLR.
- Hamilton, Ying, Leskovec (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
- Fey & Lenssen (2019). Fast Graph Representation Learning with PyTorch Geometric.
Visual: none.
```

### Slide 22 — Closing
```
Title: "Thank You"
Line: Questions and discussion welcome.
Visual: match Slide 1's style.
```
