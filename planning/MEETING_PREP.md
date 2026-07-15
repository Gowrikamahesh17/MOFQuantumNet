# First Meeting Prep — Case Study 2 Kickoff with Prof. Dr. Jalali

Main agenda: show Prof. Jalali (1) what already exists in his prior work, (2) what I plan to build on top of it, and (3) surface open points before I start coding.

---

## Part 1 — Points to Discuss / Clarify

Use these to structure the conversation and to make clear I've read his code, not just his papers.

### A. Confirm scope framing
- [ ] Confirm the topic title stands as **"Graph Neural Networks for MOF Property Prediction"** and that PLD (pore-limiting diameter) is the right target property to keep, given it's the one both `MOFGalaxyNet` and `BlackHole` already model.
- [x] ~~Confirm the 2,000-MOF CSD dataset from the `MOFGalaxyNet` repo is acceptable as the primary dataset~~ — **resolved by EDA, see B below.**

### B. Dataset & feature vector choice — decision made, not blocking (updated after Phase 1 EDA)
- His email already granted flexibility here ("you can use my dataset as well in Blackhole or MOFgalaxynet repository"), so this is no longer a hard blocker — but worth surfacing what I found and chose.
- **New finding:** his BlackHole README states `MOFGalaxyNet.csv`/`MOFCSD.csv` are "not included," but both are actually present locally with real data: `MOFCSD.csv` has **14,296 MOFs** with genuine continuous PLD, real metal names (53 unique), and linker SMILES — zero NaNs. So there are two complete, usable datasets, not one:
  - **Small set** (`SMILES_METAL_2000_NoPLD.csv`, 2,000 MOFs): 7-dim-style compact metal descriptor + SMILES, precomputed PLD category label only (no continuous PLD). I reproduced its category histogram almost exactly (1062/425/271/246 vs. paper's 1062/422/271/244).
  - **Large set** (`MOFCSD.csv`, 14,296 MOFs): richer, continuous PLD (supports regression too), real metal names, matches what `BlackHole/data_utils.py` expects (1031-dim Morgan-fingerprint feature scheme).
- **Decision:** Starting development on the small set (faster iteration, matches the paper's exact published numbers), but built the pipeline so dataset and feature scheme are both one-line config switches (`src/config.py`), not a rewrite — see `DEVELOPMENT_TODO.md`'s "Generic, switchable design" section. Will switch to the large set later if he'd prefer, without losing any work.
- **Related finding worth mentioning:** `data_utils.py`'s metal one-hot only handles 4 metals (Cu/Zn/Fe/Co), defaulting everything else to Cu — but the real data has 53 unique metals. This will need generalizing (e.g. one-hot over top-N metals + "other") if/when we switch to the large dataset.

### C. Black Hole gravity weights (discrepancy I found)
- My Expose text (drafted from the papers) states equal weights α = β = γ = 0.33 for the gravity score.
- The actual `bh_sparsification.py` code default is `weights=(0.3, 0.3, 0.4)` (degree centrality, betweenness centrality, edge-weight-sum).
- **Question:** Which is correct / intended for reproduction — should I use the code's default, or is 0.33/0.33/0.33 an updated value from somewhere else?

### D. Classification vs. regression framing
- `BlackHole/main.py` supports both a 4-class PLD classification task and a continuous PLD regression task via a `--task` flag.
- **Question:** Does he want me to run both (as his codebase already supports), or focus the case study on just classification (matching the original `MOFGalaxyNet` paper) and leave regression as a stretch goal / thesis-extension item?

### E. Reused vs. novel code — expectation setting
- Confirm it's acceptable to **directly reuse/port** his `GCN`, `GraphSAGE` model classes and `train()`/`test()` functions from `BlackHole/graphsage_model.py`, since the technical contribution is the PyG reimplementation + graph-construction experiment + baseline comparison, not reinventing model architectures.
- Confirm the case study should **not** touch GAT (present in his code) unless he wants a 3-model comparison instead of 2.

### F. Compute / dataset size sanity check
- Confirm expected runtime is reasonable for a case study (his own README notes ~10 minutes per run at 14,000-MOF scale, on CPU) — at 2,000 MOFs this should be faster, but want to flag in case he expects GPU access or a compute budget conversation.

### G. Report format
- Confirm expected deliverable format: Jupyter notebook + written report + slide deck (as I've assumed), or if SRH has a specific report template he wants me to use.

---

## Part 2 — PowerPoint Presentation Prompts (7–8 slides)

Detailed content prompts for each slide — write these out as the actual slide content, or feed each block to a slide-generation tool. Tone: confident but appropriately humble ("here's my plan, here's what I'm still deciding, here's what I need from you").

---

### Slide 1 — Title
**Prompt:** Create a clean title slide.
- Title: *Graph Neural Networks for MOF Property Prediction*
- Subtitle: Case Study 2 — Kickoff & Approach Review
- Your name, student ID, program (Applied Data Science and Analytics), supervisor (Prof. Dr. Mehrdad Jalali), date
- Visual: one clean MOF network/graph image (reuse `MOFGalaxyNet.png` or `Galaxy4.png` from the reference repo) as a background accent

### Slide 2 — Why MOFs, Why This Problem
**Prompt:** One slide explaining the motivation, written for someone already expert (keep it brief, framing not teaching).
- MOFs = porous crystalline materials (metal nodes + organic linkers), tunable for gas storage, carbon capture, catalysis
- Bottleneck: pore-limiting diameter (PLD) — the property that determines guest accessibility — is only known *after* synthesis/simulation, not from the raw building blocks
- Goal: predict PLD directly from linker + metal identity, using a graph representation of MOF-to-MOF similarity
- One line linking to his own research direction: "This directly extends your MOFGalaxyNet and Black Hole work."

### Slide 3 — What Already Exists (Baseline #1: MOFGalaxyNet)
**Prompt:** Summarize the 2023 paper/repo factually, showing you've read the actual code, not just the abstract.
- Dataset: 2,000 CSD MOFs, 7-dim node feature vector
- Graph construction: weighted similarity (linker Tanimoto + metal cosine distance), fixed threshold φ = 0.9 → 19,266 edges, mean degree 19.256
- Community detection: Girvan-Newman → 246 clusters
- Model: GCN in TensorFlow/StellarGraph, 4-class PLD classification
- Known limitation (from the paper itself): lower thresholds (e.g., φ = 0.2) underfit due to noise; no k-NN alternative has been tested
- Visual: `MOFGalaxyNet.png` diagram + a small table of the numbers above

### Slide 4 — What Already Exists (Baseline #2: Black Hole Strategy)
**Prompt:** Summarize the 2025 paper/repo, again grounded in the actual code you read.
- Purpose: gravity-inspired sparsification to cut GNN training cost on large MOF networks
- Applied to a 14,000-MOF network (836 Louvain communities); gravity score = weighted combination of degree centrality, betweenness centrality, edge-weight-sum
- Result: GCN accuracy maintained/improved after removing 30–50% of nodes; ~114 MB memory at 50% pruning vs. ~823 MB (Stratified) / ~631 MB (PageRank)
- Modern stack: PyTorch + PyTorch Geometric-compatible model code (GCN, GraphSAGE, GAT all implemented)
- Never applied to the smaller, more accessible 2,000-MOF dataset — this is one of the gaps this case study closes
- Visual: reuse `BH2.png` or the animated GIF still-frame; small bar chart of memory numbers (823 / 631 / 114 MB)

### Slide 5 — The Gap / My Proposed Contribution
**Prompt:** This is the pivot slide — state clearly what's missing and what you'll add.
- Gap 1: MOFGalaxyNet's pipeline runs on deprecated StellarGraph/TensorFlow — not reproducible or extensible today
- Gap 2: The fixed-threshold graph construction (φ = 0.9) has never been systematically compared against k-NN, despite the paper's own note about isolated/underfit nodes
- Gap 3: Black Hole sparsification has never been tested at the smaller 2,000-MOF scale
- My contribution: reimplement the full pipeline in PyTorch Geometric, run a controlled graph-construction experiment (threshold vs. k-NN), integrate Black Hole sparsification on this dataset, and benchmark everything against non-graph baselines
- Visual: simple 3-box "gap → solution" diagram

### Slide 6 — Methodology / Development Phases
**Prompt:** Show the phase pipeline as a compact visual roadmap — do not restate all sub-tasks, just phase names + one-line objectives. Mention that Phase 0 (environment) and Phase 1 (data ingestion & EDA) are already complete, with a working Streamlit control panel to demo live if there's time.
- Phase 0: Environment setup ✅ done (PyTorch Geometric + RDKit, pip/venv, no conda)
- Phase 1: Data ingestion & EDA ✅ done — reproduced the paper's PLD histogram almost exactly; also built a dataset/feature-scheme switch (`config.py`) so the choice below isn't a rewrite either way
- Phase 2: Feature engineering (7-dim vs. 1031-dim — switchable, not blocking)
- Phase 3: Graph construction experiment — threshold (φ=0.9) vs. k-NN (k=3,5,10)
- Phase 4: Black Hole sparsification (τ=0.3, τ=0.5) on best graph
- Phase 5: GCN + GraphSAGE training (PyTorch Geometric)
- Phase 6: Baseline comparison (Random Forest, k-NN classifier)
- Phase 7: Analysis, report, thesis-extension outline
- Visual: horizontal swimlane/timeline graphic, phases 1→7, with a small icon per phase (data, graph, model, evaluation); check marks on 0–1

### Slide 7 — Open Points for Discussion
**Prompt:** Directly present the clarification questions from Part 1 above as a short, scannable list — this is the slide where you invite his input. Frame the dataset point as a decision-made-but-flagged, not a blocking question, since his email already gave flexibility here.
- Dataset scale (FYI, not blocking): found `MOFCSD.csv`/`MOFGalaxyNet.csv` (14,296 MOFs, real continuous PLD) are actually available locally despite the README saying otherwise. Started on the small 2,000-MOF set for faster iteration; built the pipeline to switch to the large set with a one-line config change if preferred
- Metal encoding gap: real data has 53 unique metals, but `data_utils.py`'s one-hot only covers 4 (Cu/Zn/Fe/Co) — needs generalizing if we switch to the large set
- Black Hole gravity weights: code default (0.3/0.3/0.4) vs. equal weighting (0.33/0.33/0.33) — which is intended?
- Task framing: classification only, or classification + regression (both supported by his existing code, and only truly possible with the large set's continuous PLD)?
- Confirm scope: is porting/reusing his model code (vs. writing from scratch) the right expectation for this case study?
- Visual: none needed — keep this slide text-only and direct, it's meant to drive discussion, not distract from it

### Slide 8 — Timeline & Thesis Outlook
**Prompt:** Close with the practical plan and the forward-looking hook.
- Compact 12-week timeline (from the Expose): weeks 1–4 data/features, 5–6 graph construction + Black Hole, 7–9 modeling + baselines, 10–12 analysis/report/deck
- Thesis extension directions (his own suggestions): multi-property prediction, improved graph construction (SOAP/3D structural similarity), Black Hole weight optimization, MOF recommendation system
- Closing line: "This case study is scoped to produce a clean, reproducible foundation — and a clear decision point — for which of these directions to pursue as a thesis."
- Visual: simple Gantt-style bar or 4-block timeline; small "thesis directions" icon row at the bottom

---

## Presentation delivery notes
- Keep slides 3–4 factual and brief — the goal is to demonstrate you've read the code deeply, not to re-teach him his own papers.
- Slide 7 (open points) is the most important slide for this specific meeting — leave the most time for it.
- Have `DEVELOPMENT_TODO.md` on hand as a backup/appendix slide if he asks "what does the granular plan look like."
