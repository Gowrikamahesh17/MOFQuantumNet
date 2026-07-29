# First Meeting Prep — Case Study 2 Kickoff with Prof. Dr. Jalali

Main agenda: show Prof. Jalali (1) what already exists in his prior work, (2) what I plan to build on top of it, and (3) surface open points before I start coding.

**Meeting held 2026-07-21 — outcome recorded inline below (B, C, D resolved).**

---

## Part 1 — Points to Discuss / Clarify

Use these to structure the conversation and to make clear I've read his code, not just his papers.

### A. Confirm scope framing
- [ ] Confirm the topic title stands as **"Graph Neural Networks for MOF Property Prediction"** and that PLD (pore-limiting diameter) is the right target property to keep, given it's the one both `MOFGalaxyNet` and `BlackHole` already model.
- [x] ~~Confirm the 2,000-MOF CSD dataset from the `MOFGalaxyNet` repo is acceptable as the primary dataset~~ — **resolved by EDA, see B below.**

### B. Dataset choice — the root decision everything else cascades from — ✅ RESOLVED 2026-07-21
**Outcome: both datasets stay in scope, selectable via one config/UI switch — not an either/or pick.** Feature dimensionality and task framing cascade automatically from this switch, exactly as framed below (confirmed, not just proposed).

**Important framing:** this is not four separate open questions. It's **one** decision (which dataset) that determines the other three (feature dimensionality, metal encoding, classification-vs-regression), because the two datasets have genuinely different columns — one isn't just a bigger version of the other.

**Column-level comparison (exact columns, from actually loading both files):**

| | Small set — `SMILES_METAL_2000_NoPLD.csv` (2,000 MOFs) | Large set — `MOFCSD.csv` (14,296 MOFs) |
|---|---|---|
| Identifier | `refcode`, `mof_index` | `refcode` |
| Linker structure | `linker_smiles` | `linker SMILES` |
| Metal identity | ❌ none — only 6 anonymous numeric columns `metal_feat_0`…`metal_feat_5` | ✅ `metal` — real element symbol, 53 unique values |
| Pore size (continuous) | ❌ not present at all | ✅ `Pore Limiting Diameter` (Å) |
| Pore size (category) | ✅ `pld_category` — precomputed label, code 0–3 | derived by us, by binning `Pore Limiting Diameter` |
| Other pore geometry | ❌ none | ✅ `Largest Cavity Diameter`, `Largest Free Sphere` |

**Why this one choice cascades into the other three:**
- **Feature dimensionality (7-dim vs. 1031-dim):** the small set's `metal_feat_0..5` naturally pairs with the original paper's compact 7-dim descriptor; the large set's real `metal` name + pore-geometry columns are what BlackHole's 1031-dim Morgan-fingerprint scheme is built from. You can't mix and match — the columns each scheme needs only exist in one dataset.
- **Metal encoding:** only the large set has a `metal` column to encode at all, and it has 53 real values against `data_utils.py`'s hardcoded 4-slot one-hot — so this gap only matters if we pick the large set.
- **Classification vs. regression:** only the large set has `Pore Limiting Diameter` as an actual number — regression is impossible on the small set, since it only ever has the precomputed category label.

**New finding:** his BlackHole README states `MOFGalaxyNet.csv`/`MOFCSD.csv` are "not included," but both are actually present locally with real data: `MOFCSD.csv` has **14,296 MOFs**, continuous PLD (range 0–71.5 Å, mean 3.3 Å), 53 real metal names (Zn 2,004 / Cu 1,847 / Cd 1,313 / Co 1,046 down to single-digit rare metals), zero NaNs. Derived PLD category split: 7,106 nonporous / 3,540 medium / 2,840 small / 810 large pore.

**Decision:** Starting development on the small set (faster iteration, matches the paper's exact published numbers — I reproduced its category histogram almost exactly: 1,062/425/271/246 vs. the paper's 1,062/422/271/244). Built the pipeline so the dataset is a one-line config switch (`src/config.py`), not a rewrite — see `DEVELOPMENT_TODO.md`'s "Generic, switchable design" section. Will switch to the large set later if he'd prefer, without losing any work.

### C. Black Hole gravity weights (discrepancy I found) — ✅ RESOLVED 2026-07-24 (verified against the actual paper)
- My Expose text (drafted from the papers) states equal weights α = β = γ = 0.33 for the gravity score.
- The actual `bh_sparsification.py` code default is `weights=(0.3, 0.3, 0.4)` (degree centrality, betweenness centrality, edge-weight-sum) — different from the Expose's stated value.
- **Resolved by directly reading `BlackHole.pdf`, Section 2.3, p.7:** *"we assign equal weights to all three components"* — 0.33/0.33/0.33 is the paper's own justified main configuration (their ablation study found single-component emphasis unstable across sparsification levels; equal weighting was "consistently stable and competitive"). The code's 0.3/0.3/0.4 default does not match what the paper reports using — the paper wins as the authoritative source.
- **Implementation:** default changed to 0.33/0.33/0.33 in `src/config.py`. Three independent controls (slider + number-input text box for exact entry) in `app.py` Tab 4 — deliberately **not** auto-rebalancing: setting one to 0.50 leaves the other two untouched, so any two values can be fixed manually (e.g. 0.50 and 0.15) and the third stays put. The raw values don't need to sum to 1 — `PipelineConfig.gravity_weights_normalized` normalizes transparently before the gravity score is computed, and the UI shows both the raw sum and the normalized values live. A pruning-threshold (τ) slider was added alongside them too.

### D. Classification vs. regression framing — ✅ RESOLVED 2026-07-21
- **Outcome: cascades from the dataset switch, not an independent choice.** Small dataset → classification only, forced. Large dataset → user picks via a UI radio.
- `BlackHole/main.py` supports both a 4-class PLD classification task and a continuous PLD regression task via a `--task` flag. Both predict the same underlying property (Pore Limiting Diameter) — they just differ in how precise the output is:
  - **Classification** — predicts one of 4 buckets: nonporous / small pore / medium pore / large pore. Matches the original 2023 paper exactly, works on both datasets.
  - **Regression** — predicts the exact PLD value in Å (e.g. 3.47), not just a bucket. Only possible on the large dataset (see column table in B above).
- Resolved: run both, gated by dataset — no separate scope decision needed.

### E. Reused vs. novel code — expectation setting
- Confirm it's acceptable to **directly reuse/port** his `GCN`, `GraphSAGE`, and `GAT` model classes and `train()`/`test()` functions from `BlackHole/graphsage_model.py`, since the technical contribution is the PyG reimplementation + graph-construction experiment + baseline comparison, not reinventing model architectures.
- ~~Confirm the case study should not touch GAT... unless he wants a 3-model comparison instead of 2.~~ **Correction (2026-07-24):** this line previously claimed the professor confirmed 2 models were sufficient — that never actually happened. This was drafted as a question for a meeting that ended up covering other topics instead; it was never asked or answered. Since the actual `BlackHole.pdf` paper's own evaluation tests all three models (GAT, GCN, GraphSAGE — Section 3.4), GAT has now been added too, matching the paper's full comparison rather than a 2-model subset.

### F. Compute / dataset size sanity check
- Confirm expected runtime is reasonable for a case study (his own README notes ~10 minutes per run at 14,000-MOF scale, on CPU) — at 2,000 MOFs this should be faster, but want to flag in case he expects GPU access or a compute budget conversation. (Already tested on a MacBook Air M4 — CPU is sufficient, no GPU needed either way.)

### G. Report format
- Confirm expected deliverable format: Jupyter notebook + written report + slide deck (as I've assumed), or if SRH has a specific report template he wants me to use.

---

## Part 2 — PowerPoint Presentation Prompts (10 slides max)

Written to be pasted directly into an AI slide generator (Gamma). Each block gives: the slide's purpose, the exact content (including numbers — don't let the generator invent or round them), and a visual suggestion. Keep slides clean and visual (Gamma's default card/icon layouts, not walls of text) — but every number and column name below should still appear somewhere on the slide, even if in a small table or caption, so nothing gets glossed over in the actual conversation.

**Presenter:** Gowrika Mahesh, student, SRH University Heidelberg
**Supervisor:** Prof. Dr. Mehrdad Jalali

---

### Slide 1 — Title
- Title: *Graph Neural Networks for MOF Property Prediction*
- Subtitle: Case Study 2 — Kickoff & Approach Review
- Presenter: Gowrika Mahesh, Student, SRH University Heidelberg
- Supervisor: Prof. Dr. Mehrdad Jalali
- Date: [meeting date]
- Visual: a clean MOF network/graph image as a background accent (e.g. a molecular/graph-network motif — nodes and edges, dark or light depending on Gamma's theme)

### Slide 2 — Why MOFs, Why This Problem
- One slide, brief, written for someone already expert — framing, not teaching
- MOFs = porous crystalline materials (metal nodes + organic linkers), tunable for gas storage, carbon capture, catalysis
- Bottleneck: Pore Limiting Diameter (PLD) — the property that determines guest accessibility — is only known *after* synthesis/simulation, not from the raw building blocks
- Goal: predict PLD directly from linker + metal identity, using a graph representation of MOF-to-MOF similarity
- Closing line: "This directly extends your MOFGalaxyNet and Black Hole work."
- Visual: simple diagram — metal node + organic linker → MOF structure → pore opening (icon-style, not a technical figure)

### Slide 3 — What Already Exists: MOFGalaxyNet (2023)
- Dataset: 2,000 CSD MOFs, 7-dim node feature vector
- Graph construction: weighted similarity (linker Tanimoto + metal cosine distance), fixed threshold φ = 0.9 → 19,266 edges, mean degree 19.256
- Community detection: Girvan-Newman → 246 clusters
- Model: GCN in TensorFlow/StellarGraph, 4-class PLD classification
- Known limitation (from the paper itself): lower thresholds (e.g. φ = 0.2) underfit due to noise; no k-NN alternative has been tested
- Visual: the `MOFGalaxyNet.png` / `Galaxy4.png` network diagram from the repo, plus a compact stat callout box (2,000 MOFs · 19,266 edges · 246 clusters)

### Slide 4 — What Already Exists: Black Hole Strategy (2025)
- Purpose: gravity-inspired sparsification to cut GNN training cost on large MOF networks
- Applied to a 14,000-MOF network (836 Louvain communities); gravity score = weighted combination of degree centrality, betweenness centrality, edge-weight-sum
- Result: GCN accuracy maintained/improved after removing 30–50% of nodes
- Memory at 50% pruning: **114 MB (Black Hole)** vs. 823 MB (Stratified) vs. 631 MB (PageRank)
- Modern stack: PyTorch + PyTorch Geometric-compatible model code (GCN, GraphSAGE, GAT)
- Gap: never applied to the smaller, more accessible 2,000-MOF dataset
- Visual: a simple 3-bar comparison chart of the memory numbers (823 / 631 / 114 MB) — this is the single most persuasive number in the deck, give it a real chart, not just text

### Slide 5 — The Gap / My Proposed Contribution
- Gap 1: MOFGalaxyNet's pipeline runs on deprecated StellarGraph/TensorFlow — not reproducible or extensible today
- Gap 2: The fixed-threshold graph construction (φ = 0.9) has never been systematically compared against k-NN, despite the paper's own note about isolated/underfit nodes
- Gap 3: Black Hole sparsification has never been tested at the smaller 2,000-MOF scale
- My contribution: reimplement the full pipeline in PyTorch Geometric, run a controlled graph-construction experiment (threshold vs. k-NN), integrate Black Hole sparsification on this dataset, and benchmark everything against non-graph baselines
- Visual: simple 3-row "gap → solution" layout (icon + short label per row)

### Slide 6 — Dataset Deep-Dive: One Decision, Three Consequences
- This is a new, dedicated slide — do not compress this into a bullet elsewhere, it's the piece most likely to cause confusion if left implicit
- Headline: "Two complete datasets exist locally (despite the README saying otherwise) — picking one decides three other questions automatically."
- Small table (reproduce exactly):

  | | Small (2,000 MOFs) | Large (14,296 MOFs) |
  |---|---|---|
  | Metal identity | ❌ none (6 anonymous numbers) | ✅ `metal` column, 53 real values |
  | Continuous PLD | ❌ none | ✅ `Pore Limiting Diameter` (0–71.5 Å) |
  | Category label | ✅ precomputed (1,062/425/271/246) | derived (7,106/3,540/2,840/810) |

- Cascade diagram: **Dataset choice →** Feature vector (7-dim vs. 1031-dim) **→** Metal encoding scope (N/A vs. 53 metals to handle) **→** Task (classification-only vs. classification+regression)
- Currently building on the small set; pipeline is a one-line config switch either way — no work lost by switching later
- Visual: a horizontal flowchart/tree — one box ("Dataset choice") branching into three downstream boxes — this is the slide most worth a real diagram, not just bullets

### Slide 7 — Methodology / Development Phases
- Show as a compact roadmap — phase names + one-line objective only, full detail lives in the appendix doc
- Phase 0: Environment setup ✅ done (PyTorch Geometric + RDKit, pip/venv, no conda)
- Phase 1: Data ingestion & EDA ✅ done — reproduced the paper's PLD histogram almost exactly; live Streamlit control panel built to demo
- Phase 2: Feature engineering — build the node feature vector (compact 7-dim or fingerprint 1031-dim, per Slide 6)
- Phase 3: Graph construction experiment — threshold (φ=0.9) vs. k-NN (k=3,5,10)
- Phase 4: Black Hole sparsification (τ=0.3, τ=0.5) on best graph
- Phase 5: GCN + GraphSAGE training (PyTorch Geometric)
- Phase 6: Baseline comparison (Random Forest, k-NN classifier)
- Phase 7: Analysis, report, thesis-extension outline
- Visual: horizontal step-tracker, 8 steps (0→7), first two visually marked complete (checkmark/filled), rest outlined/pending

### Slide 8 — Project Timeline
- Give this its own slide with a real Gantt-style visual — do not just list weeks as text
- 12-week plan:

  | Week(s) | Focus |
  |---|---|
  | 1–2 | Literature deep-dive, environment setup |
  | 3 | Data loading & integrity check |
  | 4 | Feature engineering |
  | 5 | Graph construction experiment |
  | 6 | Black Hole sparsification |
  | 7–8 | GNN model training |
  | 9 | Baseline model training |
  | 10 | Analysis & interpretation |
  | 11 | Report writing |
  | 12 | Review, finalize, prepare deck |

- Callout: Weeks 1 (environment) and part of week 3 (data loading) already complete as of this meeting
- Visual: Gantt chart, 12 columns (weeks) × 10 rows (tasks above), with weeks 1 and the data-loading portion of week 3 shaded as done

### Slide 9 — Open Points for Discussion
- Most important slide for this specific meeting — the AI generator should keep this text-dense and plain (a scannable list, not icon cards), since this slide exists to drive discussion, not to impress visually
- Dataset choice (not blocking, his email already gave flexibility): started on the small 2,000-MOF set; large 14,296-MOF set confirmed available and ready to switch to in one config change if preferred — see Slide 6 for what that decides
- Black Hole gravity weights: code default (0.3/0.3/0.4) vs. equal weighting (0.33/0.33/0.33) — which is intended?
- Task framing: classification only (matches original paper, works on either dataset), or classification + regression (needs the large dataset's continuous PLD)?
- Metal encoding: large dataset has 53 real metals vs. `data_utils.py`'s hardcoded 4-slot one-hot — needs generalizing if we switch
- Confirm scope: is porting/reusing his GCN/GraphSAGE model code (vs. writing from scratch) the right expectation for this case study?
- Visual: none — plain text list, four to five short lines

### Slide 10 — Thank You
- Title: "Thank you"
- Line: "Thank you for your time and guidance, Prof. Dr. Jalali."
- Line: "Looking forward to your feedback on the approach above before I begin full implementation."
- Contact: Gowrika Mahesh — Gowrika.Mahesh@stud.srh-university.de
- Small forward-looking line (optional, keep to one sentence): thesis-extension directions already identified — multi-property prediction, improved graph construction (SOAP/3D structural similarity), Black Hole weight optimization, MOF recommendation system
- Visual: clean, minimal — matches Slide 1's style, no clutter

---

## Presentation delivery notes
- Keep slides 3–4 factual and brief — the goal is to demonstrate you've read the code deeply, not to re-teach him his own papers.
- Slide 6 (dataset cascade) and Slide 9 (open points) are the two slides that matter most for this meeting — give them the most air time.
- Every number quoted on Slides 3, 4, 6, and 8 came from actually running the code/EDA, not from the papers alone — worth saying out loud once, so he knows the numbers are verified, not copied.
- Have `DEVELOPMENT_TODO.md` on hand as a backup/appendix if he asks "what does the granular plan look like."
