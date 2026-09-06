# MOFQuantumNet — Project Overview (Plain-English Edition)

*A guide for readers with no chemistry or machine learning background.*

---

## 1. Executive Summary & The "Why"

### The real-world problem

Imagine a material that looks like a solid brick of metal and plastic, but is secretly full of microscopic
holes — like a sponge, except the holes are a billionth of a meter wide and arranged with the precision of a
crystal. That's a **Metal-Organic Framework**, or **MOF** for short. Scientists build these on purpose,
because the holes are useful: they can trap carbon dioxide from a power plant's exhaust, store hydrogen fuel
safely, or filter out one gas from a mixture of others — like a very, very fine sieve.

The single most important measurement of that sieve is called the **Pore Limiting Diameter (PLD)**: the
width of the *narrowest* point a molecule has to squeeze through to get inside. Get the PLD right, and the
MOF filters exactly the molecule you want. Get it wrong, and it lets everything through, or nothing at all.

### Why this is hard

Here's the catch: **you can't easily know a MOF's PLD just by looking at its ingredients.** A MOF is built
from two components — a **metal** (like copper or zinc) and an **organic linker** (a small molecule that
connects the metal pieces together, like the poles connecting the corners of a tent). Mix a metal and a
linker together, and the pore size that results depends on *how* they lock together in 3D space — which is
not obvious just from reading off the ingredient list. Today, the reliable way to find out a MOF's PLD is to
either physically make it in a lab, or run an expensive computer simulation of its 3D structure. Both are
slow and costly, and chemists routinely have thousands of candidate metal+linker combinations they'd like to
screen.

### The everyday analogy

Think of it like trying to guess how well two puzzle pieces will fit together, and how big the *gap* will be
around them, **without ever assembling the puzzle** — just by looking at the shape of each piece on its own.
That's roughly what this project attempts: predict the size of the "gap" (the pore) a MOF will have, using
only a description of its two separate ingredients (metal + linker), never the fully-assembled 3D structure.

The tool used to make that guess is a **Graph Neural Network (GNN)** — a type of AI model that's especially
good at reasoning over *networks of relationships*, not just standalone facts. The idea explored here: build
a "social network" where each MOF is a node, and two MOFs are connected if their ingredients are chemically
similar to each other — then let the AI use that network of similarities to guess a new MOF's pore size,
the same way you might guess a stranger's taste in music by looking at who their friends are.

---

## 2. The Method: What & How We Did It

The whole project is organized as a **7-step pipeline** — each step is its own file in the `src/` folder,
and each step's output feeds into the next one. You can run every step from a command line, or use the
point-and-click web console (`webapp/`) to describe a MOF and get a prediction, or explore every pipeline
stage's real, live-computed results.

```
 STEP 1          STEP 2           STEP 3           STEP 4            STEP 5           STEP 6            STEP 7
 Load & Clean -> Turn MOFs   ->  Connect      ->   Prune the    ->   Train the   ->   Sanity-check ->   Compare &
 the Data        into Numbers    Similar MOFs      "Friend Map"      AI Models        vs. Simple         Report
                                 into a Map                                            Methods
```

### Step 1 — Data Ingestion (`src/data_ingestion.py`)
**What it does:** Loads the raw spreadsheets of MOFs and cleans them up — fixing missing values, discarding
a handful of corrupted entries, and putting everything into one consistent format.

**Analogy:** This is the equivalent of a librarian receiving a giant, messy stack of index cards — some
handwritten in different formats, a few torn or blank — and re-copying every one onto a standard template
before filing them.

**Two source datasets are used**, switchable from one setting in the dashboard:
- **Small dataset** (`data/raw/SMILES_METAL_2000_NoPLD.csv`) — 2,000 MOFs, each already labeled with a pore
  *category* (small / medium / large / non-porous), but no exact measurement.
- **Large dataset** (`data/raw/MOFCSD.csv`) — 14,296 MOFs, with an exact pore size in Ångströms (a unit of
  length almost inconceivably small — about one ten-billionth of a meter).

### Step 2 — Feature Engineering (`src/feature_engineering.py`)
**What it does:** Converts each MOF's chemical description (metal name + linker molecule's chemical
"spelling," called a SMILES string) into a list of numbers the AI can actually work with.

**Analogy:** AI models can't read chemistry the way a chemist does — they only understand numbers. This step
is like translating a house's description ("brick, three bedrooms, sloped roof, red door") into a
standardized numeric spec sheet (square footage, room count, roof angle in degrees) that a spreadsheet
formula could compare across thousands of houses at once.

**An important fix made here:** the original reference code this project builds on made the mistake of
including the *actual pore size* as one of the numbers describing each MOF — which is like handing someone
an exam with the answer key stapled to the question sheet. This project removes that number before training,
so the AI has to genuinely learn the relationship rather than just read off the answer (see `README.md`,
*"Key implementation decisions"*, and the leakage discussion below).

### Step 3 — Graph Construction (`src/graph_construction.py`)
**What it does:** Compares every MOF to every other MOF and draws a connecting line ("edge") between two MOFs
if their metal and linker are chemically similar enough. The result is a giant "friendship map" — technically
called a **graph** — where each MOF is a dot (node) and similar MOFs are linked by lines.

**Analogy:** Picture a school yearbook superlative page turned into a web of string connecting every student
to their closest-matching classmates by hobbies and interests. Four different ways of deciding "how similar
is similar enough" were tried and compared (one fixed cutoff, three "nearest neighbor" variants), and the
best-connected version was kept for later steps.

### Step 4 — Black Hole Sparsification (`src/black_hole_sparsification.py`)
**What it does:** The full friendship map can be huge and expensive to compute over, so this step trims it
down — keeping only the most "important" or well-connected MOFs in each neighborhood, while making sure every
pore-size category still has fair representation.

**Analogy:** Imagine condensing a 500-person conference guest list down to the 150 most centrally-connected
attendees — the ones whose absence would most disrupt the overall social network — while still making sure
every department is represented, so the mini-conference is still useful even though it's much smaller and
cheaper to run.

### Step 5 — GNN Training (`src/gnn_training.py`)
**What it does:** Trains three different flavors of Graph Neural Network (called GCN, GraphSAGE, and GAT —
think of them as three different "styles" of the same reasoning approach) to predict pore size, learning from
both a MOF's own ingredient list *and* its position in the friendship map.

**Analogy:** This is the "studying for the exam" phase. The model is shown a batch of MOFs with known
answers (the *training set*), and it adjusts its own internal logic over many rounds (*epochs*) until its
guesses get as close to correct as possible — the way a student improves by working through practice
problems and checking answers, then stopping once further practice stops helping (a technique called
**early stopping**, to avoid the model just memorizing the practice set instead of actually learning the
pattern — "overfitting").

**A critical, deliberate design choice:** the model is *never* shown the correct answers for a specific,
locked-aside group of MOFs (the *test set*) until the very end, when it's graded exactly once. A subtlety
found in the original reference code was that it accidentally let the model "peek" at the test answers during
practice too — like letting a student see the final exam questions while they're still doing homework. That
mistake was found and fixed here (see Key Findings below).

### Step 6 — Baseline Comparison (`src/baseline_models.py`)
**What it does:** Trains two much simpler, "no friendship map" AI methods — Random Forest and k-Nearest
Neighbors — using only each MOF's own ingredient list, with none of the graph/network information from
Step 3.

**Analogy:** This is the control group in an experiment. If a doctor claims a new expensive treatment works,
you compare it against a cheap over-the-counter alternative to see if the expensive one is actually worth it.
Here, the "expensive treatment" is the graph-based AI; the "cheap alternative" is a much simpler method that
never even looks at the friendship map.

### Step 7 — Trade-off Analysis & Reporting (`src/graph_tradeoff_analysis.py`, `planning/CASE_STUDY_REPORT.md`)
**What it does:** Double-checks that the "best" friendship map chosen in Step 3 (picked using structural
properties) actually produces the *best predictions* downstream — since those two things aren't guaranteed
to agree — and writes up the full set of findings.

---

## 3. Key Findings & Results

### Finding 1 — The simple method won

This is the headline result, and it's an honest one rather than a flattering one: **the simple, no-graph
methods (Step 6) beat every single Graph Neural Network configuration (Step 5), on both datasets, on both
prediction tasks.**

```
 SMALL DATASET — predicting pore-size CATEGORY (accuracy, higher = better)
 ┌────────────────────────────┬────────────────────────────────────────────┐
 │ Random Forest (simple)     │ ███████████████████████████████░░░░░  66.7%│
 │ k-Nearest Neighbor (simple)│ ███████████████████████████████░░░░░  63.8%│
 │ Best Graph Neural Network  │ ██████████████████████████████░░░░░░  60.1%│
 └────────────────────────────┴────────────────────────────────────────────┘

 LARGE DATASET — predicting pore-size CATEGORY (accuracy, higher = better)
 ┌────────────────────────────┬────────────────────────────────────────────┐
 │ Random Forest (simple)     │ ██████████████████████████████████████ 85.1%│
 │ k-Nearest Neighbor (simple)│ ██████████████████████████████████░░░░ 81.3%│
 │ Best Graph Neural Network  │ █████████████████████████████░░░░░░░░░ 72.5%│
 └────────────────────────────┴────────────────────────────────────────────┘

 LARGE DATASET — predicting EXACT pore size in Ångströms (R², higher = better fit)
 ┌────────────────────────────┬────────────────────────────────────────────┐
 │ Random Forest (simple)     │ ███████████████████████████████████░░░  0.89│
 │ Best Graph Neural Network  │ █████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.23│
 └────────────────────────────┴────────────────────────────────────────────┘
```

**What these numbers mean in plain terms:**
- **Accuracy 85.1%** means: out of every 100 MOFs the model had never seen before, it correctly sorted about
  85 of them into the right pore-size bucket (non-porous / small / medium / large).
- **R² of 0.89 vs. 0.23** — R² measures how much of the *variation* in pore size the model actually explains,
  on a scale where 1.0 is a perfect predictor and 0 is no better than always guessing the average. 0.89 means
  the simple method captures the vast majority of the pattern; 0.23 means the graph-based model is only
  picking up a small fraction of it.

**Why the simple method won — the honest explanation:** the friendship map in Step 3 is built from the exact
same ingredient information (metal + linker similarity) that's *already* given directly to every model as
input. So the graph doesn't hand the AI any genuinely new information — it mostly just restates, in a more
roundabout way, facts the model already had. It's a bit like being handed a summary of a book you've already
read: it doesn't teach you anything new, and can even add noise. This is a real, useful scientific finding —
it says the *specific way* the friendship map is built here isn't adding predictive value, not that graph
methods can never work for this kind of problem. Full discussion in
[`planning/CASE_STUDY_REPORT.md`](planning/CASE_STUDY_REPORT.md).

### Finding 2 — Two real bugs found and fixed in the original reference code

This project builds on two earlier academic works (MOFGalaxyNet and the "Black Hole Strategy" paper), and in
the process of re-implementing their methods carefully, two genuine mistakes were found in their original
code — both of which would have made results look better than they really are:

1. **"Answer key" leakage:** the original code fed the model the actual pore size as one of its input
   numbers — while also asking it to predict that same pore size. Fixed by removing it entirely
   (`src/feature_engineering.py`).
2. **"Peeking at the exam" leakage:** the original training code let the model's practice-stopping decision
   be influenced by the very same MOFs it would later be tested on. Fixed with a proper three-way split —
   train / practice-check / final-exam — where the final-exam group is never touched until the very last
   step (`src/gnn_training.py::make_splits()`).

Both are documented in detail, with exact file-and-line references, in
[`planning/CASE_STUDY_REPORT.md`](planning/CASE_STUDY_REPORT.md) §4.5–4.6 — including a side-by-side
comparison showing that the *published* numbers from the original papers likely benefited from these same
issues, which is the main reason this project's numbers look lower on paper while actually being more
trustworthy.

### Finding 3 — Compressing the friendship map barely hurts accuracy

The "Black Hole" trimming step (Step 4) can shrink the friendship map down to about half its original size
while keeping prediction accuracy almost the same — in some cases even slightly *better* than using the full,
untrimmed map:

```
 Node retention at different trim strengths (both datasets, similar pattern):
   Trim level τ=0.3  →  keeps ~67% of MOFs,  ~40% of connections
   Trim level τ=0.5  →  keeps ~48% of MOFs,  ~17% of connections
```

**In plain terms:** you can throw away roughly half the map and lose almost nothing in prediction quality —
useful if you're working with a much bigger dataset than this one and need to save computing time and memory.

---

## 4. Future Scope & Next Steps

### Current limitations

- **Single-run results.** Each experiment was run once rather than repeated many times and averaged, so the
  exact numbers have some natural run-to-run wobble (the reference papers this project compares against have
  the same issue, sometimes worse — see `planning/CASE_STUDY_REPORT.md` §4.5–4.6 for a detailed audit).
- **The exact pore size is hard to predict precisely** (R² of 0.23) — good enough to rank MOFs roughly, not
  yet good enough to trust down to the decimal.
- **The friendship map, as currently built, doesn't add predictive value** — it's built from information the
  model already has, rather than something genuinely new like 3D geometry.
- **One attention-based model (GAT) sometimes "collapses"** on the smaller dataset — it starts guessing the
  same category for almost everything, rather than making varied predictions. A known sensitivity of that
  particular model style on smaller, sparser networks.

### Logical next steps

1. **Build a richer friendship map.** Instead of connecting MOFs by the same ingredient information already
   fed to the model, connect them using something genuinely new — for example, a rough 3D-shape similarity
   measure — so the graph actually teaches the model something it didn't already know.
2. **Predict more than one property at once.** Real MOF design involves trade-offs between multiple
   properties (pore size, stability, gas capacity) — extending the model to predict several jointly would be
   more useful in practice than pore size alone.
3. **Tune the trimming settings automatically.** Right now the "how aggressively to trim the map" knobs are
   manually adjustable sliders in the dashboard; a systematic search could find the settings that squeeze out
   the best accuracy-per-computing-cost trade-off.
4. **Validate the graph-selection method against real accuracy, not just structure.** This project already
   found that picking the "best" friendship map by structural neatness alone doesn't always match the map
   that gives the best real-world predictions (`src/graph_tradeoff_analysis.py`) — a smarter selection
   process that checks actual accuracy, not just tidiness, is a natural improvement.
5. **Average multiple runs.** Repeating each experiment several times (with different random starting
   points) and reporting the average would make the final numbers more statistically solid and less
   sensitive to any single lucky or unlucky run.

---

*For the full technical write-up, exact numbers, and file-by-file audit trail behind every claim in this
document, see [`planning/CASE_STUDY_REPORT.md`](planning/CASE_STUDY_REPORT.md) and
[`planning/QA_REPORT.md`](planning/QA_REPORT.md).*
