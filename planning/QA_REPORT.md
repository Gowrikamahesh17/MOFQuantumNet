# QA Report — Test Suite & Quality Assurance

Consolidated results from Phase 8 (Testing & QA). Covers unit, property-based,
torture, acceptance, and mutation testing across `src/`, plus a final coverage
pass. Companion to `planning/DEVELOPMENT_TODO.md`'s Phase 8 entry.

## 1. Summary

| Layer | Location | Count | Status |
|---|---|---|---|
| Unit | `tests/unit/` | 187 | ✅ all passing |
| Property-based (Hypothesis) | `tests/property/` | 11 test groups (thousands of generated cases) | ✅ all passing |
| Torture (extreme/edge-case) | `tests/torture/` | 26 | ✅ all passing |
| Acceptance (real data + docs claims) | `tests/acceptance/` | 19 (1 marked `slow`) | ✅ all passing |
| **Total** | `tests/` | **243** | ✅ **243/243 passing** |
| Mutation testing | `config.py`, `black_hole_sparsification.py` | 435 mutants | 367 killed (84.4%) |
| Line coverage | `src/` (via `pytest-cov`) | 879 statements | 67% overall |

Grew from 235 to 243 tests since the initial pass: 8 new unit tests cover
`graph_construction.py::sample_subgraph_for_viz()`, added when the interactive
graph-visualization feature was built (see §5 below for a flaky-test bug this
also surfaced and fixed).

Run commands:
```bash
pytest tests/ -q                                   # everything, ~10s
pytest tests/ -m "not slow" -q                      # skip the one real-training smoke test, ~4s
pytest tests/ -m "not slow" --cov=src --cov-report=term-missing -q
mutmut run && mutmut results
```

## 2. What each layer actually checks

- **Unit** — every `src/` module's individual functions in isolation: `config.py`'s
  validation/cascade rules, `data_ingestion.py`'s parsing and category derivation,
  `feature_engineering.py`'s two feature schemes, `graph_construction.py`'s four graph
  variants, `black_hole_sparsification.py`'s gravity/pruning pipeline, `gnn_training.py`'s
  three model architectures (GCN/GraphSAGE/GAT) and split logic, and `baseline_models.py`'s
  comparison tables. Not tested: `ui_theme.py` (pure Streamlit markup helpers, no
  branching logic to break) and `graph_tradeoff_analysis.py` (a thin, one-off analysis script
  already exercised manually when Phase 7's findings were produced).
- **Property-based (Hypothesis)** — invariants that must hold across the *entire* input
  space, not just hand-picked boundary values: `derive_pld_category` always returns a
  valid category and is monotonic in PLD; gravity weights always normalize to sum to 1
  regardless of raw input; the small dataset always forces classification; class weights
  always sum to 1 and stay finite; `make_splits` always partitions the full node range
  with no overlap; Black Hole sparsification never increases node/edge counts and never
  mutates the graph it's given, for any weights/threshold in the valid range.
- **Torture** — inputs a well-behaved pipeline should survive even though they're
  unrealistic: empty dataframes, single-node graphs, all-identical linkers (zero
  chemical diversity), all-invalid SMILES, zero-edge graphs, a single test node, and
  extreme (1e6-scale) feature magnitudes. This layer is what actually found bugs (§3).
- **Acceptance** — re-verifies specific claims made in `README.md` /
  `DEVELOPMENT_TODO.md` / `CASE_STUDY_REPORT.md` against the *real* data files (row
  counts, category histograms within tolerance of the paper's figures, feature
  dimensions, isolated-node rates, node-retention percentages at τ=0.3/0.5), plus one
  `@pytest.mark.slow` end-to-end smoke test that runs the full pipeline with a real
  (reduced-epoch) GCN training run. Large-dataset tests skip gracefully if
  `reference_content/` isn't present in a given checkout.
- **Mutation testing** — the check on the checks: deliberately introduces small bugs
  ("mutants") into the source and confirms the test suite actually fails when they're
  present. A mutant that survives means some line of logic can be broken without any
  test noticing.

## 3. Bugs found and fixed during this phase

Genuinely new bugs, caught by the tests written in this phase (not previously known):

1. **`build_similarity_graphs` crashed below 11 rows** (`graph_construction.py`) —
   `np.argpartition` requires `kth < n`; any dataset with fewer than
   `max(KNN_VALUES) + 1 = 11` rows raised `ValueError: kth(=10) out of bounds`. Found by
   a torture test parametrized over tiny dataset sizes. Fixed by clamping
   `max_k = min(max(KNN_VALUES), max(0, n - 1))`.
2. **`compute_fingerprint_matrix` crashed on an empty input Series**
   (`feature_engineering.py`) — `np.stack` on zero arrays raises `ValueError: need at
   least one array to stack`. Fixed with an early return of a `(0, MORGAN_N_BITS)` array.
3. **Empty-string SMILES silently treated as valid** (`feature_engineering.py`) —
   RDKit's `Chem.MolFromSmiles("")` returns a non-`None`, 0-atom `Mol` rather than
   `None`, so an empty string passed through as "valid" with a physically meaningless
   `MolWt` of 0.0. Verified empirically before fixing. Fixed `_mol_or_none()` to also
   reject `mol.GetNumAtoms() == 0`.
4. **`logging_setup.py::get_logger()` assumed `outputs/` already existed** relative to
   the current working directory. Every test run so far happened to execute from the
   repo root (where `outputs/` already exists), so this never surfaced — until mutmut
   ran pytest from its own scratch `mutants/` directory, which doesn't have that folder,
   and crashed with `FileNotFoundError`. Fixed with `os.makedirs(log_dir, exist_ok=True)`.

These are in addition to the bugs found and fixed in earlier phases (target leakage,
non-reproducible invalid-SMILES fallback, the mutation bug in
`black_hole_strategy_per_community`, the test/val split leakage in `gnn_training.py`,
the gravity-weight default mismatch with the paper) — see `DEVELOPMENT_TODO.md` for
those.

## 4. Mutation testing detail

Scoped deliberately to `src/config.py` and `src/black_hole_sparsification.py` — the
fast, pure-logic core. GNN training (real PyTorch epochs) and RDKit-heavy code were
excluded: mutmut re-runs the test suite once per mutant, and with hundreds of mutants,
anything involving real training or SMILES parsing would make a full run impractically
slow.

**Starting point:** 298/435 mutants killed (68.5%). Reviewing the 137 survivors found
several that represented genuine, fixable test gaps rather than noise — 6 new unit
tests were added to close them:

| Gap found | What survived | Test added |
|---|---|---|
| Per-community centrality could silently degrade to whole-graph centrality (`graph.subgraph(community)` → `graph.subgraph(None)`) | Nothing distinguished the two on the existing fixture in a way that was asserted | `test_centrality_is_scoped_to_the_community_not_the_whole_graph` |
| Community detection's `seed=42` could become `seed=None`, breaking run-to-run determinism | Only per-community gravity's determinism (given already-fixed communities) was checked, not `louvain_communities` itself | `test_deterministic_given_same_graph_and_weights` |
| `round(peak_memory, 2)` → `round(peak_memory, None)` silently returns an `int` instead of a `float` | Nothing checked the metric's type | `test_peak_memory_mb_is_a_rounded_float` |
| The 20%-node-retention floor (`0.2 * n_total` → `0.2 / n_total`) | Only checked on a 10-node graph, where the floor's effect was too small to distinguish from other floors in the same formula | `test_twenty_percent_floor_scales_with_graph_size` (100-node graph, where correct vs. buggy floors diverge sharply) |
| Edge-pruning budget (`... - len(test_edges_to_keep)` → `... + len(test_edges_to_keep)`) | Every existing test used a small `edge_threshold` where both formulas produced the same final count | `test_num_to_keep_subtracts_reserved_test_edges_from_the_budget` (30-node ring, sized so the two formulas diverge and both stay within budget) |
| `save_sparsified_graph`'s CSV/JSON contents (columns, values, exact metrics) | Only checked that the output files *exist*, never their contents | `test_saved_csv_and_json_contents_match_the_result` |

**After adding these:** 367/435 mutants killed (84.4%), a 22-point improvement from 6
targeted tests.

**Remaining 68 survivors**, by function:

| Function | Survivors | Assessment |
|---|---|---|
| `calculate_gravity_per_community` | 24 | Mostly arithmetic/comparison-operator mutants deep in `MinMaxScaler`/centrality plumbing where behavior is either equivalent (e.g. `reshape(-2, 1)` behaves identically to `reshape(-1, 1)` in NumPy — a true equivalent mutant, unkillable by any test) or affects only a diagnostic tie-break with no observable-in-practice difference on realistic graphs. |
| `black_hole_strategy_per_community` | 18 | Mostly community-target-rounding arithmetic in the `diff` redistribution loop — affects at most ±1 node's placement across communities, not a correctness property documented or relied on anywhere. |
| `apply_black_hole_sparsification` | 11 | Mostly metrics-dict cosmetic mutants (rounding precision on secondary fields, log-message formatting) not tied to any documented behavior. |
| `save_sparsified_graph` | 7 | Residual formatting mutants (e.g. filename tag precision) after the content-check test closed the main gap. |
| `prune_edges` | 5 | Mostly the `max(0, ...)` → `max(1, ...)` floor clamp, which only differs in the pathological case of more fixed-test-nodes than the entire edge budget — already qualitatively covered by `test_fixed_test_node_keeps_at_least_one_edge`, not worth a bespoke deterministic construction for marginal additional confidence. |
| `_memory_mb` | 3 | A one-line `psutil` wrapper feeding a diagnostic-only metric; not meaningfully testable without mocking `psutil`, and its exact value has no effect on any decision the pipeline makes. |

None of the remaining survivors were judged to represent a real risk to a documented
functional requirement — they were deliberately not chased further to avoid writing
tests whose only purpose is to kill mutants rather than protect a real invariant.

**Tool limitation found:** this version of mutmut generates **zero mutants** for methods
defined inside a class decorated with `@dataclass`. `PipelineConfig`'s own logic (inside
`__post_init__` and its two `@property` methods) therefore has 0% direct mutation
coverage, despite 100% line coverage and 41 passing unit tests covering every branch by
hand. This is a gap in the tool for this codebase's use of dataclasses, not a gap in the
tests — noted here rather than silently presenting a misleadingly clean mutation score.

## 5. Coverage

```
Name                               Stmts   Miss  Cover   Missing
----------------------------------------------------------------
src/baseline_models.py                73     28    62%   101-133
src/black_hole_sparsification.py     137     20    85%   57, 217-238
src/config.py                         35      0   100%
src/data_ingestion.py                113     58    49%   107-109, 113-156, 160-183
src/feature_engineering.py           103     16    84%   157, 176-192
src/gnn_training.py                  207     72    65%   71, 166, 176, 211-260, 264-311
src/graph_construction.py            138     32    77%   170-184, 202-220
src/logging_setup.py                   8      0   100%
src/graph_tradeoff_analysis.py                44     44     0%   12-83
src/ui_theme.py                       21     21     0%   8-145
----------------------------------------------------------------
TOTAL                                879    291    67%
```

Every module's uncovered lines were checked by hand; in every case they fall into one
of two categories, not untested decision logic:

- `if __name__ == "__main__":` demo/CLI blocks (every `src/` module has one, used for
  standalone `python -m src.module_name` runs during development — not imported or
  exercised by `app.py` or by any other module)
- `matplotlib` plotting functions (`plot_eda`, `plot_degree_distributions`,
  `plot_loss_curves`) that produce visual `.png` output — correctness here is "does the
  plot look right," not something a unit test meaningfully asserts

`ui_theme.py` (0%) is pure Streamlit markup/CSS helpers with no conditional logic to
break — verified separately via the Playwright UI-screenshot workflow
(`.claude/skills/developing-with-streamlit/`), not via `pytest`. `graph_tradeoff_analysis.py`
(0%) is the one-off analysis script that already produced Phase 7's findings
(`data/processed/graph_tradeoff_*.csv`) and isn't called from `app.py`.

## 6. Flaky test found and fixed after the initial pass

`test_deterministic_given_same_graph_and_weights` (added in §4 to close a mutation gap)
was itself flaky — it failed roughly 4 times out of 5 on repeated runs. Root cause: it
compared the entire `metrics` dict returned by two back-to-back calls, including
`elapsed_seconds` and `peak_memory_mb` — real wall-clock and RSS measurements that are
*supposed* to differ run to run, not outputs of the algorithm being tested. The
meaningful outputs (`fixed_test_nodes`, node/edge counts, retention percentages,
community count) were genuinely deterministic the whole time; the test just asserted
too much. Fixed by excluding those two diagnostic-only keys from the comparison and
re-verified stable across 6 consecutive runs. A reminder that a test surviving mutation
testing isn't sufficient — it also has to be correct about what it's actually claiming.

## 7. Overall functional & quality verification status

- **Functional requirements**: verified against real data via the acceptance suite —
  documented row counts, category distributions, feature dimensions, graph topology
  properties, and Black Hole retention percentages all match what `README.md` /
  `CASE_STUDY_REPORT.md` claim, within stated tolerances.
- **Quality requirements**: 243/243 tests passing across 4 independent testing
  methodologies; 3 real production bugs found and fixed during this phase (all in
  edge-case handling, none affecting the documented main-path results) plus 1 flaky
  test found and fixed afterward (§6); mutation testing raised confidence in the most
  correctness-sensitive module (`black_hole_sparsification.py`) from 68.5% to 84.4%
  mutant-kill rate via targeted, judgment-based test additions rather than blindly
  maximizing the score.
- **Known, accepted gaps**: `config.py`'s dataclass methods have no mutation coverage
  due to a tool limitation (still 100% line-covered, 41 unit tests); `ui_theme.py` and
  `graph_tradeoff_analysis.py` have 0% `pytest` coverage by design (verified through other
  means or already validated manually); the remaining 68 mutation survivors in
  `black_hole_sparsification.py` were reviewed individually and judged not to represent
  real risk.
