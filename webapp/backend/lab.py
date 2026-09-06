"""In-process cache for the pipeline objects the Lab endpoints (and GNN inductive
insertion, Phase 3) both need: loaded dataframe, feature matrix, candidate similarity
graphs, and Black Hole-pruned variants.

Deliberately not st.cache_data (this isn't Streamlit) — a plain dict keyed by the
arguments that actually change the result, mirroring the same "expensive step, cheap
repeat" reasoning app.py used. Process-lifetime cache, not persisted to disk; a server
restart recomputes on first request (a few seconds for the small dataset, up to ~10s for
the large dataset's similarity computation — see graph_construction.py's own docstring,
though a *cold* cache under real concurrent load can take much longer — see the locking
note below).

Real bug found and fixed: with no locking, two concurrent requests for the same
not-yet-cached key (e.g. the frontend's own Graph Construction step fires
`/api/lab/{dataset}/graph` and `/api/lab/{dataset}/graph-sample` at once via
`Promise.all`, both racing to populate the same cache entry) each independently ran the
full ~50-70s large-dataset similarity computation, competing for the same CPU cores and
inflating what should be a single ~10s computation into minutes. `_cached()` below uses a
per-(cache, key) lock with double-checked locking: the second caller blocks until the
first finishes, then reads the now-populated cache instead of recomputing. Lock identity
is keyed on `id(cache)` as well as the cache key itself — using the key alone would let
e.g. `get_df("large")` and `get_dataset_and_features("large")` share a lock despite the
latter calling the former internally, deadlocking on Python's non-reentrant `Lock`.
"""

import threading

import networkx as nx

from black_hole_sparsification import apply_black_hole_sparsification  # noqa: E402
from data_ingestion import load_dataset  # noqa: E402
from feature_engineering import build_features  # noqa: E402
from graph_construction import (  # noqa: E402
    build_similarity_graphs,
    compute_topology_metrics,
    sample_subgraph_for_viz,
    select_best_graph,
)

FEATURE_SCHEME_BY_DATASET = {"small": "compact", "large": "fingerprint"}
DEFAULT_GRAVITY_WEIGHTS = (0.33, 0.33, 0.33)
GRAPH_SAMPLE_MAX_NODES = 70

_df_cache: dict = {}
_features_cache: dict = {}
_graphs_cache: dict = {}
_bh_cache: dict = {}
_sample_cache: dict = {}

_meta_lock = threading.Lock()
_key_locks: dict = {}


def _lock_for(cache: dict, key) -> threading.Lock:
    lock_key = (id(cache), key)
    with _meta_lock:
        if lock_key not in _key_locks:
            _key_locks[lock_key] = threading.Lock()
        return _key_locks[lock_key]


def _cached(cache: dict, key, compute):
    """Double-checked locking: avoids two concurrent callers both recomputing the same
    not-yet-cached, expensive value (see module docstring)."""
    if key in cache:
        return cache[key]
    with _lock_for(cache, key):
        if key not in cache:
            cache[key] = compute()
        return cache[key]


def get_df(dataset: str):
    """Just the loaded, cleaned dataframe — cheap, no feature/graph building. Used by
    lightweight endpoints (dataset stats) that don't need the rest of the pipeline."""
    return _cached(_df_cache, dataset, lambda: load_dataset(dataset))


def get_dataset_and_features(dataset: str):
    def compute():
        df = get_df(dataset)
        features, feat_meta = build_features(df, FEATURE_SCHEME_BY_DATASET[dataset])
        return (df, features, feat_meta)
    return _cached(_features_cache, dataset, compute)


def get_graphs_and_topology(dataset: str):
    def compute():
        df, _, _ = get_dataset_and_features(dataset)
        graphs = build_similarity_graphs(df, dataset)
        topology = compute_topology_metrics(graphs)
        best = select_best_graph(topology)
        return (graphs, topology, best)
    return _cached(_graphs_cache, dataset, compute)


def get_best_graph(dataset: str):
    graphs, _, best = get_graphs_and_topology(dataset)
    return graphs[best], best


def get_pruned_graph(dataset: str, tau: float, weights: tuple = DEFAULT_GRAVITY_WEIGHTS) -> dict:
    key = (dataset, round(tau, 4), weights)

    def compute():
        df, _, _ = get_dataset_and_features(dataset)
        best_graph, _ = get_best_graph(dataset)
        return apply_black_hole_sparsification(best_graph, df, weights, tau)
    return _cached(_bh_cache, key, compute)


def get_graph_variant(dataset: str, variant_name: str, weights: tuple = DEFAULT_GRAVITY_WEIGHTS):
    """Reconstructs one of the 3 graph variants run_gnn_comparison trains on:
    "Full graph" (Phase 3's pick, unpruned), "BH-30 (τ=0.3)", or "BH-50 (τ=0.5)"."""
    if variant_name == "Full graph":
        graph, _ = get_best_graph(dataset)
        return graph
    if variant_name.startswith("BH-30"):
        return get_pruned_graph(dataset, 0.3, weights)["graph"]
    if variant_name.startswith("BH-50"):
        return get_pruned_graph(dataset, 0.5, weights)["graph"]
    raise ValueError(f"Unknown graph variant {variant_name!r}")


def get_graph_sample(dataset: str):
    """A fixed, readable node/edge sample of the Phase-3 best graph, laid out once via
    spring_layout and cached — so the "before" (Graph Construction) and "after" (Black
    Hole) views in the Lab render the *same* nodes at the *same* positions, matching how
    the original Streamlit app's graph_viz.py behaved: a pruned node visibly disappears
    from its spot rather than the whole layout re-shuffling."""
    def compute():
        graph, best_name = get_best_graph(dataset)
        node_ids = sample_subgraph_for_viz(graph, max_nodes=GRAPH_SAMPLE_MAX_NODES)
        sub = graph.subgraph(node_ids)
        pos = nx.spring_layout(sub, seed=42, weight="weight")
        return {"node_ids": node_ids, "pos": pos, "graph": graph, "best_graph": best_name}
    return _cached(_sample_cache, dataset, compute)
