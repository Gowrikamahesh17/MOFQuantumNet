"""Phase 4 — Black Hole Sparsification.

Ports calculate_gravity_per_community(), black_hole_strategy_per_community(), and
prune_edges() from bh_sparsification.py, adapted to:
  - operate on Phase 3's graphs (integer positional node ids 0..N-1, matching the
    dataframe's row order), instead of BlackHole's string-refcode nodes
  - read gravity weights and pruning threshold from PipelineConfig sliders instead of
    hardcoded constants
  - use `pld_category` from our own canonical dataset schema (works for both datasets),
    instead of BlackHole's summary_data['category']

Two deliberate adaptations, not silent deviations:
  1. The original calculate_gravity_per_community() reuses the loop variable `idx` for
     both the outer per-community loop and the inner per-node loop, so a "community has
     no eligible test nodes" warning would log the wrong index. Cosmetic only — doesn't
     affect the actual gravity scores or node selection (both keyed by node id, not by
     that loop variable) — but avoided here with distinct variable names regardless.
  2. black_hole_strategy_per_community() in the original mutates its input graph in place
     via remove_nodes_from(). That's fine for BlackHole's own usage (a fresh bootstrap-
     resampled graph per run), but not for us: our Phase-3 graph is a single cached object
     reused across every threshold/weight combination the UI sliders produce. Copies the
     graph first so repeated calls don't corrupt each other.
"""

import json
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
import psutil
from networkx.algorithms.community import louvain_communities
from sklearn.preprocessing import MinMaxScaler

from logging_setup import get_logger

logger = get_logger(__name__)


def _memory_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


def calculate_gravity_per_community(graph: nx.Graph, communities: list, weights: tuple):
    """Returns (gravity, degree_centrality, betweenness_centrality, edge_weight_sum, fixed_test_nodes)."""
    degree_weight, betweenness_weight, weight_sum_weight = weights
    gravity: dict = {}
    degree_centrality: dict = {}
    betweenness_centrality: dict = {}
    edge_weight_sum: dict = {}
    fixed_test_nodes: set = set()

    for community in communities:
        subgraph = graph.subgraph(community)
        if subgraph.number_of_nodes() == 0:
            continue

        deg_c = nx.degree_centrality(subgraph)
        bet_c = nx.betweenness_centrality(subgraph, normalized=True)
        weight_sum_c = {
            node: sum(data["weight"] for _, _, data in subgraph.edges(node, data=True))
            for node in subgraph.nodes()
        }

        community_list = list(community)
        scaler = MinMaxScaler()
        norm_degree = scaler.fit_transform(np.array([deg_c[n] for n in community_list]).reshape(-1, 1)).flatten()
        norm_betweenness = scaler.fit_transform(np.array([bet_c[n] for n in community_list]).reshape(-1, 1)).flatten()
        norm_weight_sum = scaler.fit_transform(np.array([weight_sum_c[n] for n in community_list]).reshape(-1, 1)).flatten()

        for node_idx, node in enumerate(community_list):
            gravity[node] = (
                degree_weight * norm_degree[node_idx]
                + betweenness_weight * norm_betweenness[node_idx]
                + weight_sum_weight * norm_weight_sum[node_idx]
            )
            degree_centrality[node] = deg_c[node]
            betweenness_centrality[node] = bet_c[node]
            edge_weight_sum[node] = weight_sum_c[node]

        node_degrees = {node: subgraph.degree(node) for node in community}
        eligible = [node for node, d in node_degrees.items() if d > 2]
        if not eligible and node_degrees:
            eligible = [max(node_degrees.items(), key=lambda x: x[1])[0]]
        if eligible:
            ranked = sorted(((n, gravity[n]) for n in eligible), key=lambda x: x[1], reverse=True)
            num_test = max(1, int(0.04 * len(community)))
            fixed_test_nodes.update(n for n, _ in ranked[:num_test])

    return gravity, degree_centrality, betweenness_centrality, edge_weight_sum, fixed_test_nodes


def black_hole_strategy_per_community(graph: nx.Graph, gravity: dict, communities: list,
                                       threshold: float, fixed_test_nodes: set,
                                       pld_categories: pd.Series) -> tuple:
    """PLD-stratified node retention per community. Returns (new_graph, nodes_retained)."""
    graph = graph.copy()  # never mutate the caller's (cached, shared) graph
    selected_nodes = set(fixed_test_nodes)
    nodes_to_remove = []

    n_total = graph.number_of_nodes()
    target_num_nodes = int(max(0.2 * n_total, (1 - threshold) * n_total))
    target_non_test = max(0, target_num_nodes - len(fixed_test_nodes))

    community_sizes = [len(c) for c in communities]
    total_in_communities = sum(community_sizes) or 1
    community_targets = [max(1, int(target_non_test * (size / total_in_communities))) for size in community_sizes]
    diff = target_non_test - sum(community_targets)
    for i in range(abs(diff)):
        idx = i % len(community_targets)
        community_targets[idx] = max(0, community_targets[idx] + (1 if diff > 0 else -1))

    for community, target_count in zip(communities, community_targets):
        community_nodes = [n for n in community if n in gravity and n not in fixed_test_nodes]
        if not community_nodes:
            continue

        categories = pld_categories.iloc[community_nodes]
        category_counts = categories.value_counts(normalize=True).to_dict()
        per_category_target = {cat: max(1, int(target_count * prop)) for cat, prop in category_counts.items()}

        for category, cat_target in per_category_target.items():
            candidates = [n for n in community_nodes if pld_categories.iloc[n] == category]
            ranked = sorted(candidates, key=lambda n: gravity[n], reverse=True)
            selected_nodes.update(ranked[:cat_target])

        nodes_to_remove.extend(n for n in community_nodes if n not in selected_nodes)

    graph.remove_nodes_from(nodes_to_remove)
    return graph, graph.number_of_nodes()


def prune_edges(graph: nx.Graph, edge_threshold: float, fixed_test_nodes: set) -> nx.Graph:
    """Keeps the top (1 - edge_threshold) fraction of edges by weight, guaranteeing every
    fixed test node retains at least its single highest-weight edge."""
    edges = [(u, v, d["weight"]) for u, v, d in graph.edges(data=True)]
    if not edges:
        return graph

    test_edges = [(u, v, w) for u, v, w in edges if u in fixed_test_nodes or v in fixed_test_nodes]
    non_test_edges = [(u, v, w) for u, v, w in edges if u not in fixed_test_nodes and v not in fixed_test_nodes]

    best_per_test_node = {}
    for u, v, w in test_edges:
        for node in (u, v):
            if node in fixed_test_nodes:
                if node not in best_per_test_node or w > best_per_test_node[node][2]:
                    best_per_test_node[node] = (u, v, w)
    test_edges_to_keep = list({(min(u, v), max(u, v)): (u, v, w) for u, v, w in best_per_test_node.values()}.values())

    num_to_keep = max(10, int((1 - edge_threshold) * len(edges))) - len(test_edges_to_keep)
    num_to_keep = max(0, num_to_keep)
    non_test_edges.sort(key=lambda x: x[2], reverse=True)

    new_graph = nx.Graph()
    new_graph.add_nodes_from(graph.nodes())
    new_graph.add_weighted_edges_from(test_edges_to_keep + non_test_edges[:num_to_keep])
    return new_graph


def apply_black_hole_sparsification(graph: nx.Graph, df: pd.DataFrame, weights: tuple, threshold: float) -> dict:
    """Top-level orchestrator: communities -> gravity -> node pruning -> edge pruning."""
    mem_before = _memory_mb()
    start = time.time()

    communities = louvain_communities(graph, seed=42)
    gravity, degree_c, betweenness_c, edge_weight_sum, fixed_test_nodes = calculate_gravity_per_community(
        graph, communities, weights
    )
    pld_categories = df["pld_category"].reset_index(drop=True)

    pruned_graph, nodes_retained = black_hole_strategy_per_community(
        graph, gravity, communities, threshold, fixed_test_nodes, pld_categories
    )
    pruned_graph = prune_edges(pruned_graph, threshold, fixed_test_nodes)

    peak_memory = max(mem_before, _memory_mb())
    elapsed = time.time() - start

    n_before, e_before = graph.number_of_nodes(), graph.number_of_edges()
    n_after, e_after = pruned_graph.number_of_nodes(), pruned_graph.number_of_edges()

    metrics = {
        "nodes_before": n_before,
        "edges_before": e_before,
        "nodes_after": n_after,
        "edges_after": e_after,
        "node_retention_pct": round(100 * n_after / n_before, 2) if n_before else 0.0,
        "edge_retention_pct": round(100 * e_after / e_before, 2) if e_before else 0.0,
        "isolated_nodes_after": sum(1 for _, d in pruned_graph.degree() if d == 0),
        "density_before": nx.density(graph),
        "density_after": nx.density(pruned_graph),
        "num_communities": len(communities),
        "num_fixed_test_nodes": len(fixed_test_nodes),
        "peak_memory_mb": round(peak_memory, 2),
        "elapsed_seconds": round(elapsed, 2),
    }
    logger.info(f"Black Hole sparsification (tau={threshold}, weights={weights}): {metrics}")
    return {"graph": pruned_graph, "metrics": metrics, "fixed_test_nodes": fixed_test_nodes, "gravity": gravity}


def save_sparsified_graph(result: dict, refcodes: np.ndarray, dataset: str, threshold: float,
                           output_dir: str = "data/processed") -> None:
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{dataset}_tau{threshold:.2f}"
    edges = [(refcodes[i], refcodes[j], d["weight"]) for i, j, d in result["graph"].edges(data=True)]
    pd.DataFrame(edges, columns=["source", "target", "weight"]).to_csv(
        os.path.join(output_dir, f"bh_graph_{tag}.csv"), index=False
    )
    with open(os.path.join(output_dir, f"bh_metrics_{tag}.json"), "w") as f:
        json.dump(result["metrics"], f, indent=2)
    logger.info(f"Saved Black Hole sparsified graph + metrics for {tag} to {output_dir}/")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from data_ingestion import load_dataset
    from graph_construction import build_similarity_graphs, compute_topology_metrics, select_best_graph

    for dataset_name in ("small", "large"):
        print("=" * 60)
        print(f"{dataset_name.upper()} dataset")
        print("=" * 60)
        df = load_dataset(dataset_name)
        graphs = build_similarity_graphs(df, dataset_name)
        topology = compute_topology_metrics(graphs)
        best_name = select_best_graph(topology)
        best_graph = graphs[best_name]
        print(f"Best Phase-3 graph: {best_name} ({best_graph.number_of_nodes()} nodes, {best_graph.number_of_edges()} edges)")

        for tau in (0.3, 0.5):
            result = apply_black_hole_sparsification(best_graph, df, weights=(0.33, 0.33, 0.33), threshold=tau)
            print(f"  tau={tau}: {result['metrics']}")
            save_sparsified_graph(result, df["refcode"].values, dataset_name, tau)
        print()
