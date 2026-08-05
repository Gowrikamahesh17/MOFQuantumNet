"""Phase 7 — Analysis: does Phase 3's topology-only graph selection actually pick the
best graph for downstream accuracy?

Phase 3's `select_best_graph()` picks a graph using connectivity heuristics alone
(lowest isolated-node rate, then highest modularity) — it never trains a model to
check. This script closes that gap: trains one GCN per Phase-3 graph candidate
(threshold, knn_3, knn_5, knn_10) and reports accuracy/kappa alongside the topology
metrics, so the trade-off can be assessed with real downstream performance, not just
graph structure.
"""

import os
import sys

import pandas as pd
from networkx.algorithms.community import louvain_communities

sys.path.insert(0, os.path.dirname(__file__))
from black_hole_sparsification import calculate_gravity_per_community
from data_ingestion import load_dataset
from feature_engineering import build_features
from gnn_training import (
    CATEGORY_ORDER,
    GCN,
    build_pyg_data,
    compute_class_weights,
    evaluate_model,
    graph_to_edge_arrays,
    make_splits,
    train_model,
)
from graph_construction import build_similarity_graphs, compute_topology_metrics
from logging_setup import get_logger

logger = get_logger(__name__)


def evaluate_graph_construction_tradeoff(dataset: str, scheme: str) -> pd.DataFrame:
    df = load_dataset(dataset)
    features, _ = build_features(df, scheme)
    graphs = build_similarity_graphs(df, dataset)
    topology = compute_topology_metrics(graphs)

    cat_to_code = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    labels = df["pld_category"].map(cat_to_code).values
    class_weights = compute_class_weights(labels)

    rows = []
    for name, graph in graphs.items():
        communities = louvain_communities(graph, seed=42)
        _, _, _, _, fixed_test_nodes = calculate_gravity_per_community(graph, communities, (0.33, 0.33, 0.33))
        edge_index, edge_weight = graph_to_edge_arrays(graph, len(df))
        train_idx, val_idx, test_idx = make_splits(len(df), fixed_test_nodes)
        data = build_pyg_data(features, edge_index, edge_weight, labels, train_idx, val_idx, test_idx, "classification")

        model = GCN(features.shape[1], 64, len(CATEGORY_ORDER))
        model, _ = train_model(model, data, "classification", class_weights=class_weights)
        metrics = evaluate_model(model, data, "classification")

        row = topology.loc[name]
        rows.append({
            "config": name,
            "accuracy": metrics["accuracy"],
            "cohen_kappa": metrics["cohen_kappa"],
            "isolated_node_rate": row["isolated_node_rate"],
            "mean_degree": row["mean_degree"],
            "modularity": row["modularity"],
        })
        logger.info(f"[{dataset}] {name}: accuracy={metrics['accuracy']:.4f}, kappa={metrics['cohen_kappa']:.4f}")

    result = pd.DataFrame(rows).set_index("config")
    result.to_csv(f"data/processed/phase7_graph_tradeoff_{dataset}.csv")
    return result


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    for dataset_name, scheme in (("small", "compact"), ("large", "fingerprint")):
        print("=" * 70)
        print(dataset_name.upper())
        print("=" * 70)
        print(evaluate_graph_construction_tradeoff(dataset_name, scheme))
        print()
