"""Phase 6 — Baseline Comparison.

Non-graph baselines (Random Forest, k-NN) trained on the flat Phase-2 feature matrix —
no graph structure at all. Evaluated on the exact same `fixed_test_nodes` used by the
GNNs (Phase 4/5), via the same `make_splits()` from gnn_training.py, so the master
comparison table is a genuine apples-to-apples comparison rather than similar-sounding
numbers computed on different splits.

No validation split is carved out here — unlike the GNNs, these models don't need one
for early stopping (k-NN has no training loop at all; Random Forest isn't tuned beyond
the reference code's fixed hyperparameters). Trained on train+val combined instead.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from gnn_training import CATEGORY_ORDER, make_splits
from logging_setup import get_logger

logger = get_logger(__name__)


def run_baseline_comparison(df, features: np.ndarray, task: str, fixed_test_nodes: set, return_models: bool = False) -> dict:
    """Returns {"Random Forest": {...}, "k-NN": {...}}.

    With return_models=True, each entry also carries the fitted estimator under "model" —
    used by webapp/backend/model_store.py to persist trained baselines for later inference,
    without duplicating the train/split logic here.
    """
    n = len(features)
    train_idx, val_idx, test_idx = make_splits(n, fixed_test_nodes)
    train_all_idx = np.concatenate([train_idx, val_idx])

    if task == "classification":
        cat_to_code = {c: i for i, c in enumerate(CATEGORY_ORDER)}
        labels = df["pld_category"].map(cat_to_code).values
    else:
        labels = df["pld_value"].values.astype(np.float32)

    X_train, y_train = features[train_all_idx], labels[train_all_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    if task == "classification":
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "k-NN": KNeighborsClassifier(n_neighbors=5),
        }
    else:
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "k-NN": KNeighborsRegressor(n_neighbors=5),
        }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if task == "classification":
            metrics = {
                "accuracy": float((y_pred == y_test).mean()),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, y_pred, labels=range(len(CATEGORY_ORDER))).tolist(),
                "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
            }
        else:
            metrics = {
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2": float(r2_score(y_test, y_pred)),
            }
        logger.info(f"Baseline {name}: {metrics}")
        results[name] = {"metrics": metrics, "model": model} if return_models else {"metrics": metrics}

    return results


def build_master_comparison_table(gnn_results: dict, baseline_results: dict, task: str):
    """Combines Phase 5's GNN results and Phase 6's baseline results into one table."""
    import pandas as pd

    rows = []
    for (variant, model_name), r in gnn_results.items():
        row = {"Configuration": f"{variant} — {model_name}", "Type": "GNN"}
        row.update({k: v for k, v in r["metrics"].items() if k != "confusion_matrix"})
        rows.append(row)
    for model_name, r in baseline_results.items():
        row = {"Configuration": f"Flat features — {model_name}", "Type": "Baseline"}
        row.update({k: v for k, v in r["metrics"].items() if k != "confusion_matrix"})
        rows.append(row)

    table = pd.DataFrame(rows)
    sort_col = "accuracy" if task == "classification" else "r2"
    return table.sort_values(sort_col, ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from black_hole_sparsification import apply_black_hole_sparsification
    from data_ingestion import load_dataset
    from feature_engineering import build_features
    from gnn_training import run_gnn_comparison
    from graph_construction import build_similarity_graphs, compute_topology_metrics, select_best_graph

    for dataset_name in ("small", "large"):
        print("=" * 60)
        print(f"{dataset_name.upper()} dataset")
        print("=" * 60)
        df = load_dataset(dataset_name)
        scheme = "compact" if dataset_name == "small" else "fingerprint"
        features, _ = build_features(df, scheme)
        graphs = build_similarity_graphs(df, dataset_name)
        topology = compute_topology_metrics(graphs)
        best_name = select_best_graph(topology)
        best_graph = graphs[best_name]

        gnn_results = run_gnn_comparison(df, features, best_graph, "classification", (0.33, 0.33, 0.33))
        bh30 = apply_black_hole_sparsification(best_graph, df, (0.33, 0.33, 0.33), 0.3)
        fixed_test_nodes = bh30["fixed_test_nodes"]

        baseline_results = run_baseline_comparison(df, features, "classification", fixed_test_nodes)
        for name, r in baseline_results.items():
            print(name, r["metrics"])

        table = build_master_comparison_table(gnn_results, baseline_results, "classification")
        print(table.to_string(index=False))
        print()
