"""Phase 5 — GNN Model Training.

Ports GCN and GraphSAGE from graphsage_model.py, and the train()/test() loop — with one
significant methodological fix.

Real bug found in the reference code: `train()` sets `val_mask = data.test_mask` — i.e.
early stopping (which epoch's weights get kept) is driven by the *same* set used for
final evaluation. That's test-set leakage: the stopping decision is informed by test
performance, which inflates the reported test metric. Fixed here with a genuine 3-way
split — Phase 4's `fixed_test_nodes` stay untouched as the test set (never seen during
training or early stopping); a separate validation subset is carved out of the
remaining nodes purely for early stopping.

GAT is now included as a third model, matching the paper's own evaluation (which tests
GAT, GCN, and GraphSAGE) — an earlier note here claimed the professor had confirmed a
2-model comparison was sufficient, but that was never actually confirmed; it was a
question drafted for a meeting that didn't end up covering it. Corrected.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx.algorithms.community import louvain_communities
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv

from black_hole_sparsification import apply_black_hole_sparsification, calculate_gravity_per_community
from logging_setup import get_logger

logger = get_logger(__name__)

CATEGORY_ORDER = ["nonporous", "small pore", "medium pore", "large pore"]


class GCN(nn.Module):
    def __init__(self, dim_in: int, dim_h: int, dim_out: int):
        super().__init__()
        self.conv1 = GCNConv(dim_in, dim_h)
        self.conv2 = GCNConv(dim_h, dim_out)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv2(h, edge_index, edge_weight)
        return h if self.conv2.out_channels == 1 else F.log_softmax(h, dim=-1)


class GraphSAGE(nn.Module):
    """Custom sparse mean-aggregation implementation, ported as-is from graphsage_model.py
    (the reference code's own hand-rolled version, not PyG's built-in SAGEConv)."""

    def __init__(self, dim_in: int, dim_h: int, dim_out: int):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_out)

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        device = x.device
        self_loops = torch.stack([torch.arange(num_nodes, device=device)] * 2, dim=0)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device)
            self_weight = torch.ones(num_nodes, device=device)
        else:
            self_weight = torch.ones(num_nodes, device=device) * edge_weight.mean()
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        edge_weight = torch.cat([edge_weight, self_weight], dim=0)
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
        degree = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
        norm = 1.0 / degree
        norm_adj = torch.sparse_coo_tensor(edge_index, edge_weight * norm[edge_index[0]], (num_nodes, num_nodes))
        h = torch.sparse.mm(norm_adj, x)
        h = self.linear1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = torch.sparse.mm(norm_adj, h)
        h = self.linear2(h)
        return h if self.linear2.out_features == 1 else F.log_softmax(h, dim=-1)


class GAT(nn.Module):
    """Ported as-is from graphsage_model.py. Note: GATConv (unlike GCNConv) doesn't take
    edge_weight in the reference code's usage — attention weights are learned instead, so
    edge_weight is accepted here (for a consistent call signature across all 3 models)
    but genuinely ignored, matching the original."""

    def __init__(self, dim_in: int, dim_h: int, dim_out: int, heads: int = 12):
        super().__init__()
        self.conv1 = GATConv(dim_in, dim_h, heads=heads, dropout=0.3)
        self.norm1 = nn.LayerNorm(dim_h * heads)
        self.conv2 = GATConv(dim_h * heads, dim_out, heads=1, concat=False, dropout=0.3)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv2(h, edge_index)
        return h if self.conv2.out_channels == 1 else F.log_softmax(h, dim=-1)


def graph_to_edge_arrays(graph, n_nodes: int):
    edges = list(graph.edges(data=True))
    if not edges:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    src = [u for u, v, d in edges] + [v for u, v, d in edges]
    dst = [v for u, v, d in edges] + [u for u, v, d in edges]
    w = [d["weight"] for u, v, d in edges] * 2
    return np.array([src, dst]), np.array(w, dtype=np.float32)


def make_splits(n: int, test_indices: set, val_fraction: float = 0.15, seed: int = 42):
    """Test indices are fixed (Phase 4's fixed_test_nodes) and never used for anything
    but final evaluation. A genuine validation split is carved out of the remainder,
    used only for early stopping — separate from both train and test."""
    rng = np.random.RandomState(seed)
    remaining = np.array([i for i in range(n) if i not in test_indices])
    rng.shuffle(remaining)
    n_val = max(1, int(len(remaining) * val_fraction))
    return remaining[n_val:], remaining[:n_val], np.array(sorted(test_indices))


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CATEGORY_ORDER))
    weights = 1.0 / np.sqrt(counts + 1e-6)
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float)


def build_pyg_data(features: np.ndarray, edge_index: np.ndarray, edge_weight: np.ndarray,
                    labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
                    test_idx: np.ndarray, task: str) -> Data:
    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )
    data.edge_weight = torch.tensor(edge_weight, dtype=torch.float) if edge_weight.size else None
    data.y = torch.tensor(labels, dtype=torch.long if task == "classification" else torch.float)
    data.train_mask = torch.tensor(train_idx, dtype=torch.long)
    data.val_mask = torch.tensor(val_idx, dtype=torch.long)
    data.test_mask = torch.tensor(test_idx, dtype=torch.long)
    return data


def train_model(model: nn.Module, data: Data, task: str, epochs: int = 200, lr: float = 0.005,
                 weight_decay: float = 1e-3, patience: int = 20, class_weights: torch.Tensor = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight)
        if task == "classification":
            train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        else:
            train_loss = F.mse_loss(out[data.train_mask].squeeze(-1), data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(data.x, data.edge_index, data.edge_weight)
            if task == "classification":
                val_loss = F.nll_loss(out_eval[data.val_mask], data.y[data.val_mask]).item()
            else:
                val_loss = F.mse_loss(out_eval[data.val_mask].squeeze(-1), data.y[data.val_mask]).item()

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_model(model: nn.Module, data: Data, task: str) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight)
        test_mask = data.test_mask
        if task == "classification":
            pred = out.argmax(dim=1)
            y_true = data.y[test_mask].cpu().numpy()
            y_pred = pred[test_mask].cpu().numpy()
            return {
                "accuracy": float((y_pred == y_true).mean()),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(CATEGORY_ORDER))).tolist(),
                "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
            }
        y_true = data.y[test_mask].cpu().numpy()
        y_pred = out.squeeze(-1)[test_mask].cpu().numpy()
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        }


def run_gnn_comparison(df, features: np.ndarray, best_graph, task: str, gravity_weights: tuple,
                        return_models: bool = False) -> dict:
    """Trains GCN + GraphSAGE + GAT on: the Phase-3 best graph, BH-30 (tau=0.3), BH-50 (tau=0.5).
    Returns {(variant_name, model_name): {"metrics": ..., "history": ...}}.

    With return_models=True, each entry also carries the trained model, the PyG Data it was
    trained on, and its dim_in/dim_h/dim_out — used by webapp/backend/gnn_store.py to persist
    a servable model per architecture without duplicating the train/split/graph-variant logic
    here."""
    n = len(df)

    if task == "classification":
        cat_to_code = {c: i for i, c in enumerate(CATEGORY_ORDER)}
        labels = df["pld_category"].map(cat_to_code).values
        class_weights = compute_class_weights(labels)
        dim_out = len(CATEGORY_ORDER)
    else:
        labels = df["pld_value"].values.astype(np.float32)
        class_weights = None
        dim_out = 1

    bh30 = apply_black_hole_sparsification(best_graph, df, gravity_weights, 0.3)
    bh50 = apply_black_hole_sparsification(best_graph, df, gravity_weights, 0.5)
    # Same fixed_test_nodes by construction (deterministic given the same graph + weights) —
    # reused across all 3 variants so the test set is identical for a fair comparison.
    fixed_test_nodes = bh30["fixed_test_nodes"]

    graph_variants = {
        "Full graph": best_graph,
        "BH-30 (τ=0.3)": bh30["graph"],
        "BH-50 (τ=0.5)": bh50["graph"],
    }

    results = {}
    for variant_name, graph in graph_variants.items():
        edge_index, edge_weight = graph_to_edge_arrays(graph, n)
        train_idx, val_idx, test_idx = make_splits(n, fixed_test_nodes)
        data = build_pyg_data(features, edge_index, edge_weight, labels, train_idx, val_idx, test_idx, task)

        for model_name, model_cls in (("GCN", GCN), ("GraphSAGE", GraphSAGE), ("GAT", GAT)):
            model = model_cls(features.shape[1], 64, dim_out)
            model, history = train_model(model, data, task, class_weights=class_weights)
            metrics = evaluate_model(model, data, task)
            logger.info(f"{variant_name} / {model_name}: {metrics}")
            entry = {"metrics": metrics, "history": history}
            if return_models:
                entry.update({"model": model, "dim_in": features.shape[1], "dim_h": 64, "dim_out": dim_out})
            results[(variant_name, model_name)] = entry

    return results


def plot_loss_curves(results: dict, dataset: str, output_dir: str = "outputs") -> str:
    os.makedirs(output_dir, exist_ok=True)
    variants = sorted({k[0] for k in results}, key=lambda v: ("Full" not in v, v))
    models = sorted({k[1] for k in results})

    fig, axes = plt.subplots(len(models), len(variants), figsize=(4 * len(variants), 3.5 * len(models)), squeeze=False)
    for row, model_name in enumerate(models):
        for col, variant_name in enumerate(variants):
            ax = axes[row][col]
            history = results[(variant_name, model_name)]["history"]
            ax.plot(history["train_loss"], label="train")
            ax.plot(history["val_loss"], label="val")
            ax.set_title(f"{model_name} — {variant_name}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=8)
    fig.suptitle(f"Training/validation loss — {dataset} dataset")
    fig.tight_layout()
    path = os.path.join(output_dir, f"phase5_loss_curves_{dataset}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved loss curve plot to {path}")
    return path


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from data_ingestion import load_dataset
    from feature_engineering import build_features
    from graph_construction import build_similarity_graphs, compute_topology_metrics, select_best_graph

    for dataset_name in ("small", "large"):
        print("=" * 60)
        print(f"{dataset_name.upper()} dataset")
        print("=" * 60)
        df = load_dataset(dataset_name)
        features, _ = build_features(df, "compact" if dataset_name == "small" else "fingerprint")
        graphs = build_similarity_graphs(df, dataset_name)
        topology = compute_topology_metrics(graphs)
        best_name = select_best_graph(topology)
        best_graph = graphs[best_name]

        results = run_gnn_comparison(df, features, best_graph, "classification", (0.33, 0.33, 0.33))
        for (variant, model_name), r in results.items():
            print(f"{variant:16s} {model_name:10s} {r['metrics']}")
        print()
