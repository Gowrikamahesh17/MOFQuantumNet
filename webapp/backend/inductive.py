"""Inserts a single new MOF into an already-trained GNN's graph and runs one forward pass.

The GNNs were trained transductively over a fixed similarity graph — a brand-new MOF
doesn't exist in it. To predict on one, this module:
  1. Computes the new MOF's similarity to every existing node using the *same* formula
     graph_construction.py uses to build the graph in the first place (metal similarity,
     weighted ALPHA, plus linker Tanimoto over Morgan fingerprints).
  2. Attaches it via its k=3 nearest surviving neighbors in whichever graph variant the
     target model was actually trained on (matches knn_3 — the Phase 3 pick for both
     datasets, see planning/CASE_STUDY_REPORT.md) — restricted to nodes still present in
     that variant, since a BH-pruned graph has fewer of them.
  3. Appends the new node's feature row, runs one forward pass over the augmented graph,
     reads off the prediction for just that new node.

Cost: step 1 is a single row x N matrix multiply (cheap even at N=14,296 — the same
vectorized approach graph_construction.py's own docstring describes for the full N x N
case, just one row of it). Step 3 is a forward pass through a model with well under 1M
parameters (see the accuracy-gap investigation earlier in this project's planning
thread) — sub-second once the graph + feature matrix are cached (webapp/backend/lab.py).
"""

import numpy as np
import torch

from feature_engineering import _morgan_fingerprint, compute_fingerprint_matrix  # noqa: E402
from graph_construction import ALPHA, _metal_similarity_fn  # noqa: E402

from webapp.backend.gnn_store import load_gnn_metadata, load_gnn_model  # noqa: E402
from webapp.backend.lab import DEFAULT_GRAVITY_WEIGHTS, get_dataset_and_features, get_graph_variant  # noqa: E402

K_NEIGHBORS = 3  # matches knn_3, the Phase 3 pick for both datasets

_fingerprint_cache: dict = {}


def _linker_fingerprints(dataset: str) -> np.ndarray:
    if dataset not in _fingerprint_cache:
        df, _, _ = get_dataset_and_features(dataset)
        _fingerprint_cache[dataset] = compute_fingerprint_matrix(df["linker_smiles"])
    return _fingerprint_cache[dataset]


def _similarity_to_existing_nodes(dataset: str, smiles: str, metal: str | None, metal_feat: list[float] | None) -> np.ndarray:
    """One row of graph_construction.py's similarity formula: new MOF vs. every existing node."""
    df, _, _ = get_dataset_and_features(dataset)
    fingerprints = _linker_fingerprints(dataset)
    fp_row_sums = fingerprints.sum(axis=1)

    new_fp = _morgan_fingerprint(smiles)
    intersection = fingerprints @ new_fp
    union = fp_row_sums + new_fp.sum() - intersection
    union = np.where(union == 0, 1.0, union)
    tanimoto = intersection / union

    if dataset == "small":
        cols = [f"metal_feat_{i}" for i in range(6)]
        vecs = df[cols].values.astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1)
        norms = np.where(norms == 0, 1.0, norms)
        unit_vecs = vecs / norms[:, None]
        new_vec = np.array(metal_feat, dtype=np.float32) if metal_feat is not None else vecs.mean(axis=0)
        new_norm = np.linalg.norm(new_vec) or 1.0
        metal_sim = unit_vecs @ (new_vec / new_norm)
    else:
        metal_sim = (df["metal"].values == metal).astype(np.float32)

    return ALPHA * metal_sim + (1 - ALPHA) * tanimoto


def predict_with_gnn(dataset: str, task: str, model_name: str, feature_vector: np.ndarray,
                      smiles: str, metal: str | None = None, metal_feat: list[float] | None = None) -> dict:
    """feature_vector must already be in the same space webapp/backend/featurize.py
    produces (shape (1, dim_in)) — the same feature matrix the model was trained on."""
    gnn_meta = load_gnn_metadata(dataset, task)
    model, info = load_gnn_model(dataset, task, model_name)
    variant_name = info["variant"]

    graph = get_graph_variant(dataset, variant_name, tuple(gnn_meta["gravity_weights"]))
    _, features, _ = get_dataset_and_features(dataset)

    surviving = np.array(sorted(graph.nodes()))
    sim = _similarity_to_existing_nodes(dataset, smiles, metal, metal_feat)
    sim_surviving = sim[surviving]
    top_k = min(K_NEIGHBORS, len(surviving))
    nearest_local = np.argpartition(-sim_surviving, top_k - 1)[:top_k] if top_k > 0 else np.array([], dtype=int)
    neighbor_global_ids = surviving[nearest_local]
    neighbor_weights = sim_surviving[nearest_local]

    n_existing = features.shape[0]
    new_node_id = n_existing  # append at the end

    all_node_ids = list(graph.nodes()) + [new_node_id]
    id_to_index = {nid: i for i, nid in enumerate(all_node_ids)}

    src, dst, w = [], [], []
    for u, v, d in graph.edges(data=True):
        src += [id_to_index[u], id_to_index[v]]
        dst += [id_to_index[v], id_to_index[u]]
        w += [d["weight"], d["weight"]]
    for gid, weight in zip(neighbor_global_ids, neighbor_weights):
        src += [id_to_index[new_node_id], id_to_index[gid]]
        dst += [id_to_index[gid], id_to_index[new_node_id]]
        w += [float(weight), float(weight)]

    x_augmented = np.concatenate([features[all_node_ids[:-1]], feature_vector.astype(np.float32)], axis=0)
    x = torch.tensor(x_augmented, dtype=torch.float)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)

    with torch.no_grad():
        out = model(x, edge_index, edge_weight)

    new_row = out[-1]
    result = {
        "variant_used": variant_name,
        "n_neighbors_found": int(len(neighbor_global_ids)),
        "measured_metrics": info["metrics"],
    }
    if task == "classification":
        probs = new_row.exp().numpy()  # model outputs log_softmax
        result["probs"] = probs
    else:
        result["value"] = float(new_row.squeeze().item())
    return result
