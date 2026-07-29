"""Phase 3 — Graph construction experiment.

Builds 4 candidate MOF-similarity graphs and compares their topology:
  - Branch A: fixed-threshold similarity graph (phi = 0.9), porting the weighted-similarity
    formula from Similarity.py: sim = alpha * metal_similarity + (1 - alpha) * linker_Tanimoto,
    alpha = 0.1.
  - Branch B: k-NN adjacency, k in {3, 5, 10} — guarantees every node has >= k edges,
    directly testing the original paper's own noted isolated-node problem at strict thresholds.

Deliberate adaptation from Similarity.py: that code computes metal similarity via cosine
distance over 6 numeric metal-property columns, which only exist in the small dataset.
The large dataset has no such per-row numeric metal descriptor — only a categorical metal
name (53 unique values, see Phase 2). For the large dataset, metal similarity is instead a
simple same/different indicator (1.0 if two MOFs share a metal, else 0.0). This is a
documented simplification, not a hidden one: it captures "same coordination chemistry"
without fabricating a periodic-table properties table that isn't part of the given data.

Performance: Similarity.py computes pairwise Tanimoto with one RDKit call per pair in a
raw nested Python loop (fine for 2,000 x 2,000, far too slow for 14,296 x 14,296 = ~204M
pairs). Reimplemented as a single vectorized matrix multiplication of the 0/1 fingerprint
bit-matrix (intersection = A @ A.T, union = row_sums broadcast - intersection), processed
in row blocks to keep memory bounded instead of ever materializing the full N x N matrix.
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity as nx_modularity

from feature_engineering import compute_fingerprint_matrix
from logging_setup import get_logger

logger = get_logger(__name__)

ALPHA = 0.1  # weight on metal similarity vs. linker similarity, per Similarity.py
THRESHOLD_PHI = 0.9
KNN_VALUES = (3, 5, 10)
BLOCK_SIZE = 1500


def _metal_similarity_fn(df: pd.DataFrame, dataset: str):
    """Returns a function(row_slice) -> (block_rows, N) similarity block for the metal component."""
    if dataset == "small":
        cols = [f"metal_feat_{i}" for i in range(6)]
        vecs = df[cols].values.astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit_vecs = vecs / norms  # dot product of unit vectors = cosine similarity

        def metal_sim_block(row_slice):
            return unit_vecs[row_slice] @ unit_vecs.T

        return metal_sim_block

    metals = df["metal"].values
    unique = sorted(set(metals))
    code_map = {m: i for i, m in enumerate(unique)}
    codes = np.array([code_map[m] for m in metals])

    def metal_sim_block(row_slice):
        return (codes[row_slice][:, None] == codes[None, :]).astype(np.float32)

    return metal_sim_block


def build_similarity_graphs(df: pd.DataFrame, dataset: str) -> dict[str, nx.Graph]:
    """Builds all 4 candidate graphs in one pass over the (blocked) similarity matrix."""
    n = len(df)
    fingerprints = compute_fingerprint_matrix(df["linker_smiles"])
    fp_row_sums = fingerprints.sum(axis=1)
    metal_sim_block_fn = _metal_similarity_fn(df, dataset)
    # Found via torture testing: np.argpartition requires kth < n, so a dataset smaller
    # than max(KNN_VALUES) + 1 rows would crash with a cryptic "kth out of bounds" error.
    # Clamp to what's actually available -- a tiny dataset just gets fewer neighbors per
    # node than requested (min(k, n-1)), not a crash.
    max_k = min(max(KNN_VALUES), max(0, n - 1))

    threshold_edges: list[tuple[int, int, float]] = []
    knn_candidates: dict[int, list[list[tuple[int, float]]]] = {k: [[] for _ in range(n)] for k in KNN_VALUES}

    start = time.time()
    for block_start in range(0, n, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, n)
        row_slice = slice(block_start, block_end)

        intersection = fingerprints[row_slice] @ fingerprints.T
        union = fp_row_sums[row_slice][:, None] + fp_row_sums[None, :] - intersection
        union[union == 0] = 1.0
        tanimoto_block = intersection / union

        sim_block = ALPHA * metal_sim_block_fn(row_slice) + (1 - ALPHA) * tanimoto_block

        local_rows = np.arange(block_end - block_start)
        global_rows = local_rows + block_start
        sim_block[local_rows, global_rows] = -1.0  # mask self-similarity

        # Branch A: threshold (keep i < j only, to avoid double-counting undirected edges)
        rows_idx, cols_idx = np.where(sim_block >= THRESHOLD_PHI)
        for r, c in zip(rows_idx, cols_idx):
            gi = r + block_start
            if gi < c:
                threshold_edges.append((gi, int(c), float(sim_block[r, c])))

        # Branch B: top-k per row, for every k in KNN_VALUES at once
        top_idx = np.argpartition(-sim_block, max_k, axis=1)[:, :max_k]
        for local_r in range(block_end - block_start):
            gi = local_r + block_start
            candidates = top_idx[local_r]
            sims = sim_block[local_r, candidates]
            order = np.argsort(-sims)
            candidates, sims = candidates[order], sims[order]
            for k in KNN_VALUES:
                for c, s in zip(candidates[:k], sims[:k]):
                    knn_candidates[k][gi].append((int(c), float(s)))

        logger.info(f"[{dataset}] similarity block {block_end}/{n} rows done")

    logger.info(f"[{dataset}] similarity computation took {time.time() - start:.1f}s for {n} nodes")

    graphs: dict[str, nx.Graph] = {}

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_weighted_edges_from(threshold_edges)
    graphs[f"threshold_{THRESHOLD_PHI}"] = g

    for k in KNN_VALUES:
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for i, neighbors in enumerate(knn_candidates[k]):
            for j, w in neighbors:
                g.add_edge(i, j, weight=w)
        graphs[f"knn_{k}"] = g

    return graphs


def compute_topology_metrics(graphs: dict[str, nx.Graph]) -> pd.DataFrame:
    rows = []
    for name, g in graphs.items():
        n = g.number_of_nodes()
        degrees = dict(g.degree())
        isolated = sum(1 for d in degrees.values() if d == 0)
        communities = louvain_communities(g, seed=42)
        rows.append({
            "config": name,
            "edges": g.number_of_edges(),
            "isolated_node_rate": isolated / n if n else 0.0,
            "mean_degree": sum(degrees.values()) / n if n else 0.0,
            "num_communities": len(communities),
            "modularity": nx_modularity(g, communities) if g.number_of_edges() else 0.0,
        })
    return pd.DataFrame(rows).set_index("config")


def select_best_graph(topology: pd.DataFrame) -> str:
    """Simple, transparent rule: among configs tied for the lowest isolated-node rate,
    pick the one with the highest modularity (best community structure)."""
    min_isolated = topology["isolated_node_rate"].min()
    candidates = topology[topology["isolated_node_rate"] == min_isolated]
    return candidates["modularity"].idxmax()


def plot_degree_distributions(graphs: dict[str, nx.Graph], dataset: str, output_dir: str = "outputs") -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(graphs), figsize=(4 * len(graphs), 4))
    for ax, (name, g) in zip(axes, graphs.items()):
        degrees = [d for _, d in g.degree()]
        ax.hist(degrees, bins=30, color="#4C72B0")
        ax.set_title(name)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Count")
    fig.suptitle(f"Degree distributions — {dataset} dataset")
    fig.tight_layout()
    path = os.path.join(output_dir, f"phase3_degree_distributions_{dataset}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved degree distribution plot to {path}")
    return path


def save_graphs(graphs: dict[str, nx.Graph], refcodes: np.ndarray, topology: pd.DataFrame,
                 best: str, dataset: str, output_dir: str = "data/processed") -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name, g in graphs.items():
        edges = [(refcodes[i], refcodes[j], d["weight"]) for i, j, d in g.edges(data=True)]
        pd.DataFrame(edges, columns=["source", "target", "weight"]).to_csv(
            os.path.join(output_dir, f"graph_{dataset}_{name}.csv"), index=False
        )
    topology.to_csv(os.path.join(output_dir, f"phase3_topology_{dataset}.csv"))
    with open(os.path.join(output_dir, f"phase3_best_graph_{dataset}.json"), "w") as f:
        json.dump({"dataset": dataset, "best_graph": best}, f, indent=2)
    logger.info(f"Saved {len(graphs)} graph edge lists + topology table for {dataset} dataset to {output_dir}/")


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    from data_ingestion import load_dataset

    for dataset_name in ("small", "large"):
        print("=" * 60)
        print(f"{dataset_name.upper()} dataset")
        print("=" * 60)
        df = load_dataset(dataset_name)
        graphs = build_similarity_graphs(df, dataset_name)
        topology = compute_topology_metrics(graphs)
        print(topology)
        best = select_best_graph(topology)
        print(f"\nSelected best graph: {best}")
        plot_degree_distributions(graphs, dataset_name)
        save_graphs(graphs, df["refcode"].values, topology, best, dataset_name)
        print()
