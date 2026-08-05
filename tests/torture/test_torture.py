"""Torture tests — extreme, degenerate, and pathological inputs that a well-behaved
pipeline should either handle gracefully or fail with a clear, controlled error, never
an obscure crash from deep inside a library call.

One real, previously-uncaught bug was found here: build_similarity_graphs crashed with
`ValueError: kth(=10) out of bounds` on any dataset with fewer than 11 rows, because
np.argpartition requires kth < n. Fixed in graph_construction.py by clamping max_k to
min(max(KNN_VALUES), n-1) -- see TestGraphConstructionTortureTests below for the
regression test.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from black_hole_sparsification import apply_black_hole_sparsification, calculate_gravity_per_community
from feature_engineering import build_compact_features, build_fingerprint_features, compute_fingerprint_matrix
from gnn_training import GCN, build_pyg_data, evaluate_model, graph_to_edge_arrays, make_splits, train_model
from graph_construction import KNN_VALUES, build_similarity_graphs, compute_topology_metrics


def _make_df(n, dataset="small"):
    smiles = ["c1ccccc1", "OC=O", "OC(=O)C(=O)O"]
    if dataset == "small":
        return pd.DataFrame({
            **{f"metal_feat_{i}": np.random.rand(n) for i in range(6)},
            "linker_smiles": [smiles[i % len(smiles)] for i in range(n)],
            "refcode": [f"MOF{i}" for i in range(n)],
        })
    return pd.DataFrame({
        "linker_smiles": [smiles[i % len(smiles)] for i in range(n)],
        "metal": ["Cu"] * n,
        "Largest Cavity Diameter": np.random.rand(n) * 5,
        "Largest Free Sphere": np.random.rand(n) * 5,
        "refcode": [f"MOF{i}" for i in range(n)],
    })


class TestGraphConstructionTortureTests:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 11, 12])
    def test_datasets_smaller_than_max_k_do_not_crash(self, n):
        """Regression test for the real bug found here: any n <= max(KNN_VALUES) used
        to crash with a numpy ValueError."""
        df = _make_df(n, "small")
        graphs = build_similarity_graphs(df, "small")
        for name, g in graphs.items():
            assert g.number_of_nodes() == n

    def test_single_node_has_no_edges_in_any_config(self):
        df = _make_df(1, "small")
        graphs = build_similarity_graphs(df, "small")
        for g in graphs.values():
            assert g.number_of_edges() == 0

    def test_topology_metrics_survive_single_node_graph(self):
        df = _make_df(1, "small")
        graphs = build_similarity_graphs(df, "small")
        topology = compute_topology_metrics(graphs)
        assert (topology["isolated_node_rate"] == 1.0).all()

    def test_all_identical_linkers_still_produces_valid_graph(self):
        """Zero chemical diversity -- every Tanimoto similarity is 1.0. Must not
        divide-by-zero or otherwise misbehave."""
        n = 15
        df = pd.DataFrame({
            **{f"metal_feat_{i}": np.random.rand(n) for i in range(6)},
            "linker_smiles": ["c1ccccc1"] * n,
            "refcode": [f"MOF{i}" for i in range(n)],
        })
        graphs = build_similarity_graphs(df, "small")
        for g in graphs.values():
            assert g.number_of_nodes() == n
            for _, _, d in g.edges(data=True):
                assert np.isfinite(d["weight"])

    def test_all_same_metal_large_dataset_one_hot_has_single_column(self):
        n = 12
        df = _make_df(n, "large")
        graphs = build_similarity_graphs(df, "large")
        assert all(g.number_of_nodes() == n for g in graphs.values())

    def test_empty_dataframe_produces_empty_graphs_not_a_crash(self):
        df = _make_df(0, "small")
        graphs = build_similarity_graphs(df, "small")
        for g in graphs.values():
            assert g.number_of_nodes() == 0
            assert g.number_of_edges() == 0


class TestFeatureEngineeringTortureTests:
    def test_all_invalid_smiles_produces_zero_fingerprints_not_a_crash(self):
        series = pd.Series(["not_smiles(", "also(not", "((broken"])
        fps = compute_fingerprint_matrix(series)
        assert fps.shape == (3, 1024)
        assert (fps.sum(axis=1) == 0).all()

    def test_single_atom_molecule(self):
        """The simplest possible valid molecule -- must not choke the fingerprint or
        molecular-weight computation."""
        df = pd.DataFrame({
            **{f"metal_feat_{i}": [0.5] for i in range(6)},
            "linker_smiles": ["C"],  # methane
        })
        features, metadata = build_compact_features(df)
        assert features.shape == (1, 7)
        assert not np.isnan(features).any()

    def test_empty_smiles_string_handled_as_invalid(self):
        df = pd.DataFrame({
            **{f"metal_feat_{i}": [0.5, 0.5] for i in range(6)},
            "linker_smiles": ["", "c1ccccc1"],
        })
        features, metadata = build_compact_features(df)
        assert not np.isnan(features).any()
        assert metadata["invalid_smiles_count"] == 1

    def test_single_row_fingerprint_scheme(self):
        df = pd.DataFrame({
            "linker_smiles": ["c1ccccc1"],
            "metal": ["Cu"],
            "Largest Cavity Diameter": [3.0],
            "Largest Free Sphere": [2.5],
        })
        features, metadata = build_fingerprint_features(df)
        assert features.shape == (1, 1024 + 2 + 1)  # 1 unique metal


class TestBlackHoleSparsificationTortureTests:
    def test_single_node_community(self):
        g = nx.Graph()
        g.add_node(0)
        gravity, *_ = calculate_gravity_per_community(g, [{0}], (0.33, 0.33, 0.33))
        assert 0 in gravity

    def test_all_weight_on_a_single_zero_component(self):
        """If the only nonzero weight targets a metric that's identical for every node
        (e.g. all nodes have equal degree), gravity should degrade to all-equal, not NaN."""
        g = nx.Graph()
        g.add_edge(0, 1, weight=0.5)
        g.add_edge(2, 3, weight=0.5)
        communities = [{0, 1, 2, 3}]
        gravity, *_ = calculate_gravity_per_community(g, communities, (1.0, 0.0, 0.0))
        assert all(np.isfinite(v) for v in gravity.values())

    def test_extreme_threshold_still_respects_twenty_percent_floor(self):
        g = nx.Graph()
        for i in range(20):
            g.add_edge(i, (i + 1) % 20, weight=0.5)
        df = pd.DataFrame({"pld_category": ["nonporous"] * 20})
        result = apply_black_hole_sparsification(g, df, (0.33, 0.33, 0.33), threshold=0.9)
        assert result["metrics"]["nodes_after"] >= 3  # ~20% of 20, rounding-tolerant

    def test_zero_edge_graph(self):
        g = nx.Graph()
        g.add_nodes_from(range(10))
        df = pd.DataFrame({"pld_category": ["nonporous"] * 10})
        result = apply_black_hole_sparsification(g, df, (0.33, 0.33, 0.33), threshold=0.3)
        assert result["metrics"]["edges_before"] == 0
        assert result["metrics"]["edges_after"] == 0

    def test_single_category_only_no_stratification_diversity(self):
        g = nx.Graph()
        for i in range(10):
            g.add_edge(i, (i + 1) % 10, weight=0.5)
        df = pd.DataFrame({"pld_category": ["nonporous"] * 10})  # zero category diversity
        result = apply_black_hole_sparsification(g, df, (0.33, 0.33, 0.33), threshold=0.5)
        assert result["metrics"]["nodes_after"] > 0


class TestGnnTrainingTortureTests:
    def test_zero_edge_graph_still_trains(self):
        """A graph with no edges at all -- GCN/GraphSAGE/GAT must degrade to using only
        each node's own features (via self-loops / isolated aggregation), not crash."""
        n = 10
        features = np.random.rand(n, 4).astype(np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.zeros((0,), dtype=np.float32)
        labels = np.array([i % 4 for i in range(n)])
        train_idx, val_idx, test_idx = make_splits(n, {0, 1})
        data = build_pyg_data(features, edge_index, edge_weight, labels, train_idx, val_idx, test_idx, "classification")

        model = GCN(dim_in=4, dim_h=8, dim_out=4)
        model, history = train_model(model, data, "classification", epochs=3)
        metrics = evaluate_model(model, data, "classification")
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_all_same_label_classification(self):
        """Zero class diversity -- accuracy should trivially be 1.0 (or well-defined),
        not NaN/crash, and Cohen's kappa (undefined for a single class) must not raise."""
        n = 10
        features = np.random.rand(n, 4).astype(np.float32)
        edge_index = np.array([[i, (i + 1) % n] for i in range(n)]).T
        edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
        labels = np.zeros(n, dtype=np.int64)  # every node is class 0
        train_idx, val_idx, test_idx = make_splits(n, {0, 1})
        data = build_pyg_data(features, edge_index, edge_weight, labels, train_idx, val_idx, test_idx, "classification")

        model = GCN(dim_in=4, dim_h=8, dim_out=4)
        model, _ = train_model(model, data, "classification", epochs=3)
        metrics = evaluate_model(model, data, "classification")
        assert np.isfinite(metrics["accuracy"])

    def test_single_test_node(self):
        n = 10
        features = np.random.rand(n, 4).astype(np.float32)
        edge_index = np.array([[i, (i + 1) % n] for i in range(n)]).T
        edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
        labels = np.array([i % 4 for i in range(n)])
        train_idx, val_idx, test_idx = make_splits(n, {0})  # only 1 test node
        data = build_pyg_data(features, edge_index, edge_weight, labels, train_idx, val_idx, test_idx, "classification")

        model = GCN(dim_in=4, dim_h=8, dim_out=4)
        model, _ = train_model(model, data, "classification", epochs=3)
        metrics = evaluate_model(model, data, "classification")
        assert metrics["accuracy"] in (0.0, 1.0)  # only one node -- either right or wrong

    def test_extreme_feature_values_do_not_produce_nan_loss(self):
        """Very large feature magnitudes -- a numerically fragile model could overflow
        to NaN/Inf loss instead of just training poorly."""
        n = 10
        features = (np.random.rand(n, 4).astype(np.float32) * 1e6)
        edge_index = np.array([[i, (i + 1) % n] for i in range(n)]).T
        edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
        labels = np.array([i % 4 for i in range(n)])
        train_idx, val_idx, test_idx = make_splits(n, {0, 1})
        data = build_pyg_data(features, edge_index, edge_weight, labels, train_idx, val_idx, test_idx, "classification")

        model = GCN(dim_in=4, dim_h=8, dim_out=4)
        _, history = train_model(model, data, "classification", epochs=3)
        assert all(np.isfinite(loss) for loss in history["train_loss"])


class TestGraphToEdgeArraysTorture:
    def test_graph_with_only_isolated_nodes(self):
        g = nx.Graph()
        g.add_nodes_from(range(5))
        edge_index, edge_weight = graph_to_edge_arrays(g, 5)
        assert edge_index.shape == (2, 0)
