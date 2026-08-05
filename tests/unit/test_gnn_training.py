"""Unit tests for src/gnn_training.py — splits, class weights, models, and the fixed
train/eval loop (real bug: reference code used the test set for early stopping)."""

import networkx as nx
import numpy as np
import pytest
import torch

from gnn_training import (
    CATEGORY_ORDER,
    GAT,
    GCN,
    GraphSAGE,
    build_pyg_data,
    compute_class_weights,
    evaluate_model,
    graph_to_edge_arrays,
    make_splits,
    train_model,
)


class TestGraphToEdgeArrays:
    def test_empty_graph_returns_empty_arrays(self):
        g = nx.Graph()
        g.add_nodes_from(range(5))
        edge_index, edge_weight = graph_to_edge_arrays(g, 5)
        assert edge_index.shape == (2, 0)
        assert edge_weight.shape == (0,)

    def test_edges_are_bidirectional(self):
        """PyG message passing needs both directions for an undirected graph."""
        g = nx.Graph()
        g.add_edge(0, 1, weight=0.7)
        edge_index, edge_weight = graph_to_edge_arrays(g, 2)
        pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        assert (0, 1) in pairs and (1, 0) in pairs

    def test_weight_count_matches_edge_index_count(self):
        g = nx.Graph()
        g.add_edge(0, 1, weight=0.5)
        g.add_edge(1, 2, weight=0.8)
        edge_index, edge_weight = graph_to_edge_arrays(g, 3)
        assert edge_index.shape[1] == len(edge_weight)


class TestMakeSplits:
    def test_test_set_matches_input_exactly(self):
        train, val, test = make_splits(20, {2, 5, 9})
        assert set(test.tolist()) == {2, 5, 9}

    def test_train_val_test_are_disjoint(self):
        train, val, test = make_splits(20, {2, 5, 9})
        assert set(train) & set(val) == set()
        assert set(train) & set(test) == set()
        assert set(val) & set(test) == set()

    def test_train_val_test_cover_all_nodes(self):
        train, val, test = make_splits(20, {2, 5, 9})
        assert set(train) | set(val) | set(test) == set(range(20))

    def test_deterministic_given_same_seed(self):
        train_a, val_a, _ = make_splits(20, {2, 5, 9}, seed=42)
        train_b, val_b, _ = make_splits(20, {2, 5, 9}, seed=42)
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(val_a, val_b)

    def test_val_fraction_approximately_respected(self):
        train, val, test = make_splits(1000, set(), val_fraction=0.2)
        assert abs(len(val) / 1000 - 0.2) < 0.01

    def test_at_least_one_validation_node_for_tiny_input(self):
        """val_fraction * remaining could round to 0 for small graphs -- must floor to 1,
        or early stopping would have an empty validation set to evaluate against."""
        train, val, test = make_splits(5, {0, 1, 2, 3}, val_fraction=0.15)
        assert len(val) >= 1


class TestComputeClassWeights:
    def test_weights_sum_to_one(self):
        weights = compute_class_weights(np.array([0, 0, 0, 1, 2, 3]))
        assert weights.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_rare_class_gets_higher_weight(self):
        # class 0 appears 10x, class 1 appears once
        labels = np.array([0] * 10 + [1])
        weights = compute_class_weights(labels)
        assert weights[1] > weights[0]

    def test_missing_class_does_not_crash(self):
        """minlength=len(CATEGORY_ORDER) must cover classes with zero examples."""
        weights = compute_class_weights(np.array([0, 0]))
        assert len(weights) == len(CATEGORY_ORDER)
        assert torch.isfinite(weights).all()


class TestBuildPygData:
    def test_shapes_and_dtypes_classification(self):
        features = np.random.rand(5, 4).astype(np.float32)
        edge_index = np.array([[0, 1], [1, 0]])
        edge_weight = np.array([0.5, 0.5], dtype=np.float32)
        labels = np.array([0, 1, 2, 3, 0])
        data = build_pyg_data(features, edge_index, edge_weight, labels, np.array([0, 1]), np.array([2]), np.array([3, 4]), "classification")
        assert data.x.shape == (5, 4)
        assert data.y.dtype == torch.long
        assert data.train_mask.tolist() == [0, 1]
        assert data.test_mask.tolist() == [3, 4]

    def test_regression_labels_are_float(self):
        features = np.random.rand(3, 2).astype(np.float32)
        data = build_pyg_data(
            features, np.array([[0], [1]]), np.array([1.0]),
            np.array([1.5, 2.5, 3.5]), np.array([0]), np.array([1]), np.array([2]), "regression",
        )
        assert data.y.dtype == torch.float


class TestModelForwardShapes:
    @pytest.mark.parametrize("model_cls", [GCN, GraphSAGE, GAT])
    def test_classification_output_is_log_softmax_shaped(self, model_cls):
        model = model_cls(dim_in=4, dim_h=8, dim_out=len(CATEGORY_ORDER))
        x = torch.rand(6, 4)
        edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]])
        out = model(x, edge_index)
        assert out.shape == (6, len(CATEGORY_ORDER))
        # log_softmax rows should sum to ~1 after exponentiating
        assert torch.allclose(out.exp().sum(dim=1), torch.ones(6), atol=1e-4)

    @pytest.mark.parametrize("model_cls", [GCN, GraphSAGE, GAT])
    def test_regression_output_is_single_value_per_node(self, model_cls):
        model = model_cls(dim_in=4, dim_h=8, dim_out=1)
        x = torch.rand(6, 4)
        edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]])
        out = model(x, edge_index)
        assert out.shape == (6, 1)


class TestTrainAndEvaluate:
    @pytest.fixture
    def tiny_classification_data(self):
        n = 20
        features = np.random.RandomState(0).rand(n, 5).astype(np.float32)
        edge_index = np.array([[i, (i + 1) % n] for i in range(n)]).T
        edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
        labels = np.array([i % len(CATEGORY_ORDER) for i in range(n)])
        train_idx, val_idx, test_idx = make_splits(n, {0, 1, 2}, seed=0)
        return build_pyg_data(features, edge_index, edge_weight, labels, train_idx, val_idx, test_idx, "classification")

    def test_train_model_runs_and_returns_history(self, tiny_classification_data):
        model = GCN(dim_in=5, dim_h=8, dim_out=len(CATEGORY_ORDER))
        weights = compute_class_weights(tiny_classification_data.y.numpy())
        trained, history = train_model(model, tiny_classification_data, "classification", epochs=5, class_weights=weights)
        assert len(history["train_loss"]) > 0
        assert len(history["train_loss"]) == len(history["val_loss"])

    def test_early_stopping_never_exceeds_epoch_budget(self, tiny_classification_data):
        model = GCN(dim_in=5, dim_h=8, dim_out=len(CATEGORY_ORDER))
        _, history = train_model(model, tiny_classification_data, "classification", epochs=5, patience=2)
        assert len(history["train_loss"]) <= 5

    def test_evaluate_model_returns_valid_classification_metrics(self, tiny_classification_data):
        model = GCN(dim_in=5, dim_h=8, dim_out=len(CATEGORY_ORDER))
        train_model(model, tiny_classification_data, "classification", epochs=3)
        metrics = evaluate_model(model, tiny_classification_data, "classification")
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert -1.0 <= metrics["cohen_kappa"] <= 1.0
        assert len(metrics["confusion_matrix"]) == len(CATEGORY_ORDER)

    def test_validation_set_never_overlaps_test_set_during_training(self, tiny_classification_data):
        """The actual regression test for the fixed leakage bug: val_mask must never
        equal test_mask (the reference code's bug), and must be disjoint from it."""
        data = tiny_classification_data
        assert set(data.val_mask.tolist()) != set(data.test_mask.tolist())
        assert set(data.val_mask.tolist()).isdisjoint(set(data.test_mask.tolist()))
