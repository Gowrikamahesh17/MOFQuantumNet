"""Property-based tests (Hypothesis) — invariants that must hold across the whole input
space, not just the specific boundary values covered by the unit tests."""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from black_hole_sparsification import apply_black_hole_sparsification  # noqa: E402
from config import PipelineConfig  # noqa: E402
from data_ingestion import derive_pld_category  # noqa: E402
from gnn_training import compute_class_weights, make_splits  # noqa: E402

CATEGORY_ORDER = ["nonporous", "small pore", "medium pore", "large pore"]


class TestDerivePldCategoryProperties:
    @given(st.floats(allow_nan=True, allow_infinity=True))
    def test_always_returns_a_valid_category(self, value):
        assert derive_pld_category(value) in set(CATEGORY_ORDER)

    @given(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    )
    def test_monotonic_in_pld_value(self, a, b):
        """Larger PLD can never map to a *smaller* category than a smaller PLD."""
        assume(a <= b)
        assert CATEGORY_ORDER.index(derive_pld_category(a)) <= CATEGORY_ORDER.index(derive_pld_category(b))

    @given(st.text(min_size=1, max_size=10))
    def test_arbitrary_garbage_strings_never_crash(self, garbage):
        assert derive_pld_category(garbage) in set(CATEGORY_ORDER)


class TestGravityWeightsNormalizedProperties:
    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    )
    def test_always_sums_to_one(self, a, b, c):
        config = PipelineConfig(gravity_degree_weight=a, gravity_betweenness_weight=b, gravity_edge_weight_sum_weight=c)
        assert sum(config.gravity_weights_normalized) == pytest.approx(1.0, abs=1e-6)

    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    )
    def test_all_normalized_components_non_negative(self, a, b, c):
        config = PipelineConfig(gravity_degree_weight=a, gravity_betweenness_weight=b, gravity_edge_weight_sum_weight=c)
        assert all(w >= 0.0 for w in config.gravity_weights_normalized)


class TestSmallDatasetTaskCascadeProperty:
    @given(st.sampled_from(["classification", "regression"]))
    def test_small_dataset_is_always_classification_regardless_of_requested_task(self, task):
        assert PipelineConfig(dataset="small", task=task).task == "classification"


class TestComputeClassWeightsProperties:
    @given(st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=100))
    def test_weights_always_sum_to_one(self, labels):
        weights = compute_class_weights(np.array(labels))
        assert weights.sum().item() == pytest.approx(1.0, abs=1e-4)

    @given(st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=100))
    def test_weights_always_finite_and_non_negative(self, labels):
        weights = compute_class_weights(np.array(labels))
        assert bool((weights >= 0).all())
        assert bool(weights.isfinite().all())


class TestMakeSplitsProperties:
    @given(
        st.integers(min_value=5, max_value=200),
        st.data(),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_train_val_test_always_partition_the_full_range(self, n, data):
        test_size = data.draw(st.integers(min_value=0, max_value=n - 2))
        test_indices = set(data.draw(st.lists(st.integers(min_value=0, max_value=n - 1),
                                               min_size=test_size, max_size=test_size, unique=True)))
        train, val, test = make_splits(n, test_indices)
        assert set(train) | set(val) | set(test) == set(range(n))
        assert set(train) & set(val) == set()
        assert set(train) & set(test) == set()
        assert set(val) & set(test) == set()


class TestBlackHoleSparsificationProperties:
    """A fixed small graph, but weights/threshold drawn from the full valid space —
    the pruning result must obey its own invariants regardless of configuration."""

    @staticmethod
    def _make_graph():
        g = nx.Graph()
        edges = [
            (0, 1, 0.9), (1, 2, 0.8), (2, 3, 0.7), (3, 4, 0.6), (4, 0, 0.5),
            (5, 6, 0.9), (6, 7, 0.8), (7, 8, 0.7), (8, 9, 0.6), (9, 5, 0.5),
            (0, 5, 0.3),
        ]
        for u, v, w in edges:
            g.add_edge(u, v, weight=w)
        return g

    @given(
        st.floats(min_value=0.01, max_value=1.0),
        st.floats(min_value=0.01, max_value=1.0),
        st.floats(min_value=0.01, max_value=1.0),
        st.floats(min_value=0.0, max_value=0.9),
    )
    @settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_never_increases_node_or_edge_count(self, w1, w2, w3, threshold):
        graph = self._make_graph()
        df = pd.DataFrame({"pld_category": (["nonporous", "small pore", "medium pore", "large pore"] * 3)[:10]})
        result = apply_black_hole_sparsification(graph, df, (w1, w2, w3), threshold)
        m = result["metrics"]
        assert m["nodes_after"] <= m["nodes_before"]
        assert m["edges_after"] <= m["edges_before"]

    @given(
        st.floats(min_value=0.01, max_value=1.0),
        st.floats(min_value=0.01, max_value=1.0),
        st.floats(min_value=0.01, max_value=1.0),
        st.floats(min_value=0.0, max_value=0.9),
    )
    @settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_original_graph_never_mutated(self, w1, w2, w3, threshold):
        graph = self._make_graph()
        n_before, e_before = graph.number_of_nodes(), graph.number_of_edges()
        df = pd.DataFrame({"pld_category": (["nonporous", "small pore", "medium pore", "large pore"] * 3)[:10]})
        apply_black_hole_sparsification(graph, df, (w1, w2, w3), threshold)
        assert graph.number_of_nodes() == n_before
        assert graph.number_of_edges() == e_before
