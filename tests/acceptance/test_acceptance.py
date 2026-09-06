"""Acceptance tests — verify the documented, functional requirements against the real
datasets, not synthetic fixtures. Each test class maps to a specific claim made in
README.md / planning/DEVELOPMENT_TODO.md / planning/CASE_STUDY_REPORT.md.

Marked `acceptance` (and the heaviest ones additionally `slow`) so the fast unit/
property/torture suite can run on its own: `pytest -m "not slow"`.

Large-dataset tests skip gracefully if data/raw/MOFCSD.csv isn't present locally (it's
tracked in this repo, but the skip guard is kept as a defensive fallback for a partial
checkout rather than assumed).
"""

import os

import numpy as np
import pytest

from black_hole_sparsification import apply_black_hole_sparsification
from config import PipelineConfig
from data_ingestion import load_dataset
from feature_engineering import build_features
from graph_construction import KNN_VALUES, build_similarity_graphs, compute_topology_metrics, select_best_graph

pytestmark = pytest.mark.acceptance

LARGE_DATA_PATH = "data/raw/MOFCSD.csv"
skip_if_no_large_data = pytest.mark.skipif(
    not os.path.exists(LARGE_DATA_PATH), reason="data/raw/MOFCSD.csv not present in this checkout"
)


class TestPhase1DataRequirements:
    """README claim: 'reproduced the original paper's PLD class histogram almost exactly'."""

    def test_small_dataset_loads_with_documented_row_count(self):
        df = load_dataset("small")
        assert len(df) == 2004

    def test_small_dataset_category_histogram_matches_paper_within_tolerance(self):
        """Paper-reported: nonporous 1062, small 422, medium 271, large 244.
        Documented finding: our load reproduces 1062/425/271/246 -- within a few MOFs."""
        df = load_dataset("small")
        counts = df["pld_category"].value_counts().to_dict()
        paper = {"nonporous": 1062, "small pore": 422, "medium pore": 271, "large pore": 244}
        for category, paper_count in paper.items():
            assert abs(counts.get(category, 0) - paper_count) <= 5, f"{category} drifted too far from the paper"

    def test_canonical_schema_columns_present(self):
        df = load_dataset("small")
        assert {"refcode", "linker_smiles", "metal", "pld_category", "pld_value"}.issubset(df.columns)

    @skip_if_no_large_data
    def test_large_dataset_loads_with_documented_row_count(self):
        df = load_dataset("large")
        assert len(df) == 14296

    @skip_if_no_large_data
    def test_large_dataset_has_no_missing_values_in_key_columns(self):
        """Documented finding: 'zero NaNs' in MOFCSD.csv."""
        df = load_dataset("large")
        for col in ["linker_smiles", "metal", "pld_value"]:
            assert df[col].notna().all()

    @skip_if_no_large_data
    def test_large_dataset_has_documented_metal_diversity(self):
        """Documented finding: 53 unique metals, motivating the generalized one-hot
        (vs. the reference code's hardcoded 4-metal map)."""
        df = load_dataset("large")
        assert df["metal"].nunique() >= 50


class TestPhase2FeatureRequirements:
    def test_compact_scheme_produces_documented_shape(self):
        df = load_dataset("small")
        features, metadata = build_features(df, "compact")
        assert features.shape == (len(df), 7)
        assert not np.isnan(features).any()

    @skip_if_no_large_data
    def test_fingerprint_scheme_never_leaks_the_target(self):
        """The case study's central Phase 2 finding: PLD must never appear as an input
        feature (it's what the label is derived from)."""
        df = load_dataset("large")
        _, metadata = build_features(df, "fingerprint")
        assert "Pore Limiting Diameter" in metadata["excluded_due_to_leakage"]

    @skip_if_no_large_data
    def test_fingerprint_scheme_dimension_matches_documented_value(self):
        """README/report both state 1,079 = 1024 fingerprint bits + 2 pore-geometry + 53 metals."""
        df = load_dataset("large")
        features, metadata = build_features(df, "fingerprint")
        assert metadata["dimensions"] == 1024 + 2 + df["metal"].nunique()


@pytest.fixture(scope="module")
def small_topology():
    df = load_dataset("small")
    graphs = build_similarity_graphs(df, "small")
    return compute_topology_metrics(graphs)


@pytest.fixture(scope="module")
def small_best_graph():
    df = load_dataset("small")
    graphs = build_similarity_graphs(df, "small")
    topology = compute_topology_metrics(graphs)
    best = select_best_graph(topology)
    return df, graphs[best]


class TestPhase3GraphConstructionRequirements:
    """Report Section 4.1: k-NN guarantees 0% isolated nodes; threshold leaves ~20% isolated."""

    def test_all_four_configs_present(self, small_topology):
        expected = {f"threshold_{0.9}"} | {f"knn_{k}" for k in KNN_VALUES}
        assert set(small_topology.index) == expected

    def test_knn_configs_have_zero_isolated_nodes(self, small_topology):
        for k in KNN_VALUES:
            assert small_topology.loc[f"knn_{k}", "isolated_node_rate"] == 0.0

    def test_threshold_config_has_isolated_nodes_near_documented_value(self, small_topology):
        """Report states 20.5% isolated at phi=0.9 on the small dataset."""
        isolated = small_topology.loc["threshold_0.9", "isolated_node_rate"]
        assert abs(isolated - 0.205) < 0.02

    def test_select_best_graph_picks_a_zero_isolated_config(self, small_topology):
        best = select_best_graph(small_topology)
        assert small_topology.loc[best, "isolated_node_rate"] == 0.0


class TestPhase4BlackHoleRequirements:
    """Report Section 4.2: node retention ~66% at tau=0.3, ~48% at tau=0.5."""

    def test_node_retention_near_documented_value_at_tau_0_3(self, small_best_graph):
        df, graph = small_best_graph
        result = apply_black_hole_sparsification(graph, df, (0.33, 0.33, 0.33), 0.3)
        assert abs(result["metrics"]["node_retention_pct"] - 66.3) < 5.0

    def test_node_retention_near_documented_value_at_tau_0_5(self, small_best_graph):
        df, graph = small_best_graph
        result = apply_black_hole_sparsification(graph, df, (0.33, 0.33, 0.33), 0.5)
        assert abs(result["metrics"]["node_retention_pct"] - 47.9) < 5.0

    def test_fixed_test_nodes_survive_pruning_at_both_reference_thresholds(self, small_best_graph):
        df, graph = small_best_graph
        r30 = apply_black_hole_sparsification(graph, df, (0.33, 0.33, 0.33), 0.3)
        r50 = apply_black_hole_sparsification(graph, df, (0.33, 0.33, 0.33), 0.5)
        # Same graph + weights -> deterministic, identical fixed test set across thresholds
        assert r30["fixed_test_nodes"] == r50["fixed_test_nodes"]


class TestConfigCascadeRequirement:
    """README's central architectural claim: dataset is the one root switch."""

    def test_small_dataset_end_to_end_cascade(self):
        config = PipelineConfig(dataset="small", task="regression")  # regression requested but impossible
        assert config.task == "classification"
        assert config.feature_scheme == "compact"

    def test_large_dataset_end_to_end_cascade(self):
        config = PipelineConfig(dataset="large", task="regression")
        assert config.task == "regression"
        assert config.feature_scheme == "fingerprint"


@pytest.mark.slow
class TestFullPipelineSmokeTest:
    """The heaviest acceptance test: exercises every phase in sequence against the real
    small dataset, with a reduced epoch budget so it's slow-ish but not the full ~10-15s
    run. Verifies the whole chain actually fits together end-to-end, not just each
    module in isolation."""

    def test_config_to_evaluation_end_to_end(self):
        from gnn_training import GCN, build_pyg_data, compute_class_weights, evaluate_model, graph_to_edge_arrays, make_splits, train_model

        config = PipelineConfig(dataset="small", task="classification")
        df = load_dataset(config.dataset)
        features, _ = build_features(df, config.feature_scheme)
        graphs = build_similarity_graphs(df, config.dataset)
        topology = compute_topology_metrics(graphs)
        best_name = select_best_graph(topology)
        best_graph = graphs[best_name]

        bh_result = apply_black_hole_sparsification(best_graph, df, config.gravity_weights_normalized, config.pruning_threshold)
        fixed_test_nodes = bh_result["fixed_test_nodes"]

        cat_to_code = {c: i for i, c in enumerate(["nonporous", "small pore", "medium pore", "large pore"])}
        labels = df["pld_category"].map(cat_to_code).values
        class_weights = compute_class_weights(labels)

        edge_index, edge_weight = graph_to_edge_arrays(best_graph, len(df))
        train_idx, val_idx, test_idx = make_splits(len(df), fixed_test_nodes)
        data = build_pyg_data(features, edge_index, edge_weight, labels, train_idx, val_idx, test_idx, "classification")

        model = GCN(dim_in=features.shape[1], dim_h=64, dim_out=4)
        model, history = train_model(model, data, "classification", epochs=10, class_weights=class_weights)
        metrics = evaluate_model(model, data, "classification")

        assert len(history["train_loss"]) > 0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert not np.isnan(metrics["accuracy"])
