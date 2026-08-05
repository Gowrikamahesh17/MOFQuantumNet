"""Unit tests for src/baseline_models.py — non-graph Random Forest / k-NN baselines
and the master comparison table combining them with Phase 5's GNN results."""

import numpy as np
import pytest

from baseline_models import build_master_comparison_table, run_baseline_comparison
from gnn_training import CATEGORY_ORDER


@pytest.fixture
def features_for(large_df):
    return np.random.RandomState(0).rand(len(large_df), 8).astype(np.float32)


class TestRunBaselineComparisonClassification:
    def test_returns_both_models(self, large_df, features_for):
        results = run_baseline_comparison(large_df, features_for, "classification", {0, 1, 2, 3, 4})
        assert set(results.keys()) == {"Random Forest", "k-NN"}

    def test_expected_metric_keys(self, large_df, features_for):
        results = run_baseline_comparison(large_df, features_for, "classification", {0, 1, 2, 3, 4})
        expected = {"accuracy", "macro_f1", "confusion_matrix", "cohen_kappa"}
        for r in results.values():
            assert expected.issubset(r["metrics"].keys())

    def test_accuracy_in_valid_range(self, large_df, features_for):
        results = run_baseline_comparison(large_df, features_for, "classification", {0, 1, 2, 3, 4})
        for r in results.values():
            assert 0.0 <= r["metrics"]["accuracy"] <= 1.0

    def test_confusion_matrix_always_full_size(self, large_df, features_for):
        """labels=range(len(CATEGORY_ORDER)) must be passed explicitly -- otherwise a
        test set missing one class would silently shrink the confusion matrix."""
        results = run_baseline_comparison(large_df, features_for, "classification", {0, 1, 2, 3, 4})
        for r in results.values():
            cm = r["metrics"]["confusion_matrix"]
            assert len(cm) == len(CATEGORY_ORDER)
            assert all(len(row) == len(CATEGORY_ORDER) for row in cm)

    def test_evaluated_only_on_fixed_test_nodes(self, large_df, features_for):
        """Changing which nodes are 'test' must change results -- otherwise the split
        isn't actually being respected."""
        results_a = run_baseline_comparison(large_df, features_for, "classification", {0, 1, 2, 3, 4})
        results_b = run_baseline_comparison(large_df, features_for, "classification", {10, 11, 12, 13, 14})
        # Not a strict inequality assertion (could coincidentally match) -- but the
        # confusion matrices should almost certainly differ since predictions are on
        # entirely different rows.
        assert results_a["Random Forest"]["metrics"]["confusion_matrix"] != results_b["Random Forest"]["metrics"]["confusion_matrix"]


class TestRunBaselineComparisonRegression:
    def test_expected_metric_keys(self, large_df, features_for):
        results = run_baseline_comparison(large_df, features_for, "regression", {0, 1, 2, 3, 4})
        expected = {"mae", "rmse", "r2"}
        for r in results.values():
            assert expected.issubset(r["metrics"].keys())

    def test_mae_and_rmse_are_non_negative(self, large_df, features_for):
        results = run_baseline_comparison(large_df, features_for, "regression", {0, 1, 2, 3, 4})
        for r in results.values():
            assert r["metrics"]["mae"] >= 0.0
            assert r["metrics"]["rmse"] >= 0.0

    def test_r2_is_at_most_one(self, large_df, features_for):
        results = run_baseline_comparison(large_df, features_for, "regression", {0, 1, 2, 3, 4})
        for r in results.values():
            assert r["metrics"]["r2"] <= 1.0 + 1e-6


class TestBuildMasterComparisonTable:
    @pytest.fixture
    def sample_gnn_results(self):
        return {
            ("Full graph", "GCN"): {"metrics": {"accuracy": 0.6, "cohen_kappa": 0.3}},
            ("BH-30", "GAT"): {"metrics": {"accuracy": 0.55, "cohen_kappa": 0.25}},
        }

    @pytest.fixture
    def sample_baseline_results(self):
        return {
            "Random Forest": {"metrics": {"accuracy": 0.75, "cohen_kappa": 0.6}},
            "k-NN": {"metrics": {"accuracy": 0.65, "cohen_kappa": 0.4}},
        }

    def test_combines_gnn_and_baseline_rows(self, sample_gnn_results, sample_baseline_results):
        table = build_master_comparison_table(sample_gnn_results, sample_baseline_results, "classification")
        assert len(table) == 4

    def test_sorted_descending_by_accuracy_for_classification(self, sample_gnn_results, sample_baseline_results):
        table = build_master_comparison_table(sample_gnn_results, sample_baseline_results, "classification")
        assert table["accuracy"].is_monotonic_decreasing
        assert table.iloc[0]["Configuration"] == "Flat features — Random Forest"

    def test_type_column_labels_correctly(self, sample_gnn_results, sample_baseline_results):
        table = build_master_comparison_table(sample_gnn_results, sample_baseline_results, "classification")
        gnn_rows = table[table["Type"] == "GNN"]
        baseline_rows = table[table["Type"] == "Baseline"]
        assert len(gnn_rows) == 2
        assert len(baseline_rows) == 2

    def test_empty_gnn_results_still_works(self, sample_baseline_results):
        """The app's fallback path (Phase 5 not run yet in this session) passes {} for
        gnn_results -- must not crash, should show baselines only."""
        table = build_master_comparison_table({}, sample_baseline_results, "classification")
        assert len(table) == 2
        assert (table["Type"] == "Baseline").all()

    def test_sorted_by_r2_for_regression(self):
        gnn = {("Full graph", "GCN"): {"metrics": {"r2": 0.2}}}
        baseline = {"Random Forest": {"metrics": {"r2": 0.9}}}
        table = build_master_comparison_table(gnn, baseline, "regression")
        assert table.iloc[0]["Configuration"] == "Flat features — Random Forest"
