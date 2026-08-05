"""Unit tests for src/config.py — PipelineConfig cascade, validation, normalization."""

import pytest

from config import DATASET_CHOICES, FEATURE_SCHEME_BY_DATASET, TASK_CHOICES, PipelineConfig


class TestDatasetValidation:
    def test_rejects_unknown_dataset(self):
        with pytest.raises(ValueError, match="dataset must be one of"):
            PipelineConfig(dataset="medium")

    @pytest.mark.parametrize("dataset", DATASET_CHOICES)
    def test_accepts_known_datasets(self, dataset):
        PipelineConfig(dataset=dataset)  # must not raise


class TestTaskValidation:
    def test_rejects_unknown_task(self):
        with pytest.raises(ValueError, match="task must be one of"):
            PipelineConfig(task="clustering")

    @pytest.mark.parametrize("task", TASK_CHOICES)
    def test_accepts_known_tasks(self, task):
        PipelineConfig(dataset="large", task=task)  # must not raise


class TestSmallDatasetForcesClassification:
    """Small dataset has no continuous PLD — regression must be impossible on it."""

    def test_small_dataset_with_regression_is_coerced_to_classification(self):
        config = PipelineConfig(dataset="small", task="regression")
        assert config.task == "classification"

    def test_small_dataset_with_classification_stays_classification(self):
        config = PipelineConfig(dataset="small", task="classification")
        assert config.task == "classification"

    def test_large_dataset_with_regression_stays_regression(self):
        config = PipelineConfig(dataset="large", task="regression")
        assert config.task == "regression"


class TestFeatureSchemeCascade:
    """feature_scheme is derived, not independently settable."""

    def test_small_dataset_gets_compact_scheme(self):
        assert PipelineConfig(dataset="small").feature_scheme == "compact"

    def test_large_dataset_gets_fingerprint_scheme(self):
        assert PipelineConfig(dataset="large").feature_scheme == "fingerprint"

    def test_feature_scheme_has_no_setter(self):
        config = PipelineConfig(dataset="small")
        with pytest.raises(AttributeError):
            config.feature_scheme = "fingerprint"

    def test_matches_module_level_mapping(self):
        for dataset in DATASET_CHOICES:
            assert PipelineConfig(dataset=dataset).feature_scheme == FEATURE_SCHEME_BY_DATASET[dataset]


class TestPruningThresholdValidation:
    @pytest.mark.parametrize("threshold", [-0.1, 0.91, 1.0, -1.0])
    def test_rejects_out_of_range(self, threshold):
        with pytest.raises(ValueError, match="pruning_threshold"):
            PipelineConfig(pruning_threshold=threshold)

    @pytest.mark.parametrize("threshold", [0.0, 0.3, 0.5, 0.9])
    def test_accepts_boundary_and_typical_values(self, threshold):
        PipelineConfig(pruning_threshold=threshold)  # must not raise


class TestGravityWeightValidation:
    @pytest.mark.parametrize("field", [
        "gravity_degree_weight", "gravity_betweenness_weight", "gravity_edge_weight_sum_weight",
    ])
    @pytest.mark.parametrize("bad_value", [-0.01, 1.01, -5.0, 5.0])
    def test_rejects_out_of_range(self, field, bad_value):
        with pytest.raises(ValueError):
            PipelineConfig(**{field: bad_value})

    @pytest.mark.parametrize("value", [0.0, 1.0, 0.33])
    def test_accepts_boundary_values(self, value):
        PipelineConfig(gravity_degree_weight=value)  # must not raise


class TestGravityWeightsNormalized:
    def test_default_is_equal_thirds(self):
        norm = PipelineConfig().gravity_weights_normalized
        assert norm == pytest.approx((1 / 3, 1 / 3, 1 / 3))

    def test_always_sums_to_one(self):
        config = PipelineConfig(
            gravity_degree_weight=0.5, gravity_betweenness_weight=0.15, gravity_edge_weight_sum_weight=0.33,
        )
        assert sum(config.gravity_weights_normalized) == pytest.approx(1.0)

    def test_preserves_relative_ratio(self):
        # 0.2 : 0.2 : 0.6 -- doubling everything shouldn't change the normalized ratio
        config = PipelineConfig(
            gravity_degree_weight=0.2, gravity_betweenness_weight=0.2, gravity_edge_weight_sum_weight=0.6,
        )
        norm = config.gravity_weights_normalized
        assert norm == pytest.approx((0.2, 0.2, 0.6))

    def test_all_zero_falls_back_to_equal_thirds_not_divide_by_zero(self):
        config = PipelineConfig(
            gravity_degree_weight=0.0, gravity_betweenness_weight=0.0, gravity_edge_weight_sum_weight=0.0,
        )
        assert config.gravity_weights_normalized == pytest.approx((1 / 3, 1 / 3, 1 / 3))

    def test_raw_values_unaffected_by_normalization(self):
        """Normalization must be read-only — must not mutate the raw slider values,
        since the UI displays both the raw and normalized values side by side."""
        config = PipelineConfig(gravity_degree_weight=0.5, gravity_betweenness_weight=0.15, gravity_edge_weight_sum_weight=0.33)
        _ = config.gravity_weights_normalized
        assert config.gravity_degree_weight == 0.5
        assert config.gravity_betweenness_weight == 0.15
        assert config.gravity_edge_weight_sum_weight == 0.33
