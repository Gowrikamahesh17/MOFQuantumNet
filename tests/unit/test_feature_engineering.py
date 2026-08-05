"""Unit tests for src/feature_engineering.py — compact + fingerprint node feature schemes."""

import json

import numpy as np
import pandas as pd
import pytest

import feature_engineering as fe
from feature_engineering import (
    build_compact_features,
    build_features,
    build_fingerprint_features,
    compute_fingerprint_matrix,
    save_features,
)

INVALID_SMILES = "not_a_smiles("


class TestComputeFingerprintMatrix:
    def test_shape_matches_bit_count(self, small_df):
        fps = compute_fingerprint_matrix(small_df["linker_smiles"])
        assert fps.shape == (len(small_df), fe.MORGAN_N_BITS)

    def test_values_are_zero_or_one(self, small_df):
        fps = compute_fingerprint_matrix(small_df["linker_smiles"])
        assert set(np.unique(fps)).issubset({0.0, 1.0})

    def test_identical_smiles_give_identical_fingerprints(self):
        series = pd.Series(["c1ccccc1", "c1ccccc1"])
        fps = compute_fingerprint_matrix(series)
        np.testing.assert_array_equal(fps[0], fps[1])

    def test_different_molecules_give_different_fingerprints(self):
        series = pd.Series(["c1ccccc1", "OC(=O)C(=O)O"])
        fps = compute_fingerprint_matrix(series)
        assert not np.array_equal(fps[0], fps[1])

    def test_invalid_smiles_is_deterministic_zero_vector(self):
        """Deviates from BlackHole's data_utils.py (unseeded np.random.randn) —
        must be the same all-zero vector every call, not random noise."""
        series = pd.Series([INVALID_SMILES, INVALID_SMILES])
        fps = compute_fingerprint_matrix(series)
        assert fps[0].sum() == 0
        np.testing.assert_array_equal(fps[0], fps[1])


class TestBuildCompactFeatures:
    def test_output_shape(self, small_df):
        features, _ = build_compact_features(small_df)
        assert features.shape == (len(small_df), 7)

    def test_values_are_min_max_scaled(self, small_df):
        features, _ = build_compact_features(small_df)
        assert features.min() >= 0.0 - 1e-6
        assert features.max() <= 1.0 + 1e-6

    def test_no_nans_even_with_invalid_smiles(self, small_df):
        df = small_df.copy()
        df.loc[0, "linker_smiles"] = INVALID_SMILES
        features, metadata = build_compact_features(df)
        assert not np.isnan(features).any()
        assert metadata["invalid_smiles_count"] == 1

    def test_metadata_reports_correct_scheme_and_dims(self, small_df):
        _, metadata = build_compact_features(small_df)
        assert metadata["scheme"] == "compact"
        assert metadata["dimensions"] == 7

    def test_missing_required_column_raises(self, small_df):
        df = small_df.drop(columns=["metal_feat_0"])
        with pytest.raises(ValueError, match="Compact scheme requires"):
            build_compact_features(df)


class TestBuildFingerprintFeatures:
    def test_output_dimensions(self, large_df):
        features, metadata = build_fingerprint_features(large_df)
        n_metals = large_df["metal"].nunique()
        assert features.shape == (len(large_df), fe.MORGAN_N_BITS + 2 + n_metals)
        assert metadata["dimensions"] == features.shape[1]

    def test_pld_value_excluded_from_features(self, large_df):
        """The central Phase 2 finding: raw PLD must never appear in the feature
        matrix (target leakage) — only the metadata should document the exclusion."""
        _, metadata = build_fingerprint_features(large_df)
        assert "Pore Limiting Diameter" in metadata["excluded_due_to_leakage"]
        assert "Pore Limiting Diameter" not in metadata.get("pore_geometry_columns", [])

    def test_metal_one_hot_is_one_hot(self, large_df):
        features, metadata = build_fingerprint_features(large_df)
        n_metals = metadata["n_unique_metals"]
        one_hot_block = features[:, -n_metals:]
        # Every row's one-hot slice must sum to exactly 1 (one metal, no more no less).
        row_sums = one_hot_block.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(large_df)))

    def test_metal_one_hot_generalizes_beyond_four_metals(self, large_df):
        """Must not silently default unknown metals to a hardcoded slot (the bug found
        in BlackHole's data_utils.py, which only handles Cu/Zn/Fe/Co)."""
        df = large_df.copy()
        df.loc[0, "metal"] = "Uranium-Does-Not-Exist-In-Any-Hardcoded-Map"
        features, metadata = build_fingerprint_features(df)
        assert metadata["n_unique_metals"] == df["metal"].nunique()
        assert "Uranium-Does-Not-Exist-In-Any-Hardcoded-Map" in metadata["metal_to_index"]

    def test_missing_required_column_raises(self, large_df):
        df = large_df.drop(columns=["metal"])
        with pytest.raises(ValueError, match="Fingerprint scheme requires"):
            build_fingerprint_features(df)

    def test_invalid_smiles_counted(self, large_df):
        df = large_df.copy()
        df.loc[0, "linker_smiles"] = INVALID_SMILES
        _, metadata = build_fingerprint_features(df)
        assert metadata["invalid_smiles_count"] == 1


class TestBuildFeaturesDispatcher:
    def test_compact_scheme_dispatches_correctly(self, small_df):
        features, metadata = build_features(small_df, "compact")
        assert metadata["scheme"] == "compact"

    def test_fingerprint_scheme_dispatches_correctly(self, large_df):
        features, metadata = build_features(large_df, "fingerprint")
        assert metadata["scheme"] == "fingerprint"

    def test_unknown_scheme_raises(self, small_df):
        with pytest.raises(ValueError, match="Unknown feature_scheme"):
            build_features(small_df, "quantum")

    def test_row_count_matches_input(self, small_df):
        features, _ = build_features(small_df, "compact")
        assert features.shape[0] == len(small_df)

    def test_catches_nan_introduced_downstream(self, small_df, monkeypatch):
        """A safety-net check: even if a bug crept into build_compact_features and
        produced NaNs, build_features must catch it rather than silently propagating."""
        def _broken(df):
            arr = np.zeros((len(df), 7), dtype=np.float32)
            arr[0, 0] = np.nan
            return arr, {"scheme": "compact"}

        monkeypatch.setattr(fe, "build_compact_features", _broken)
        with pytest.raises(ValueError, match="NaNs"):
            build_features(small_df, "compact")

    def test_catches_wrong_row_count_downstream(self, small_df, monkeypatch):
        def _broken(df):
            return np.zeros((len(df) - 1, 7), dtype=np.float32), {"scheme": "compact"}

        monkeypatch.setattr(fe, "build_compact_features", _broken)
        with pytest.raises(ValueError, match="rows"):
            build_features(small_df, "compact")


class TestSaveFeatures:
    def test_round_trip(self, small_df, tmp_path):
        features, metadata = build_compact_features(small_df)
        save_features(features, metadata, small_df["refcode"], output_dir=str(tmp_path))

        loaded = np.load(tmp_path / "features_compact.npy")
        np.testing.assert_array_equal(loaded, features)

        index_df = pd.read_csv(tmp_path / "features_compact_index.csv")
        assert list(index_df["refcode"]) == list(small_df["refcode"])

        with open(tmp_path / "features_compact_metadata.json") as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata["scheme"] == "compact"
