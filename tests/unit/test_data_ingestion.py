"""Unit tests for src/data_ingestion.py — category derivation, loading, canonical schema."""

import math

import pandas as pd
import pytest

import data_ingestion
from data_ingestion import (
    INVALID_SMILES,
    INVALID_SMILES_REPLACEMENT,
    derive_pld_category,
    load_dataset,
    load_large_dataset,
    load_small_dataset,
)


class TestDerivePldCategory:
    @pytest.mark.parametrize("pld,expected", [
        (0.0, "nonporous"),
        (1.0, "nonporous"),
        (2.399, "nonporous"),
        (2.4, "small pore"),       # lower boundary is inclusive on the "small" side
        (3.999, "small pore"),
        (4.0, "medium pore"),      # lower boundary is inclusive on the "medium" side
        (7.999, "medium pore"),
        (8.0, "large pore"),       # lower boundary is inclusive on the "large" side
        (100.0, "large pore"),
    ])
    def test_bin_boundaries(self, pld, expected):
        assert derive_pld_category(pld) == expected

    def test_nan_is_nonporous(self):
        assert derive_pld_category(float("nan")) == "nonporous"

    def test_none_is_nonporous(self):
        assert derive_pld_category(None) == "nonporous"

    def test_non_numeric_string_is_nonporous(self):
        assert derive_pld_category("not a number") == "nonporous"

    def test_negative_pld_is_nonporous(self):
        # Physically meaningless, but must not crash — should degrade gracefully.
        assert derive_pld_category(-5.0) == "nonporous"

    def test_numeric_string_is_parsed(self):
        assert derive_pld_category("5.0") == "medium pore"

    def test_infinity_is_large_pore(self):
        assert derive_pld_category(float("inf")) == "large pore"

    def test_result_is_always_one_of_four_categories(self):
        valid = {"nonporous", "small pore", "medium pore", "large pore"}
        for value in (-10, 0, 2.4, 4.0, 8.0, 1000, float("nan"), None, "garbage"):
            assert derive_pld_category(value) in valid


class TestLoadSmallDataset:
    def test_shape_and_columns(self, tmp_path):
        csv_path = tmp_path / "small.csv"
        csv_path.write_text(
            "0.1,0.2,0.3,0.4,0.5,0.6,0,MOF0001,c1ccccc1,0\n"
            "0.2,0.3,0.4,0.5,0.6,0.7,1,MOF0002,OC=O,2\n"
        )
        df = load_small_dataset(path=str(csv_path))
        assert df.shape == (2, 10)
        assert list(df.columns) == data_ingestion.SMALL_DATASET_COLUMNS

    def test_no_header_row_consumed_as_data(self, tmp_path):
        """The real file has no header row — the first line is real data, not a header
        to be discarded. A regression here would silently drop one MOF every load."""
        csv_path = tmp_path / "small.csv"
        csv_path.write_text("0.1,0.2,0.3,0.4,0.5,0.6,0,MOF0001,c1ccccc1,0\n")
        df = load_small_dataset(path=str(csv_path))
        assert len(df) == 1
        assert df.iloc[0]["refcode"] == "MOF0001"


class TestLoadLargeDataset:
    def _write_csv(self, tmp_path, rows):
        csv_path = tmp_path / "large.csv"
        header = "refcode,linker SMILES,metal,Largest Cavity Diameter,Pore Limiting Diameter,Largest Free Sphere\n"
        csv_path.write_text(header + "\n".join(rows) + "\n")
        return str(csv_path)

    def test_invalid_smiles_replaced_with_benzene(self, tmp_path):
        path = self._write_csv(tmp_path, [
            f"MOF0001,{INVALID_SMILES},Cu,4.0,2.5,3.9",
            "MOF0002,c1ccccc1,Zn,4.0,2.5,3.9",
        ])
        df = load_large_dataset(path=path)
        assert (df["linker SMILES"] == INVALID_SMILES).sum() == 0
        assert df.loc["MOF0001", "linker SMILES"] == INVALID_SMILES_REPLACEMENT

    def test_pld_category_column_added(self, tmp_path):
        path = self._write_csv(tmp_path, ["MOF0001,c1ccccc1,Cu,4.0,2.5,3.9"])
        df = load_large_dataset(path=path)
        assert "pld_category" in df.columns
        assert df.loc["MOF0001", "pld_category"] == "small pore"  # 2.5 -> [2.4, 4.0)

    def test_refcode_is_index(self, tmp_path):
        path = self._write_csv(tmp_path, ["MOF0001,c1ccccc1,Cu,4.0,2.5,3.9"])
        df = load_large_dataset(path=path)
        assert df.index.name == "refcode"
        assert "MOF0001" in df.index


class TestLoadDatasetCanonicalSchema:
    """load_dataset() is the single switch point everything downstream depends on —
    both branches must return the same canonical columns regardless of source."""

    CANONICAL_COLUMNS = {"refcode", "linker_smiles", "metal", "pld_category", "pld_value"}

    def test_rejects_unknown_dataset_name(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("medium")

    def test_small_dataset_has_canonical_columns(self, monkeypatch, small_df):
        raw = small_df.drop(columns=["metal", "pld_value"]).copy()
        raw["pld_category"] = [0, 1, 2, 3] * (len(raw) // 4) + [0] * (len(raw) % 4)
        monkeypatch.setattr(data_ingestion, "load_small_dataset", lambda: raw)
        df = load_dataset("small")
        assert self.CANONICAL_COLUMNS.issubset(df.columns)

    def test_small_dataset_category_codes_mapped_to_labels(self, monkeypatch, small_df):
        raw = small_df.drop(columns=["metal", "pld_value"]).iloc[:4].copy()
        raw["pld_category"] = [0, 1, 2, 3]
        monkeypatch.setattr(data_ingestion, "load_small_dataset", lambda: raw)
        df = load_dataset("small")
        assert list(df["pld_category"]) == ["nonporous", "small pore", "medium pore", "large pore"]

    def test_small_dataset_metal_and_pld_value_are_none(self, monkeypatch, small_df):
        raw = small_df.drop(columns=["metal", "pld_value"]).copy()
        raw["pld_category"] = 0
        monkeypatch.setattr(data_ingestion, "load_small_dataset", lambda: raw)
        df = load_dataset("small")
        assert df["metal"].isna().all()
        assert df["pld_value"].isna().all()

    def test_large_dataset_has_canonical_columns(self, monkeypatch, large_df):
        raw = large_df.rename(columns={"linker_smiles": "linker SMILES", "pld_value": "Pore Limiting Diameter"})
        raw = raw.set_index("refcode", drop=True)
        monkeypatch.setattr(data_ingestion, "load_large_dataset", lambda: raw)
        df = load_dataset("large")
        assert self.CANONICAL_COLUMNS.issubset(df.columns)

    def test_large_dataset_refcode_matches_index(self, monkeypatch, large_df):
        raw = large_df.rename(columns={"linker_smiles": "linker SMILES", "pld_value": "Pore Limiting Diameter"})
        raw = raw.set_index("refcode", drop=True)
        monkeypatch.setattr(data_ingestion, "load_large_dataset", lambda: raw)
        df = load_dataset("large")
        assert list(df["refcode"]) == list(df.index)
