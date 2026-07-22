"""Phase 2 — Feature engineering.

Two schemes, cascading from the dataset choice (src/config.py):
  - "compact"     -> small dataset (2,000 MOFs): 6 existing metal_feat columns
                     + 1 linker descriptor (molecular weight) = 7-dim, Min-Max scaled.
  - "fingerprint" -> large dataset (14,296 MOFs): 1024-bit Morgan fingerprint (radius=2)
                     + 2 pore-geometry features + one-hot metal (N = actual unique metals).

Deliberate deviation from BlackHole's data_utils.py: that code includes the raw
'Pore Limiting Diameter' value as an input feature. Since PLD is exactly what both
the classification label (pld_category) and the regression target are derived from,
including it as a feature is direct target leakage — trivializes the task instead of
learning real structure-property patterns. Excluded here regardless of task; only the
two other pore-geometry columns (Largest Cavity Diameter, Largest Free Sphere) are kept,
since those are merely correlated with PLD, not identical to it.

Also deviates from data_utils.py's invalid-SMILES fallback, which returns
`np.random.randn(n_bits)` with no fixed seed (non-reproducible run-to-run). Uses an
all-zero vector instead — deterministic, and the row is logged so it's traceable.
"""

import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.preprocessing import MinMaxScaler

from logging_setup import get_logger

RDLogger.DisableLog("rdApp.*")
logger = get_logger(__name__)

MORGAN_RADIUS = 2
MORGAN_N_BITS = 1024
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_N_BITS)


def _mol_or_none(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Invalid SMILES could not be parsed: {smiles!r}")
    return mol


def _morgan_fingerprint(smiles: str) -> np.ndarray:
    mol = _mol_or_none(smiles)
    if mol is None:
        return np.zeros(MORGAN_N_BITS, dtype=np.float32)
    return _MORGAN_GENERATOR.GetFingerprintAsNumPy(mol).astype(np.float32)


def compute_fingerprint_matrix(smiles_series: pd.Series) -> np.ndarray:
    """Public, reusable helper — also used by Phase 3 (graph construction) for
    linker similarity, independent of which node-feature scheme (Phase 2) is active."""
    return np.stack(smiles_series.apply(_morgan_fingerprint).values).astype(np.float32)


def _molecular_weight(smiles: str) -> float:
    mol = _mol_or_none(smiles)
    if mol is None:
        return np.nan
    return Descriptors.MolWt(mol)


def build_compact_features(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """7-dim compact scheme for the small dataset."""
    metal_cols = [f"metal_feat_{i}" for i in range(6)]
    missing = [c for c in metal_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Compact scheme requires columns {metal_cols}, missing: {missing}")

    linker_mw = df["linker_smiles"].apply(_molecular_weight)
    n_invalid = linker_mw.isna().sum()
    if n_invalid:
        logger.warning(f"{n_invalid} linker SMILES failed to parse for MolWt, filling with median")
        linker_mw = linker_mw.fillna(linker_mw.median())

    raw = np.column_stack([df[metal_cols].values, linker_mw.values])
    scaled = MinMaxScaler().fit_transform(raw)

    metadata = {
        "scheme": "compact",
        "dimensions": 7,
        "columns": metal_cols + ["linker_molecular_weight"],
        "scaling": "column-wise MinMaxScaler over all 7 dims",
        "invalid_smiles_count": int(n_invalid),
    }
    return scaled.astype(np.float32), metadata


def build_fingerprint_features(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Generalized fingerprint scheme for the large dataset.

    Metal one-hot is built dynamically over the metals actually present (53 in the
    current data), replacing BlackHole's hardcoded 4-slot map (Cu/Zn/Fe/Co) that
    silently mis-encoded anything else as Cu.
    """
    required = ["linker_smiles", "metal", "Largest Cavity Diameter", "Largest Free Sphere"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fingerprint scheme requires columns {required}, missing: {missing}")

    fingerprints = compute_fingerprint_matrix(df["linker_smiles"])
    n_invalid = int((fingerprints.sum(axis=1) == 0).sum())
    if n_invalid:
        logger.warning(f"{n_invalid} linker SMILES produced an all-zero fingerprint (invalid SMILES fallback)")

    pore_geometry = df[["Largest Cavity Diameter", "Largest Free Sphere"]].values.astype(np.float32)

    unique_metals = sorted(df["metal"].dropna().unique().tolist())
    metal_to_index = {m: i for i, m in enumerate(unique_metals)}
    metal_one_hot = np.zeros((len(df), len(unique_metals)), dtype=np.float32)
    for row_idx, metal in enumerate(df["metal"].values):
        metal_one_hot[row_idx, metal_to_index[metal]] = 1.0

    features = np.concatenate([fingerprints, pore_geometry, metal_one_hot], axis=1)

    metadata = {
        "scheme": "fingerprint",
        "dimensions": features.shape[1],
        "morgan_bits": MORGAN_N_BITS,
        "morgan_radius": MORGAN_RADIUS,
        "pore_geometry_columns": ["Largest Cavity Diameter", "Largest Free Sphere"],
        "excluded_due_to_leakage": ["Pore Limiting Diameter"],
        "n_unique_metals": len(unique_metals),
        "metal_to_index": metal_to_index,
        "invalid_smiles_count": n_invalid,
    }
    return features.astype(np.float32), metadata


def build_features(df: pd.DataFrame, feature_scheme: str) -> tuple[np.ndarray, dict]:
    if feature_scheme == "compact":
        features, metadata = build_compact_features(df)
    elif feature_scheme == "fingerprint":
        features, metadata = build_fingerprint_features(df)
    else:
        raise ValueError(f"Unknown feature_scheme {feature_scheme!r}")

    if np.isnan(features).any():
        raise ValueError("Feature matrix contains NaNs after build — check upstream cleaning")
    if np.isinf(features).any():
        raise ValueError("Feature matrix contains infinite values after build")
    if features.shape[0] != len(df):
        raise ValueError(f"Feature matrix has {features.shape[0]} rows, expected {len(df)}")

    logger.info(f"Built {feature_scheme} feature matrix: shape={features.shape}")
    return features, metadata


def save_features(features: np.ndarray, metadata: dict, refcodes: pd.Series, output_dir: str = "data/processed") -> None:
    os.makedirs(output_dir, exist_ok=True)
    scheme = metadata["scheme"]
    np.save(os.path.join(output_dir, f"features_{scheme}.npy"), features)
    pd.DataFrame({"refcode": refcodes}).to_csv(os.path.join(output_dir, f"features_{scheme}_index.csv"), index=False)
    with open(os.path.join(output_dir, f"features_{scheme}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved {scheme} features to {output_dir}/ ({features.shape})")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from data_ingestion import load_dataset

    for dataset_name, scheme in (("small", "compact"), ("large", "fingerprint")):
        print("=" * 60)
        print(f"{dataset_name.upper()} dataset -> {scheme} scheme")
        print("=" * 60)
        df = load_dataset(dataset_name)
        features, metadata = build_features(df, scheme)
        print(f"Shape: {features.shape}")
        print(f"Metadata: { {k: v for k, v in metadata.items() if k != 'metal_to_index'} }")
        if scheme == "fingerprint":
            print(f"Unique metals encoded: {metadata['n_unique_metals']}")
        save_features(features, metadata, df["refcode"], output_dir="data/processed")
        print()
