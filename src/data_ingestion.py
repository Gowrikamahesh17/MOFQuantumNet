"""Phase 1 — Data ingestion & EDA.

Loads both candidate datasets found in the reference material:
  1. The small MOFGalaxyNet dataset (2,000 MOFs, 6-dim metal features + SMILES +
     a precomputed PLD category label, no continuous PLD value).
  2. The larger BlackHole dataset (14,296 MOFs, real continuous PLD + linker
     SMILES + metal + pore geometry), despite the BlackHole README stating
     these files are "not included" — they are present locally.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from config import DATASET_CHOICES
from logging_setup import get_logger

logger = get_logger(__name__)

SMALL_DATASET_COLUMNS = [
    "metal_feat_0", "metal_feat_1", "metal_feat_2",
    "metal_feat_3", "metal_feat_4", "metal_feat_5",
    "mof_index", "refcode", "linker_smiles", "pld_category",
]

# Reported in Jalali et al. (2023) / the Expose, for the 2,000-MOF dataset
PAPER_CATEGORY_COUNTS = {"nonporous": 1062, "small pore": 422, "medium pore": 271, "large pore": 244}

INVALID_SMILES = "F[Si](F)(F)(F)(F)F"
INVALID_SMILES_REPLACEMENT = "c1ccccc1"  # benzene fallback, per BlackHole's data_utils.py


def derive_pld_category(pld: float) -> str:
    """Bin a continuous Pore Limiting Diameter (Å) into a category.

    Ported from BlackHole-main/data_utils.py::derive_pld_category.
    """
    try:
        pld = float(pld)
        if pd.isna(pld):
            return "nonporous"
        if pld < 2.4:
            return "nonporous"
        elif pld < 4.0:
            return "small pore"
        elif pld < 8.0:
            return "medium pore"
        else:
            return "large pore"
    except (ValueError, TypeError):
        logger.warning(f"Invalid PLD value: {pld}, defaulting to 'nonporous'")
        return "nonporous"


def load_small_dataset(path: str = "data/raw/SMILES_METAL_2000_NoPLD.csv") -> pd.DataFrame:
    """Load the 2,000-MOF MOFGalaxyNet dataset (no header row on disk)."""
    df = pd.read_csv(path, header=None, names=SMALL_DATASET_COLUMNS)
    logger.info(f"Loaded small dataset: {df.shape} from {path}")
    return df


def load_large_dataset(path: str = "reference_content/BlackHole-main/MOFCSD.csv") -> pd.DataFrame:
    """Load the 14,296-MOF BlackHole dataset with continuous PLD."""
    df = pd.read_csv(path, index_col=0)
    logger.info(f"Loaded large dataset: {df.shape} from {path}")

    invalid_count = (df["linker SMILES"] == INVALID_SMILES).sum()
    if invalid_count:
        logger.warning(f"Found {invalid_count} invalid SMILES ('{INVALID_SMILES}'), replacing with benzene")
        df["linker SMILES"] = df["linker SMILES"].replace(INVALID_SMILES, INVALID_SMILES_REPLACEMENT)

    df["pld_category"] = df["Pore Limiting Diameter"].apply(derive_pld_category)
    return df


def load_dataset(name: str) -> pd.DataFrame:
    """Load either dataset and return it in a canonical schema.

    Canonical columns: refcode, linker_smiles, metal (None for 'small'),
    pld_category (str), pld_value (None for 'small' — no continuous PLD).
    Source-specific extra columns are preserved alongside.

    This is the single switch point: everything downstream (feature
    engineering, graph construction, Streamlit app) should call this instead
    of the source-specific loaders directly, so swapping datasets is a
    one-line change in config.py, not a rewrite.
    """
    if name not in DATASET_CHOICES:
        raise ValueError(f"Unknown dataset {name!r}, expected one of {DATASET_CHOICES}")

    if name == "small":
        df = load_small_dataset()
        code_to_label = {0: "nonporous", 1: "small pore", 2: "medium pore", 3: "large pore"}
        df["pld_category"] = df["pld_category"].map(code_to_label)
        df["metal"] = None
        df["pld_value"] = None
        return df

    df = load_large_dataset()
    df = df.rename(columns={"linker SMILES": "linker_smiles", "Pore Limiting Diameter": "pld_value"})
    df["refcode"] = df.index
    return df


def summarize_categories(series: pd.Series, label: str) -> dict:
    counts = series.value_counts().to_dict()
    logger.info(f"{label} category counts: {counts}")
    return counts


def plot_eda(small: pd.DataFrame, large: pd.DataFrame, output_dir: str = "outputs") -> None:
    os.makedirs(output_dir, exist_ok=True)
    category_order = ["nonporous", "small pore", "medium pore", "large pore"]

    # Class balance: small dataset (raw 0-3 codes) vs large dataset (derived categories)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    small_code_to_label = {0: "nonporous", 1: "small pore", 2: "medium pore", 3: "large pore"}
    small_counts = small["pld_category"].map(small_code_to_label).value_counts().reindex(category_order)
    axes[0].bar(category_order, small_counts.values, color="#4C72B0")
    axes[0].set_title("Small dataset (2,000 MOF)\nPLD category distribution")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)

    large_counts = large["pld_category"].value_counts().reindex(category_order)
    axes[1].bar(category_order, large_counts.values, color="#DD8452")
    axes[1].set_title("Large dataset (14,296 MOF)\nDerived PLD category distribution")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "phase1_class_balance.png"), dpi=150)
    plt.close(fig)

    # Metal-type distribution (large dataset only — small dataset has no metal name column)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    metal_counts = large["metal"].value_counts()
    ax.bar(metal_counts.index, metal_counts.values, color="#55A868")
    ax.set_title(f"Metal-type distribution — large dataset ({metal_counts.shape[0]} unique metals)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "phase1_metal_distribution.png"), dpi=150)
    plt.close(fig)

    # Continuous PLD histogram (large dataset only)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(large["Pore Limiting Diameter"], bins=60, color="#8172B2")
    for edge in (2.4, 4.0, 8.0):
        ax.axvline(edge, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Pore Limiting Diameter distribution — large dataset\n(dashed lines = category bin edges)")
    ax.set_xlabel("PLD (Å)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "phase1_pld_histogram.png"), dpi=150)
    plt.close(fig)

    logger.info(f"Saved EDA plots to {output_dir}/")


if __name__ == "__main__":
    small = load_small_dataset()
    large = load_large_dataset()

    print("=" * 60)
    print("SMALL DATASET (2,000-MOF MOFGalaxyNet)")
    print("=" * 60)
    print(f"Shape: {small.shape}")
    print(f"Duplicate refcodes: {small['refcode'].duplicated().sum()}")
    print(f"Category label distribution (raw codes 0-3): {small['pld_category'].value_counts().to_dict()}")
    print(f"Paper-reported counts for comparison: {PAPER_CATEGORY_COUNTS}")

    print()
    print("=" * 60)
    print("LARGE DATASET (14,296-MOF BlackHole / MOFCSD)")
    print("=" * 60)
    print(f"Shape: {large.shape}")
    print(f"Unique metals: {large['metal'].nunique()} (BlackHole's data_utils.py one-hot only covers 4: Cu/Zn/Fe/Co)")
    print(f"Derived PLD category distribution: {large['pld_category'].value_counts().to_dict()}")
    print(f"PLD range: min={large['Pore Limiting Diameter'].min():.3f}, "
          f"max={large['Pore Limiting Diameter'].max():.3f}, "
          f"mean={large['Pore Limiting Diameter'].mean():.3f}")

    plot_eda(small, large)
    print("\nSaved plots: outputs/phase1_class_balance.png, phase1_metal_distribution.png, phase1_pld_histogram.png")
