"""Shared fixtures — small, fast, synthetic stand-ins for the two real datasets, matching
the exact canonical schema `data_ingestion.py::load_dataset()` produces, so every test
exercises the real code paths without the cost of the real 2,000/14,296-row datasets."""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_ingestion import derive_pld_category  # noqa: E402

# All genuinely valid, parseable SMILES — RDKit-dependent code under test needs real
# molecules, not placeholder strings.
VALID_SMILES = [
    "c1ccccc1",                  # benzene
    "OC(=O)c1ccncc1",            # isonicotinic acid
    "OC=O",                      # formic acid
    "OC(=O)c1ccc(cc1)C(=O)O",    # terephthalic acid
    "OC(=O)C(=O)O",              # oxalic acid
]
INVALID_SMILES = "not_a_smiles("
CATEGORY_ORDER = ["nonporous", "small pore", "medium pore", "large pore"]
METALS = ["Cu", "Zn", "Fe", "Co", "Mn"]


def make_small_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "metal_feat_0": rng.rand(n),
        "metal_feat_1": rng.rand(n),
        "metal_feat_2": rng.rand(n),
        "metal_feat_3": rng.rand(n),
        "metal_feat_4": rng.rand(n),
        "metal_feat_5": rng.rand(n),
        "mof_index": np.arange(n),
        "refcode": [f"MOF{i:04d}" for i in range(n)],
        "linker_smiles": [VALID_SMILES[i % len(VALID_SMILES)] for i in range(n)],
        "pld_category": [CATEGORY_ORDER[i % len(CATEGORY_ORDER)] for i in range(n)],
        "metal": [None] * n,
        "pld_value": [None] * n,
    })


def make_large_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    pld_values = rng.uniform(0.5, 15.0, size=n)
    df = pd.DataFrame({
        "refcode": [f"MOF{i:04d}" for i in range(n)],
        "linker_smiles": [VALID_SMILES[i % len(VALID_SMILES)] for i in range(n)],
        "metal": [METALS[i % len(METALS)] for i in range(n)],
        "Largest Cavity Diameter": pld_values + rng.uniform(0.1, 1.0, size=n),
        "pld_value": pld_values,
        "Largest Free Sphere": pld_values * 0.9,
    })
    df["pld_category"] = df["pld_value"].apply(derive_pld_category)
    return df


@pytest.fixture
def small_df() -> pd.DataFrame:
    return make_small_df()


@pytest.fixture
def large_df() -> pd.DataFrame:
    return make_large_df()


@pytest.fixture
def tiny_graph() -> nx.Graph:
    """10 nodes, 2 loose 5-cycles + 1 bridge edge — enough structure for Louvain to
    find >1 community, small enough to reason about by hand."""
    g = nx.Graph()
    g.add_nodes_from(range(10))
    edges = [
        (0, 1, 0.9), (1, 2, 0.8), (2, 3, 0.7), (3, 4, 0.6), (4, 0, 0.5),
        (5, 6, 0.9), (6, 7, 0.8), (7, 8, 0.7), (8, 9, 0.6), (9, 5, 0.5),
        (0, 5, 0.3),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, weight=w)
    return g
