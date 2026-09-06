"""Trains and persists the baseline models (Random Forest, k-NN) that back the Predict
console, plus the metadata a later single-MOF prediction needs to reproduce the exact
same feature vector at inference time.

This module handles baselines only; GNN persistence lives in gnn_store.py and inductive
insertion for a brand-new MOF in inductive.py — split out because per the case study's own
finding (see planning/CASE_STUDY_REPORT.md), baselines are both more accurate and simpler
to serve (no similarity-graph insertion step needed).

Uses the exact same fixed-test-node split as the GNNs (via gnn_training.make_splits,
called through the existing run_baseline_comparison), so the accuracy/kappa/R² numbers
saved here match planning/CASE_STUDY_REPORT.md rather than being computed on a different,
merely similar-looking split.
"""

import json
import os

import joblib
import numpy as np

from baseline_models import run_baseline_comparison  # noqa: E402
from feature_engineering import _molecular_weight  # noqa: E402
from logging_setup import get_logger  # noqa: E402

from webapp.backend.lab import (  # noqa: E402
    DEFAULT_GRAVITY_WEIGHTS,
    FEATURE_SCHEME_BY_DATASET,
    get_best_graph,
    get_dataset_and_features,
    get_pruned_graph,
)

logger = get_logger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_PRUNING_THRESHOLD = 0.3

_MODEL_SLUG = {"Random Forest": "random_forest", "k-NN": "knn"}


def _metadata_path(dataset: str, task: str, models_dir: str) -> str:
    return os.path.join(models_dir, f"{dataset}_{task}_metadata.json")


def _model_path(dataset: str, task: str, model_name: str, models_dir: str) -> str:
    return os.path.join(models_dir, f"{dataset}_{task}_{_MODEL_SLUG[model_name]}.joblib")


def train_and_save_baselines(
    dataset: str, task: str, models_dir: str = MODELS_DIR,
    gravity_weights: tuple = DEFAULT_GRAVITY_WEIGHTS, pruning_threshold: float = DEFAULT_PRUNING_THRESHOLD,
) -> dict:
    """Trains RF + k-NN for one (dataset, task) pair, saves both to disk, returns the
    metadata dict that gets written alongside them (also used directly by callers that
    just trained and want the result without a re-read).

    gravity_weights/pruning_threshold only affect which nodes end up as the fixed test
    set (Black Hole picks it) — the retrain button (webapp/backend/jobs.py) exposes both
    so a different configuration produces a genuinely different, re-evaluated split."""
    df, features, feat_meta = get_dataset_and_features(dataset)
    scheme = FEATURE_SCHEME_BY_DATASET[dataset]
    _, best_graph_name = get_best_graph(dataset)
    bh = get_pruned_graph(dataset, pruning_threshold, gravity_weights)
    fixed_test_nodes = bh["fixed_test_nodes"]

    results = run_baseline_comparison(df, features, task, fixed_test_nodes, return_models=True)

    os.makedirs(models_dir, exist_ok=True)
    saved_models = {}
    for name, r in results.items():
        path = _model_path(dataset, task, name, models_dir)
        joblib.dump(r["model"], path)
        saved_models[name] = {"metrics": r["metrics"]}
        logger.info(f"Saved {dataset}/{task}/{name} -> {path} ({r['metrics']})")

    inference_meta = {
        "scheme": scheme,
        "n_unique_metals": feat_meta.get("n_unique_metals"),
        "metal_to_index": feat_meta.get("metal_to_index"),
        "invalid_smiles_count": feat_meta.get("invalid_smiles_count"),
    }
    if scheme == "compact":
        inference_meta["compact_columns"] = feat_meta["columns"]
        metal_cols = [f"metal_feat_{i}" for i in range(6)]
        linker_mw = df["linker_smiles"].apply(_molecular_weight)
        raw = np.column_stack([df[metal_cols].values, linker_mw.fillna(linker_mw.median()).values])
        inference_meta["scaler_min"] = raw.min(axis=0).tolist()
        inference_meta["scaler_max"] = raw.max(axis=0).tolist()
        inference_meta["metal_feat_medians"] = df[metal_cols].median().tolist()
    else:
        inference_meta["pore_geometry_columns"] = feat_meta["pore_geometry_columns"]
        inference_meta["pore_geometry_medians"] = df[feat_meta["pore_geometry_columns"]].median().tolist()

    metadata = {
        "dataset": dataset,
        "task": task,
        "best_graph": best_graph_name,
        "gravity_weights": list(gravity_weights),
        "pruning_threshold": pruning_threshold,
        "n_rows": len(df),
        "models": saved_models,
        "inference": inference_meta,
    }
    with open(_metadata_path(dataset, task, models_dir), "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved metadata for {dataset}/{task} -> {_metadata_path(dataset, task, models_dir)}")
    return metadata


def load_metadata(dataset: str, task: str, models_dir: str = MODELS_DIR) -> dict:
    path = _metadata_path(dataset, task, models_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained artifacts for {dataset}/{task} at {path} — run "
            f"`python -m webapp.backend.train_models --all` first."
        )
    with open(path) as f:
        return json.load(f)


def load_model(dataset: str, task: str, model_name: str, models_dir: str = MODELS_DIR):
    path = _model_path(dataset, task, model_name, models_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained {model_name} artifact for {dataset}/{task} at {path} — run "
            f"`python -m webapp.backend.train_models --all` first."
        )
    return joblib.load(path)


def artifacts_exist(dataset: str, task: str, models_dir: str = MODELS_DIR) -> bool:
    return os.path.exists(_metadata_path(dataset, task, models_dir))
