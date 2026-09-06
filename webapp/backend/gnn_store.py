"""Trains and persists GCN / GraphSAGE / GAT for a (dataset, task) pair.

For each model architecture, `run_gnn_comparison` already trains it on all 3 graph
variants (Full graph, BH-30, BH-50) as part of the case study's own methodology — this
module keeps whichever variant scored best on the measured test metric for *that specific
architecture* (accuracy for classification, R² for regression), the same criterion the
case study report itself uses to name a "best GNN config". Not necessarily the same
variant across all three architectures — that's expected, and matches the report (e.g.
large/classification's best GCN is BH-30, but GAT and GraphSAGE may differ).

Persists the model's state_dict (torch.save) plus enough metadata to rebuild the exact
architecture and know which graph variant to reconstruct at inference time — the graph
itself isn't serialized here, `webapp/backend/lab.py::get_graph_variant()` rebuilds it
deterministically from the same (dataset, variant_name, gravity_weights) on demand.
"""

import json
import os

import torch

from gnn_training import GAT, GCN, GraphSAGE, run_gnn_comparison  # noqa: E402
from logging_setup import get_logger  # noqa: E402

from webapp.backend.lab import DEFAULT_GRAVITY_WEIGHTS, get_best_graph, get_dataset_and_features  # noqa: E402

logger = get_logger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_CLASSES = {"GCN": GCN, "GraphSAGE": GraphSAGE, "GAT": GAT}
_MODEL_SLUG = {"GCN": "gcn", "GraphSAGE": "graphsage", "GAT": "gat"}


def _gnn_metadata_path(dataset: str, task: str, models_dir: str) -> str:
    return os.path.join(models_dir, f"{dataset}_{task}_gnn_metadata.json")


def _gnn_weights_path(dataset: str, task: str, model_name: str, models_dir: str) -> str:
    return os.path.join(models_dir, f"{dataset}_{task}_{_MODEL_SLUG[model_name]}.pt")


def train_and_save_gnns(
    dataset: str, task: str, models_dir: str = MODELS_DIR, gravity_weights: tuple = DEFAULT_GRAVITY_WEIGHTS,
) -> dict:
    """gravity_weights is the one configuration knob exposed to the retrain button here —
    pruning_threshold isn't, because run_gnn_comparison structurally always trains and
    compares 3 fixed variants (Full graph, BH-30, BH-50); that comparison *is* the case
    study's methodology, not an arbitrary default to override. Different gravity weights
    still produce genuinely different BH-30/BH-50 graphs (the gravity score they prune by
    changes), so this remains a real "different configuration" lever, not a token one."""
    df, features, _ = get_dataset_and_features(dataset)
    best_graph, best_graph_name = get_best_graph(dataset)

    results = run_gnn_comparison(df, features, best_graph, task, gravity_weights, return_models=True)

    rank_key = "accuracy" if task == "classification" else "r2"
    best_per_model: dict = {}
    for (variant_name, model_name), r in results.items():
        score = r["metrics"][rank_key]
        current = best_per_model.get(model_name)
        if current is None or score > current["score"]:
            best_per_model[model_name] = {"variant": variant_name, "score": score, "result": r}

    os.makedirs(models_dir, exist_ok=True)
    saved_models = {}
    for model_name, info in best_per_model.items():
        r = info["result"]
        path = _gnn_weights_path(dataset, task, model_name, models_dir)
        torch.save(r["model"].state_dict(), path)
        saved_models[model_name] = {
            "variant": info["variant"],
            "dim_in": r["dim_in"],
            "dim_h": r["dim_h"],
            "dim_out": r["dim_out"],
            "metrics": {k: v for k, v in r["metrics"].items() if k != "confusion_matrix"},
        }
        logger.info(f"Saved {dataset}/{task}/{model_name} (variant={info['variant']}) -> {path} ({r['metrics']})")

    # All 9 runs' metrics + loss history, not just the 3 that get served — the Lab's
    # "Training" view shows the full comparison grid the case study report itself uses,
    # not just whichever variant ended up fastest to serve predictions with.
    all_runs = [
        {
            "variant": variant_name,
            "model": model_name,
            "metrics": {k: v for k, v in r["metrics"].items() if k != "confusion_matrix"},
            "history": r["history"],
        }
        for (variant_name, model_name), r in results.items()
    ]

    metadata = {
        "dataset": dataset,
        "task": task,
        "phase3_best_graph": best_graph_name,
        "gravity_weights": list(gravity_weights),
        "models": saved_models,
        "all_runs": all_runs,
    }
    with open(_gnn_metadata_path(dataset, task, models_dir), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved GNN metadata for {dataset}/{task} -> {_gnn_metadata_path(dataset, task, models_dir)}")
    return metadata


def gnn_artifacts_exist(dataset: str, task: str, models_dir: str = MODELS_DIR) -> bool:
    return os.path.exists(_gnn_metadata_path(dataset, task, models_dir))


def load_gnn_metadata(dataset: str, task: str, models_dir: str = MODELS_DIR) -> dict:
    path = _gnn_metadata_path(dataset, task, models_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained GNN artifacts for {dataset}/{task} at {path} — run "
            f"`python -m webapp.backend.train_gnns --all` first."
        )
    with open(path) as f:
        return json.load(f)


def load_gnn_model(dataset: str, task: str, model_name: str, models_dir: str = MODELS_DIR):
    """Returns (model, model_info) with the model in eval() mode, ready for a forward pass."""
    metadata = load_gnn_metadata(dataset, task, models_dir)
    info = metadata["models"][model_name]
    model_cls = MODEL_CLASSES[model_name]
    model = model_cls(info["dim_in"], info["dim_h"], info["dim_out"])
    path = _gnn_weights_path(dataset, task, model_name, models_dir)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model, info
