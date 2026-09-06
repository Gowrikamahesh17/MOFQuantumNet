"""FastAPI backend for the MOFQuantumNet Predict + Lab console.

Single process serves both this API and the static frontend (webapp/frontend/) —
one `uvicorn webapp.backend.api:app` command, one URL, per the "one app, not two"
decision in planning/DEVELOPMENT_TODO.md (Part 2).

Run: .venv/bin/uvicorn webapp.backend.api:app --reload
Requires trained artifacts first: python -m webapp.backend.train_models --all
"""

import os
from collections import Counter

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webapp.backend.featurize import featurize_new_mof  # noqa: E402
from webapp.backend.gnn_store import gnn_artifacts_exist, load_gnn_metadata, train_and_save_gnns  # noqa: E402
from webapp.backend.inductive import predict_with_gnn  # noqa: E402
from webapp.backend.jobs import get_job, list_jobs, start_job  # noqa: E402
from webapp.backend.lab import (  # noqa: E402
    get_df,
    get_dataset_and_features,
    get_graph_sample,
    get_graphs_and_topology,
    get_pruned_graph,
)
from webapp.backend.model_store import artifacts_exist, load_metadata, load_model, train_and_save_baselines  # noqa: E402
from logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = FastAPI(title="MOFQuantumNet Console API")


def _dataset_stats(dataset: str) -> dict:
    df = get_df(dataset)
    category_order = ["nonporous", "small pore", "medium pore", "large pore"]
    counts = df["pld_category"].value_counts().reindex(category_order).fillna(0).astype(int)
    stats = {
        "dataset": dataset,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_refcodes": int(df["refcode"].duplicated().sum()),
        "category_counts": counts.to_dict(),
        "unique_metals": int(df["metal"].nunique()) if df["metal"].notna().any() else None,
    }
    if df["pld_value"].notna().any():
        bins = pd.cut(df["pld_value"], bins=20).value_counts().sort_index()
        stats["pld_histogram"] = [int(v) for v in bins.values]
    if df["metal"].notna().any():
        metal_counts = df["metal"].value_counts()
        stats["metal_distribution"] = {str(k): int(v) for k, v in metal_counts.items()}
    return stats


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/datasets")
def datasets():
    return {"small": _dataset_stats("small"), "large": _dataset_stats("large")}


@app.get("/api/datasets/{dataset}")
def dataset_detail(dataset: str):
    if dataset not in ("small", "large"):
        raise HTTPException(404, f"unknown dataset {dataset!r}")
    return _dataset_stats(dataset)


def _require_dataset(dataset: str) -> None:
    if dataset not in ("small", "large"):
        raise HTTPException(404, f"unknown dataset {dataset!r}")


@app.get("/api/lab/{dataset}/features")
def lab_features(dataset: str):
    """Feature-engineering summary (scheme, dimensions, malformed-SMILES count) — the one
    Lab stage that previously had no endpoint at all."""
    _require_dataset(dataset)
    _, features, feat_meta = get_dataset_and_features(dataset)
    return {
        "dataset": dataset,
        "scheme": feat_meta["scheme"],
        "dimensions": int(features.shape[1]),
        "invalid_smiles_count": feat_meta.get("invalid_smiles_count"),
        "n_unique_metals": feat_meta.get("n_unique_metals"),
    }


@app.get("/api/lab/{dataset}/graph")
def lab_graph(dataset: str):
    """Real topology comparison table (Phase 3) — replaces the frozen numbers the mockup
    hardcoded with the same computation, run live and cached in-process."""
    _require_dataset(dataset)
    graphs, topology, best = get_graphs_and_topology(dataset)
    table = topology.reset_index().to_dict(orient="records")
    degrees = [d for _, d in graphs[best].degree()]
    degree_counts = Counter(degrees)
    max_degree = max(degree_counts) if degree_counts else 0
    degree_histogram = [degree_counts.get(d, 0) for d in range(max_degree + 1)]
    return {"dataset": dataset, "best_graph": best, "topology": table, "degree_histogram": degree_histogram}


@app.get("/api/lab/{dataset}/graph-sample")
def lab_graph_sample(dataset: str):
    """A real (not procedurally faked) sampled subgraph for the network visualization —
    node positions via spring_layout, colored by each node's actual PLD category."""
    _require_dataset(dataset)
    sample = get_graph_sample(dataset)
    df = get_df(dataset)
    node_ids, pos, graph = sample["node_ids"], sample["pos"], sample["graph"]
    index_of = {nid: i for i, nid in enumerate(node_ids)}
    nodes = [
        {"id": int(nid), "x": float(pos[nid][0]), "y": float(pos[nid][1]), "category": df["pld_category"].iloc[nid]}
        for nid in node_ids
    ]
    edges = [
        [index_of[u], index_of[v]]
        for u, v in graph.subgraph(node_ids).edges()
        if u in index_of and v in index_of
    ]
    return {
        "dataset": dataset, "best_graph": sample["best_graph"],
        "n_sampled": len(node_ids), "n_total": graph.number_of_nodes(),
        "nodes": nodes, "edges": edges,
    }


@app.get("/api/lab/{dataset}/pruning-sample")
def lab_pruning_sample(dataset: str, tau: float = 0.3):
    """Which of graph-sample's *same* nodes survive pruning at this τ, so the frontend can
    render before/after using identical positions — a node disappears from its spot, the
    layout never re-shuffles."""
    _require_dataset(dataset)
    if not 0.0 <= tau <= 0.9:
        raise HTTPException(400, "tau must be in [0.0, 0.9]")
    sample = get_graph_sample(dataset)
    bh = get_pruned_graph(dataset, tau)
    survived_ids = set(bh["graph"].nodes())
    fixed_ids = set(bh["fixed_test_nodes"])
    nodes = [
        {"id": int(nid), "survived": nid in survived_ids, "fixed": nid in fixed_ids}
        for nid in sample["node_ids"]
    ]
    return {"dataset": dataset, "tau": tau, "nodes": nodes}


@app.get("/api/lab/{dataset}/pruning")
def lab_pruning(dataset: str, tau: float = 0.3):
    """Real Black Hole retention stats at a given pruning threshold (default weights,
    equal 0.33/0.33/0.33 per the paper's own main configuration)."""
    _require_dataset(dataset)
    if not 0.0 <= tau <= 0.9:
        raise HTTPException(400, "tau must be in [0.0, 0.9]")
    result = get_pruned_graph(dataset, tau)
    return {"dataset": dataset, "tau": tau, "metrics": result["metrics"]}


def _require_valid_task(dataset: str, task: str) -> None:
    if dataset == "small" and task == "regression":
        raise HTTPException(400, "small dataset has no continuous PLD — classification only")


@app.get("/api/lab/{dataset}/training")
def lab_training(dataset: str, task: str = "classification"):
    """The real 9-run GCN/GraphSAGE/GAT comparison (3 architectures x 3 graph variants),
    served from whatever's already been trained via `train_gnns.py` — this endpoint never
    triggers training itself (that's minutes of real compute, wrong for a GET request).
    `served_models` is what POST /api/predict actually uses (best variant per architecture);
    `all_runs` is the full grid for the Lab's training view, when available — artifacts
    trained before this field existed just won't have it, handled gracefully rather than
    erroring."""
    _require_dataset(dataset)
    _require_valid_task(dataset, task)
    if not gnn_artifacts_exist(dataset, task):
        raise HTTPException(
            503,
            f"GNN models not trained yet for {dataset}/{task} — run "
            f"`python -m webapp.backend.train_gnns --dataset {dataset} --task {task}` "
            "(real training time: ~35-40s small dataset, ~9 minutes large dataset).",
        )
    metadata = load_gnn_metadata(dataset, task)
    return {
        "dataset": dataset,
        "task": task,
        "phase3_best_graph": metadata["phase3_best_graph"],
        "served_models": metadata["models"],
        "all_runs": metadata.get("all_runs", []),
    }


@app.get("/api/lab/{dataset}/baselines")
def lab_baselines(dataset: str, task: str = "classification"):
    """Master comparison table — baselines always, GNN rows once trained. Sorted by the
    task's own rank metric (accuracy for classification, R² for regression), matching
    src/baseline_models.py::build_master_comparison_table's own convention."""
    _require_dataset(dataset)
    _require_valid_task(dataset, task)
    if not artifacts_exist(dataset, task):
        raise HTTPException(
            503,
            f"no baseline artifacts for {dataset}/{task} yet — run "
            f"`python -m webapp.backend.train_models --dataset {dataset} --task {task}`",
        )
    sort_key = "accuracy" if task == "classification" else "r2"
    baseline_meta = load_metadata(dataset, task)

    rows = []
    for name, info in baseline_meta["models"].items():
        m = {k: v for k, v in info["metrics"].items() if k != "confusion_matrix"}
        rows.append({"configuration": f"Flat features — {name}", "type": "Baseline", **m})
    if gnn_artifacts_exist(dataset, task):
        gnn_meta = load_gnn_metadata(dataset, task)
        for name, info in gnn_meta["models"].items():
            rows.append({"configuration": f"{info['variant']} — {name}", "type": "GNN", **info["metrics"]})

    rows.sort(key=lambda r: r.get(sort_key, float("-inf")), reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    return {"dataset": dataset, "task": task, "master_table": rows}


@app.get("/api/lab/{dataset}/findings")
def lab_findings(dataset: str, task: str = "classification"):
    """The headline scoreboard — computed from whatever's actually been trained, not
    frozen prose. Mirrors the mockup's "Findings" stage, but live."""
    _require_dataset(dataset)
    _require_valid_task(dataset, task)
    if not artifacts_exist(dataset, task):
        raise HTTPException(503, f"no results yet for {dataset}/{task}")

    sort_key = "accuracy" if task == "classification" else "r2"
    baseline_meta = load_metadata(dataset, task)
    best_baseline_name = max(baseline_meta["models"], key=lambda n: baseline_meta["models"][n]["metrics"][sort_key])
    best_baseline_metrics = baseline_meta["models"][best_baseline_name]["metrics"]
    result = {
        "dataset": dataset,
        "task": task,
        "best_baseline": {
            "model": best_baseline_name,
            **{k: v for k, v in best_baseline_metrics.items() if k != "confusion_matrix"},
        },
    }
    if gnn_artifacts_exist(dataset, task):
        gnn_meta = load_gnn_metadata(dataset, task)
        best_gnn_name = max(gnn_meta["models"], key=lambda n: gnn_meta["models"][n]["metrics"][sort_key])
        best_gnn_info = gnn_meta["models"][best_gnn_name]
        result["best_gnn"] = {"model": best_gnn_name, "variant": best_gnn_info["variant"], **best_gnn_info["metrics"]}
        result["baseline_beats_gnn"] = best_baseline_metrics[sort_key] > best_gnn_info["metrics"][sort_key]
    return result


class RetrainRequest(BaseModel):
    task: str = "classification"
    target: str = "baselines"  # "baselines" | "gnns" | "both"
    gravity_degree_weight: float = 0.33
    gravity_betweenness_weight: float = 0.33
    gravity_edge_weight_sum_weight: float = 0.33
    pruning_threshold: float = 0.3  # baselines only — GNNs always compare Full/BH-30/BH-50 as a set


@app.post("/api/lab/{dataset}/retrain")
def retrain(dataset: str, req: RetrainRequest):
    """Kicks off real training in the background and returns immediately with job ids to
    poll — never blocks the request thread, since a GNN run is minutes of real compute.
    Different gravity weights (and, for baselines, a different pruning threshold) produce
    a genuinely different fixed-test-node split and re-evaluated metrics, not a no-op."""
    _require_dataset(dataset)
    _require_valid_task(dataset, req.task)
    if req.target not in ("baselines", "gnns", "both"):
        raise HTTPException(400, "target must be 'baselines', 'gnns', or 'both'")
    if not 0.0 <= req.pruning_threshold <= 0.9:
        raise HTTPException(400, "pruning_threshold must be in [0.0, 0.9]")
    weights = (req.gravity_degree_weight, req.gravity_betweenness_weight, req.gravity_edge_weight_sum_weight)
    if any(not 0.0 <= w <= 1.0 for w in weights):
        raise HTTPException(400, "gravity weights must each be in [0.0, 1.0]")
    logger.info(f"Retrain request: dataset={dataset} task={req.task} target={req.target} weights={weights} tau={req.pruning_threshold}")

    jobs = []
    if req.target in ("baselines", "both"):
        job_id = start_job(
            "baselines", train_and_save_baselines, dataset, req.task,
            gravity_weights=weights, pruning_threshold=req.pruning_threshold,
        )
        jobs.append({"job_id": job_id, "kind": "baselines"})
    if req.target in ("gnns", "both"):
        job_id = start_job("gnns", train_and_save_gnns, dataset, req.task, gravity_weights=weights)
        jobs.append({"job_id": job_id, "kind": "gnns"})

    return {"dataset": dataset, "task": req.task, "jobs": jobs}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job {job_id!r}")
    return job


@app.get("/api/jobs")
def jobs_list():
    return {"jobs": list_jobs()}


class PredictRequest(BaseModel):
    dataset: str
    task: str = "classification"
    smiles: str
    metal: str | None = None
    metal_feat: list[float] | None = None
    largest_cavity_diameter: float | None = None
    largest_free_sphere: float | None = None


CATEGORY_ORDER = ["nonporous", "small pore", "medium pore", "large pore"]
BASELINE_MODEL_NAMES = ["Random Forest", "k-NN"]
GNN_MODEL_NAMES = ["GCN", "GraphSAGE", "GAT"]


@app.post("/api/predict")
def predict(req: PredictRequest):
    logger.info(f"Predict request: dataset={req.dataset} task={req.task} metal={req.metal!r} smiles={req.smiles!r}")
    if req.dataset not in ("small", "large"):
        raise HTTPException(400, f"unknown dataset {req.dataset!r}")
    _require_valid_task(req.dataset, req.task)
    if not artifacts_exist(req.dataset, req.task):
        raise HTTPException(
            503,
            f"no trained artifacts for {req.dataset}/{req.task} yet — run "
            f"`python -m webapp.backend.train_models --dataset {req.dataset} --task {req.task}`",
        )

    metadata = load_metadata(req.dataset, req.task)

    try:
        x, caveats = featurize_new_mof(
            metadata,
            smiles=req.smiles,
            metal=req.metal,
            metal_feat=req.metal_feat,
            largest_cavity_diameter=req.largest_cavity_diameter,
            largest_free_sphere=req.largest_free_sphere,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    rows = []
    for name in BASELINE_MODEL_NAMES:
        model = load_model(req.dataset, req.task, name)
        measured = metadata["models"][name]["metrics"]
        row = {"model": name, "type": "baseline"}
        if req.task == "classification":
            proba = model.predict_proba(x)[0]
            pred_idx = int(proba.argmax())
            row["predicted_category"] = CATEGORY_ORDER[pred_idx]
            row["confidence"] = {CATEGORY_ORDER[i]: round(float(p), 4) for i, p in enumerate(proba)}
            row["measured_accuracy"] = measured["accuracy"]
            row["measured_cohen_kappa"] = measured["cohen_kappa"]
            row["rank_metric"] = measured["accuracy"]
        else:
            pred = float(model.predict(x)[0])
            row["predicted_pld"] = round(pred, 3)
            row["measured_mae"] = measured["mae"]
            row["measured_r2"] = measured["r2"]
            row["rank_metric"] = measured["r2"]
        rows.append(row)

    graph_note = None
    if gnn_artifacts_exist(req.dataset, req.task):
        for name in GNN_MODEL_NAMES:
            try:
                gnn_result = predict_with_gnn(
                    req.dataset, req.task, name, x,
                    smiles=req.smiles, metal=req.metal, metal_feat=req.metal_feat,
                )
            except (FileNotFoundError, KeyError):
                continue
            row = {
                "model": name, "type": "graph",
                "graph_variant": gnn_result["variant_used"],
                "n_neighbors_found": gnn_result["n_neighbors_found"],
            }
            measured = gnn_result["measured_metrics"]
            if req.task == "classification":
                probs = gnn_result["probs"]
                pred_idx = int(probs.argmax())
                row["predicted_category"] = CATEGORY_ORDER[pred_idx]
                row["confidence"] = {CATEGORY_ORDER[i]: round(float(p), 4) for i, p in enumerate(probs)}
                row["measured_accuracy"] = measured["accuracy"]
                row["measured_cohen_kappa"] = measured["cohen_kappa"]
                row["rank_metric"] = measured["accuracy"]
            else:
                row["predicted_pld"] = round(gnn_result["value"], 3)
                row["measured_mae"] = measured["mae"]
                row["measured_r2"] = measured["r2"]
                row["rank_metric"] = measured["r2"]
            rows.append(row)
    else:
        graph_note = (
            f"Graph models (GCN/GraphSAGE/GAT) not trained yet for {req.dataset}/{req.task} — run "
            f"`python -m webapp.backend.train_gnns --dataset {req.dataset} --task {req.task}`."
        )

    rows.sort(key=lambda r: r["rank_metric"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    return {
        "dataset": req.dataset,
        "task": req.task,
        "input": {"smiles": req.smiles, "metal": req.metal},
        "caveats": caveats,
        "leaderboard": rows,
        "note": graph_note or (
            "Ranked by each model's own measured accuracy/R² on the held-out test set. "
            "Graph models were inserted into their trained similarity graph via 3 nearest "
            "chemical neighbors before predicting — see graph_variant/n_neighbors_found per row."
        ),
    }


if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
