"""Tests for the Lab endpoints — graph topology, pruning, training grid, baselines master
table, and findings. Graph/pruning have no artifact dependency (computed live); the rest
skip gracefully if the relevant trained artifacts aren't present.
"""

import pytest

from webapp.backend.gnn_store import gnn_artifacts_exist
from webapp.backend.model_store import artifacts_exist


def _client():
    from fastapi.testclient import TestClient

    from webapp.backend.api import app

    return TestClient(app)


def test_lab_graph_unknown_dataset_is_404():
    r = _client().get("/api/lab/nope/graph")
    assert r.status_code == 404


def test_lab_graph_large_selects_knn3():
    r = _client().get("/api/lab/large/graph")
    assert r.status_code == 200
    body = r.json()
    assert body["best_graph"] == "knn_3"
    configs = {row["config"] for row in body["topology"]}
    assert configs == {"threshold_0.9", "knn_3", "knn_5", "knn_10"}
    knn3 = next(row for row in body["topology"] if row["config"] == "knn_3")
    assert knn3["isolated_node_rate"] == 0.0


def test_lab_pruning_rejects_out_of_range_tau():
    r = _client().get("/api/lab/large/pruning?tau=1.5")
    assert r.status_code == 400


def test_lab_pruning_small_tau05_matches_expected_retention_order():
    r = _client().get("/api/lab/small/pruning?tau=0.5")
    assert r.status_code == 200
    m = r.json()["metrics"]
    assert m["nodes_after"] < m["nodes_before"]
    assert 40 < m["node_retention_pct"] < 55  # report: 47.9%


@pytest.mark.skipif(not artifacts_exist("large", "classification"), reason="train large/classification first")
def test_lab_baselines_master_table_sorted_descending():
    r = _client().get("/api/lab/large/baselines?task=classification")
    assert r.status_code == 200
    table = r.json()["master_table"]
    accuracies = [row["accuracy"] for row in table]
    assert accuracies == sorted(accuracies, reverse=True)
    assert [row["rank"] for row in table] == list(range(1, len(table) + 1))


@pytest.mark.skipif(not artifacts_exist("large", "classification"), reason="train large/classification first")
def test_lab_findings_reports_best_baseline():
    r = _client().get("/api/lab/large/findings?task=classification")
    assert r.status_code == 200
    body = r.json()
    assert body["best_baseline"]["model"] in ("Random Forest", "k-NN")
    assert "accuracy" in body["best_baseline"]


@pytest.mark.skipif(
    not (artifacts_exist("large", "classification") and gnn_artifacts_exist("large", "classification")),
    reason="train large/classification baseline + GNN artifacts first",
)
def test_lab_findings_flags_baseline_beats_gnn():
    r = _client().get("/api/lab/large/findings?task=classification")
    assert r.status_code == 200
    body = r.json()
    assert "best_gnn" in body
    assert body["baseline_beats_gnn"] is True


@pytest.mark.skipif(not gnn_artifacts_exist("small", "classification"), reason="train small GNN artifacts first")
def test_lab_training_served_models_present():
    r = _client().get("/api/lab/small/training")
    assert r.status_code == 200
    body = r.json()
    assert set(body["served_models"]) == {"GCN", "GraphSAGE", "GAT"}
    assert body["phase3_best_graph"] == "knn_3"


def test_retrain_rejects_bad_target():
    r = _client().post("/api/lab/small/retrain", json={"target": "not-a-real-target"})
    assert r.status_code == 400


def test_retrain_rejects_out_of_range_weight():
    r = _client().post("/api/lab/small/retrain", json={"gravity_degree_weight": 5.0})
    assert r.status_code == 400


def test_job_status_unknown_id_is_404():
    r = _client().get("/api/jobs/does-not-exist")
    assert r.status_code == 404


def test_retrain_small_baselines_runs_in_background_and_completes():
    """Real end-to-end retrain — small dataset only (fast, ~10s). Deliberately never
    exercises the large dataset here; that's real minutes of compute, wrong for a test."""
    import time

    r = _client().post(
        "/api/lab/small/retrain",
        json={"task": "classification", "target": "baselines", "pruning_threshold": 0.35},
    )
    assert r.status_code == 200
    job_id = r.json()["jobs"][0]["job_id"]

    for _ in range(60):
        status = _client().get(f"/api/jobs/{job_id}").json()
        if status["status"] != "running":
            break
        time.sleep(1)
    else:
        pytest.fail("retrain job didn't finish within 60s")

    assert status["status"] == "done"
    assert status["result"]["pruning_threshold"] == 0.35
    assert "Random Forest" in status["result"]["models"]
