"""Tests for the FastAPI Predict console backend.

Skips gracefully if the trained artifacts aren't present (run
`python -m webapp.backend.train_models --all` first) rather than training inside the test
suite — training takes real minutes for the large dataset, out of place in a fast test run.
"""

import pytest

from webapp.backend.gnn_store import gnn_artifacts_exist
from webapp.backend.model_store import artifacts_exist


def _client():
    from fastapi.testclient import TestClient

    from webapp.backend.api import app

    return TestClient(app)


def test_health():
    r = _client().get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_datasets_endpoint():
    r = _client().get("/api/datasets")
    assert r.status_code == 200
    body = r.json()
    assert body["large"]["rows"] == 14296
    assert body["large"]["unique_metals"] == 53
    assert body["small"]["rows"] == 2004
    assert sum(body["small"]["category_counts"].values()) == body["small"]["rows"]


def test_predict_unknown_dataset_is_400():
    r = _client().post("/api/predict", json={"dataset": "nope", "smiles": "C"})
    assert r.status_code == 400


def test_predict_small_regression_is_rejected():
    r = _client().post("/api/predict", json={"dataset": "small", "task": "regression", "smiles": "C"})
    assert r.status_code == 400


@pytest.mark.skipif(not artifacts_exist("large", "classification"), reason="train large/classification artifacts first")
def test_predict_large_classification_ranks_by_measured_accuracy():
    """Baseline rows specifically — deliberately doesn't assert the total leaderboard size,
    since GNN rows join in once those artifacts exist too (see
    test_predict_large_classification_baselines_beat_gnns for the full 5-model case)."""
    r = _client().post(
        "/api/predict",
        json={"dataset": "large", "task": "classification", "metal": "Cu", "smiles": "OC(=O)c1ccc(cc1)C(=O)O"},
    )
    assert r.status_code == 200
    board = r.json()["leaderboard"]
    baseline_rows = [row for row in board if row["type"] == "baseline"]
    assert {row["model"] for row in baseline_rows} == {"Random Forest", "k-NN"}
    assert baseline_rows[0]["measured_accuracy"] >= baseline_rows[1]["measured_accuracy"]
    assert board[0]["rank"] == 1
    assert set(board[0]["confidence"]) == {"nonporous", "small pore", "medium pore", "large pore"}
    assert r.json()["caveats"] == {"used_median_geometry": True}


@pytest.mark.skipif(not artifacts_exist("large", "classification"), reason="train large/classification artifacts first")
def test_predict_large_classification_uses_supplied_geometry_without_caveat():
    r = _client().post(
        "/api/predict",
        json={
            "dataset": "large", "task": "classification", "metal": "Zn", "smiles": "OC(=O)c1ccncc1",
            "largest_cavity_diameter": 6.2, "largest_free_sphere": 5.1,
        },
    )
    assert r.status_code == 200
    assert r.json()["caveats"] == {}


@pytest.mark.skipif(not artifacts_exist("small", "classification"), reason="train small/classification artifacts first")
def test_predict_small_classification_flags_median_metal_feat():
    r = _client().post("/api/predict", json={"dataset": "small", "smiles": "OC(=O)c1ccncc1"})
    assert r.status_code == 200
    assert r.json()["caveats"] == {"used_median_metal_feat": True}


@pytest.mark.skipif(not artifacts_exist("large", "regression"), reason="train large/regression artifacts first")
def test_predict_large_regression_ranks_by_r2():
    r = _client().post(
        "/api/predict",
        json={"dataset": "large", "task": "regression", "metal": "Cu", "smiles": "OC(=O)c1ccc(cc1)C(=O)O"},
    )
    assert r.status_code == 200
    board = r.json()["leaderboard"]
    assert board[0]["measured_r2"] >= board[1]["measured_r2"]
    assert isinstance(board[0]["predicted_pld"], float)


@pytest.mark.skipif(
    not (artifacts_exist("small", "classification") and gnn_artifacts_exist("small", "classification")),
    reason="train small/classification baseline + GNN artifacts first",
)
def test_predict_small_classification_includes_all_5_models_ranked():
    r = _client().post("/api/predict", json={"dataset": "small", "smiles": "OC(=O)c1ccncc1"})
    assert r.status_code == 200
    board = r.json()["leaderboard"]
    assert {row["model"] for row in board} == {"Random Forest", "k-NN", "GCN", "GraphSAGE", "GAT"}
    assert [row["rank"] for row in board] == list(range(1, 6))
    accuracies = [row["measured_accuracy"] for row in board]
    assert accuracies == sorted(accuracies, reverse=True)
    for row in board:
        if row["type"] == "graph":
            assert row["n_neighbors_found"] == 3
            assert "graph_variant" in row


@pytest.mark.skipif(
    not (artifacts_exist("large", "classification") and gnn_artifacts_exist("large", "classification")),
    reason="train large/classification baseline + GNN artifacts first",
)
def test_predict_large_classification_baselines_beat_gnns():
    """Regression test for the case study's own headline finding — if this ever flips,
    it's a real result worth knowing about, not just an assertion to relax."""
    r = _client().post(
        "/api/predict",
        json={"dataset": "large", "task": "classification", "metal": "Cu", "smiles": "OC(=O)c1ccc(cc1)C(=O)O"},
    )
    assert r.status_code == 200
    board = r.json()["leaderboard"]
    best_baseline = max(row["measured_accuracy"] for row in board if row["type"] == "baseline")
    best_graph = max(row["measured_accuracy"] for row in board if row["type"] == "graph")
    assert best_baseline > best_graph
