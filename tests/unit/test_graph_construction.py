"""Unit tests for src/graph_construction.py — similarity graph construction + topology metrics."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from graph_construction import (
    KNN_VALUES,
    THRESHOLD_PHI,
    build_similarity_graphs,
    compute_topology_metrics,
    sample_subgraph_for_viz,
    save_graphs,
    select_best_graph,
)

EXPECTED_KEYS = {f"threshold_{THRESHOLD_PHI}"} | {f"knn_{k}" for k in KNN_VALUES}


class TestBuildSimilarityGraphs:
    def test_returns_all_four_expected_configs(self, small_df):
        graphs = build_similarity_graphs(small_df, "small")
        assert set(graphs.keys()) == EXPECTED_KEYS

    @pytest.mark.parametrize("dataset_fixture", ["small_df", "large_df"])
    def test_every_graph_has_exactly_n_nodes(self, dataset_fixture, request):
        df = request.getfixturevalue(dataset_fixture)
        dataset = "small" if dataset_fixture == "small_df" else "large"
        graphs = build_similarity_graphs(df, dataset)
        for name, g in graphs.items():
            assert g.number_of_nodes() == len(df), f"{name} has wrong node count"

    def test_node_ids_are_positional_range(self, small_df):
        graphs = build_similarity_graphs(small_df, "small")
        for g in graphs.values():
            assert set(g.nodes()) == set(range(len(small_df)))

    def test_no_self_loops(self, small_df):
        graphs = build_similarity_graphs(small_df, "small")
        for name, g in graphs.items():
            assert nx.number_of_selfloops(g) == 0, f"{name} has self-loops"

    def test_graphs_are_undirected(self, small_df):
        graphs = build_similarity_graphs(small_df, "small")
        for g in graphs.values():
            assert isinstance(g, nx.Graph) and not isinstance(g, nx.DiGraph)

    @pytest.mark.parametrize("k", KNN_VALUES)
    def test_knn_guarantees_minimum_degree_k(self, small_df, k):
        """The entire point of k-NN construction: every node gets >= k edges,
        directly solving the fixed-threshold graph's isolated-node problem."""
        graphs = build_similarity_graphs(small_df, "small")
        degrees = dict(graphs[f"knn_{k}"].degree())
        assert min(degrees.values()) >= k

    def test_edge_weights_are_bounded_zero_to_one(self, small_df):
        graphs = build_similarity_graphs(small_df, "small")
        for name, g in graphs.items():
            for _, _, data in g.edges(data=True):
                assert -1e-6 <= data["weight"] <= 1.0 + 1e-6, f"{name} has out-of-range weight"

    def test_large_dataset_uses_categorical_metal_similarity(self, large_df):
        """Documented adaptation: large dataset has no numeric metal descriptor, so
        two MOFs sharing a metal should be measurably more similar than two that don't,
        all else equal."""
        df = large_df.copy()
        df.loc[0, "linker_smiles"] = df.loc[1, "linker_smiles"]  # force identical linker
        df.loc[0, "metal"] = df.loc[1, "metal"]                  # force same metal
        df.loc[2, "linker_smiles"] = df.loc[1, "linker_smiles"]  # same linker, different metal
        if df.loc[2, "metal"] == df.loc[1, "metal"]:
            df.loc[2, "metal"] = [m for m in df["metal"].unique() if m != df.loc[1, "metal"]][0]

        graphs = build_similarity_graphs(df, "large")
        g = graphs[f"knn_{KNN_VALUES[0]}"]
        sim_same_metal = g.get_edge_data(0, 1, {}).get("weight", 0)
        sim_diff_metal = g.get_edge_data(1, 2, {}).get("weight", 0)
        assert sim_same_metal >= sim_diff_metal


class TestComputeTopologyMetrics:
    def test_returns_expected_columns(self, small_df):
        graphs = build_similarity_graphs(small_df, "small")
        topology = compute_topology_metrics(graphs)
        expected = {"edges", "isolated_node_rate", "mean_degree", "num_communities", "modularity"}
        assert expected.issubset(topology.columns)

    def test_isolated_rate_is_a_fraction(self, small_df):
        graphs = build_similarity_graphs(small_df, "small")
        topology = compute_topology_metrics(graphs)
        assert (topology["isolated_node_rate"] >= 0).all()
        assert (topology["isolated_node_rate"] <= 1).all()

    def test_knn_graphs_have_zero_isolated_nodes(self, small_df):
        graphs = build_similarity_graphs(small_df, "small")
        topology = compute_topology_metrics(graphs)
        for k in KNN_VALUES:
            assert topology.loc[f"knn_{k}", "isolated_node_rate"] == 0.0

    def test_mean_degree_matches_definition(self, small_df):
        graphs = build_similarity_graphs(small_df, "small")
        topology = compute_topology_metrics(graphs)
        for name, g in graphs.items():
            expected = 2 * g.number_of_edges() / g.number_of_nodes()
            assert topology.loc[name, "mean_degree"] == pytest.approx(expected)

    def test_empty_graph_does_not_crash(self):
        g = nx.Graph()
        g.add_nodes_from(range(5))  # no edges at all
        topology = compute_topology_metrics({"empty": g})
        assert topology.loc["empty", "isolated_node_rate"] == 1.0
        assert topology.loc["empty", "modularity"] == 0.0


class TestSelectBestGraph:
    def test_picks_lowest_isolated_rate(self):
        topology = pd.DataFrame({
            "isolated_node_rate": [0.2, 0.0, 0.0],
            "modularity": [0.9, 0.5, 0.6],
        }, index=["threshold", "knn_a", "knn_b"])
        assert select_best_graph(topology) == "knn_b"  # tied at 0.0 isolated, knn_b has higher modularity

    def test_tie_break_is_highest_modularity(self):
        topology = pd.DataFrame({
            "isolated_node_rate": [0.0, 0.0],
            "modularity": [0.3, 0.99],
        }, index=["a", "b"])
        assert select_best_graph(topology) == "b"

    def test_single_candidate(self):
        topology = pd.DataFrame({"isolated_node_rate": [0.1], "modularity": [0.5]}, index=["only"])
        assert select_best_graph(topology) == "only"


class TestSampleSubgraphForViz:
    def test_returns_all_nodes_when_graph_smaller_than_cap(self):
        g = nx.cycle_graph(10)
        sampled = sample_subgraph_for_viz(g, max_nodes=90)
        assert set(sampled) == set(g.nodes())

    def test_caps_at_max_nodes_for_larger_graph(self):
        g = nx.connected_watts_strogatz_graph(200, 4, 0.1, seed=1)
        sampled = sample_subgraph_for_viz(g, max_nodes=50)
        assert len(sampled) == 50

    def test_sample_has_no_duplicate_nodes(self):
        g = nx.connected_watts_strogatz_graph(200, 4, 0.1, seed=1)
        sampled = sample_subgraph_for_viz(g, max_nodes=50)
        assert len(sampled) == len(set(sampled))

    def test_sample_is_a_subset_of_graph_nodes(self):
        g = nx.connected_watts_strogatz_graph(200, 4, 0.1, seed=1)
        sampled = sample_subgraph_for_viz(g, max_nodes=50)
        assert set(sampled).issubset(set(g.nodes()))

    def test_sample_stays_connected_via_bfs(self):
        """The BFS-from-highest-degree-node construction guarantees the induced
        subgraph on the sample is connected, not an arbitrary scattered set."""
        g = nx.connected_watts_strogatz_graph(200, 4, 0.1, seed=1)
        sampled = sample_subgraph_for_viz(g, max_nodes=50)
        sub = g.subgraph(sampled)
        assert nx.is_connected(sub)

    def test_deterministic_given_same_graph(self):
        g = nx.connected_watts_strogatz_graph(150, 4, 0.1, seed=2)
        assert sample_subgraph_for_viz(g, max_nodes=40) == sample_subgraph_for_viz(g, max_nodes=40)

    def test_empty_graph_returns_empty_sample(self):
        assert sample_subgraph_for_viz(nx.Graph(), max_nodes=50) == []

    def test_disconnected_graph_does_not_crash_and_respects_cap(self):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3)])
        g.add_edges_from([(10, 11), (11, 12)])  # separate component, unreachable from the BFS seed
        sampled = sample_subgraph_for_viz(g, max_nodes=3)
        assert len(sampled) == 3


class TestSaveGraphs:
    def test_writes_expected_files(self, small_df, tmp_path):
        graphs = build_similarity_graphs(small_df, "small")
        topology = compute_topology_metrics(graphs)
        best = select_best_graph(topology)
        save_graphs(graphs, small_df["refcode"].values, topology, best, "small", output_dir=str(tmp_path))

        for name in graphs:
            assert (tmp_path / f"graph_small_{name}.csv").exists()
        assert (tmp_path / "phase3_topology_small.csv").exists()
        assert (tmp_path / "phase3_best_graph_small.json").exists()

    def test_edge_list_uses_refcodes_not_integer_ids(self, small_df, tmp_path):
        graphs = build_similarity_graphs(small_df, "small")
        topology = compute_topology_metrics(graphs)
        best = select_best_graph(topology)
        save_graphs(graphs, small_df["refcode"].values, topology, best, "small", output_dir=str(tmp_path))

        edges_df = pd.read_csv(tmp_path / f"graph_small_{best}.csv")
        if len(edges_df):
            assert edges_df["source"].iloc[0] in set(small_df["refcode"])
