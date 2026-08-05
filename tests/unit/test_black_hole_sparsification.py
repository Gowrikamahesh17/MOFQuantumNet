"""Unit tests for src/black_hole_sparsification.py — the most correctness-sensitive
ported module: gravity scoring, PLD-stratified node pruning, and edge pruning."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from networkx.algorithms.community import louvain_communities

from black_hole_sparsification import (
    apply_black_hole_sparsification,
    black_hole_strategy_per_community,
    calculate_gravity_per_community,
    prune_edges,
    save_sparsified_graph,
)

CATEGORIES_10 = pd.Series(
    ["nonporous", "small pore", "medium pore", "large pore"] * 2 + ["nonporous", "small pore"]
)  # 10 entries, positionally indexed 0..9 to match tiny_graph's node ids


class TestCalculateGravityPerCommunity:
    def test_every_community_node_gets_a_gravity_score(self, tiny_graph):
        communities = louvain_communities(tiny_graph, seed=42)
        gravity, *_ = calculate_gravity_per_community(tiny_graph, communities, (0.33, 0.33, 0.33))
        all_community_nodes = {n for c in communities for n in c}
        assert set(gravity.keys()) == all_community_nodes

    def test_gravity_in_unit_range_when_weights_sum_to_one(self, tiny_graph):
        communities = louvain_communities(tiny_graph, seed=42)
        gravity, *_ = calculate_gravity_per_community(tiny_graph, communities, (0.33, 0.33, 0.34))
        for value in gravity.values():
            assert -1e-6 <= value <= 1.0 + 1e-6

    def test_does_not_mutate_input_graph(self, tiny_graph):
        n_before, e_before = tiny_graph.number_of_nodes(), tiny_graph.number_of_edges()
        communities = louvain_communities(tiny_graph, seed=42)
        calculate_gravity_per_community(tiny_graph, communities, (0.33, 0.33, 0.33))
        assert tiny_graph.number_of_nodes() == n_before
        assert tiny_graph.number_of_edges() == e_before

    def test_fixed_test_nodes_are_a_subset_of_graph_nodes(self, tiny_graph):
        communities = louvain_communities(tiny_graph, seed=42)
        *_, fixed_test_nodes = calculate_gravity_per_community(tiny_graph, communities, (0.33, 0.33, 0.33))
        assert fixed_test_nodes.issubset(set(tiny_graph.nodes()))

    def test_deterministic_given_same_inputs(self, tiny_graph):
        communities = louvain_communities(tiny_graph, seed=42)
        result1 = calculate_gravity_per_community(tiny_graph, communities, (0.3, 0.3, 0.4))
        result2 = calculate_gravity_per_community(tiny_graph, communities, (0.3, 0.3, 0.4))
        assert result1[0] == result2[0]  # gravity dict
        assert result1[4] == result2[4]  # fixed_test_nodes set

    def test_no_eligible_high_degree_node_falls_back_without_crashing(self):
        """Every node has degree <= 2 -- calculate_gravity_per_community must fall back
        to the highest-degree node instead of crashing on an empty eligible list."""
        g = nx.Graph()
        g.add_edge(0, 1, weight=0.5)
        g.add_edge(1, 2, weight=0.5)
        communities = [{0, 1, 2}]
        gravity, _, _, _, fixed_test_nodes = calculate_gravity_per_community(g, communities, (0.33, 0.33, 0.33))
        assert len(fixed_test_nodes) >= 1

    def test_centrality_is_scoped_to_the_community_not_the_whole_graph(self, tiny_graph):
        """Found via mutation testing: mutating `graph.subgraph(community)` to
        `graph.subgraph(None)` (= the whole graph) survived the existing suite, because
        no test distinguished per-community centrality from whole-graph centrality.
        tiny_graph's two 5-cycles are only joined by one bridge edge, so degree
        centrality computed within each 5-node community must differ from degree
        centrality computed over all 10 nodes."""
        communities = louvain_communities(tiny_graph, seed=42)
        assert len(communities) >= 2, "test assumes tiny_graph splits into >1 community"
        gravity, degree_c, *_ = calculate_gravity_per_community(tiny_graph, communities, (1.0, 0.0, 0.0))

        whole_graph_degree_c = nx.degree_centrality(tiny_graph)
        assert degree_c != whole_graph_degree_c

    def test_weights_change_relative_ranking(self):
        """A node with high degree but low edge-weight-sum should rank differently
        depending on which factor is weighted more heavily."""
        g = nx.Graph()
        g.add_edge(0, 1, weight=0.01)
        g.add_edge(0, 2, weight=0.01)
        g.add_edge(0, 3, weight=0.01)  # node 0: high degree, low weight sum
        g.add_edge(4, 5, weight=0.99)  # node 4: low degree, high weight sum
        communities = [{0, 1, 2, 3, 4, 5}]

        gravity_degree_heavy, *_ = calculate_gravity_per_community(g, communities, (1.0, 0.0, 0.0))
        gravity_weight_heavy, *_ = calculate_gravity_per_community(g, communities, (0.0, 0.0, 1.0))
        assert gravity_degree_heavy[0] > gravity_degree_heavy[4]
        assert gravity_weight_heavy[4] >= gravity_weight_heavy[0]


class TestBlackHoleStrategyPerCommunity:
    def test_fixed_test_nodes_always_retained(self, tiny_graph):
        communities = louvain_communities(tiny_graph, seed=42)
        gravity, *_ = calculate_gravity_per_community(tiny_graph, communities, (0.33, 0.33, 0.33))
        fixed_test_nodes = {0, 5}  # force specific nodes as "fixed", regardless of gravity
        pruned, _ = black_hole_strategy_per_community(
            tiny_graph, gravity, communities, threshold=0.9, fixed_test_nodes=fixed_test_nodes, pld_categories=CATEGORIES_10,
        )
        assert fixed_test_nodes.issubset(set(pruned.nodes()))

    def test_does_not_mutate_input_graph(self, tiny_graph):
        """Deliberate deviation from the reference code, which mutates in place —
        critical here since the caller reuses one cached graph across many calls."""
        n_before = tiny_graph.number_of_nodes()
        communities = louvain_communities(tiny_graph, seed=42)
        gravity, *_ = calculate_gravity_per_community(tiny_graph, communities, (0.33, 0.33, 0.33))
        black_hole_strategy_per_community(tiny_graph, gravity, communities, 0.5, set(), CATEGORIES_10)
        assert tiny_graph.number_of_nodes() == n_before

    def test_two_calls_on_same_graph_are_independent(self, tiny_graph):
        """The exact scenario the graph.copy() fix targets: calling this twice with
        different thresholds on the SAME graph object must not have the second call
        see effects of the first."""
        communities = louvain_communities(tiny_graph, seed=42)
        gravity, *_ = calculate_gravity_per_community(tiny_graph, communities, (0.33, 0.33, 0.33))
        pruned_a, n_a = black_hole_strategy_per_community(tiny_graph, gravity, communities, 0.8, set(), CATEGORIES_10)
        pruned_b, n_b = black_hole_strategy_per_community(tiny_graph, gravity, communities, 0.1, set(), CATEGORIES_10)
        # A more aggressive threshold (0.8) must retain no more nodes than a gentler one (0.1)
        assert n_a <= n_b

    def test_threshold_zero_retains_all_nodes(self, tiny_graph):
        communities = louvain_communities(tiny_graph, seed=42)
        gravity, *_ = calculate_gravity_per_community(tiny_graph, communities, (0.33, 0.33, 0.33))
        pruned, n = black_hole_strategy_per_community(tiny_graph, gravity, communities, 0.0, set(), CATEGORIES_10)
        assert n == tiny_graph.number_of_nodes()

    def test_retains_at_least_twenty_percent_floor(self, tiny_graph):
        """apply_black_hole_sparsification's own target formula: max(0.2*N, (1-tau)*N) —
        even at the most aggressive threshold (0.9), at least 20% of nodes survive."""
        communities = louvain_communities(tiny_graph, seed=42)
        gravity, *_ = calculate_gravity_per_community(tiny_graph, communities, (0.33, 0.33, 0.33))
        _, n = black_hole_strategy_per_community(tiny_graph, gravity, communities, 0.9, set(), CATEGORIES_10)
        assert n >= 0.2 * tiny_graph.number_of_nodes() - 1  # -1 for integer rounding slack

    def test_twenty_percent_floor_scales_with_graph_size(self):
        """Found via mutation testing: `0.2 * n_total` mutated to `0.2 / n_total` in the
        target-node formula survived on tiny_graph (n=10), because at that scale both the
        correct floor (2) and the buggy one (~0) get overshadowed by other floors elsewhere
        in the function. On a 100-node graph the two formulas diverge sharply (floor of 20
        vs. ~0), making the bug's effect on retained node count large enough to assert on."""
        n = 100
        g = nx.Graph()
        for i in range(n):
            g.add_edge(i, (i + 1) % n, weight=0.5)
        communities = louvain_communities(g, seed=42)
        gravity, *_ = calculate_gravity_per_community(g, communities, (0.33, 0.33, 0.33))
        _, retained = black_hole_strategy_per_community(g, gravity, communities, 0.9, set(), pd.Series(["nonporous"] * n))
        assert retained >= 0.2 * n - 2  # correct formula: max(0.2*100, 0.1*100) = 20


class TestPruneEdges:
    def test_fixed_test_node_keeps_at_least_one_edge(self, tiny_graph):
        pruned = prune_edges(tiny_graph, edge_threshold=0.99, fixed_test_nodes={0})
        assert pruned.degree(0) >= 1

    def test_never_introduces_new_edges(self, tiny_graph):
        original_edges = {frozenset(e) for e in tiny_graph.edges()}
        pruned = prune_edges(tiny_graph, edge_threshold=0.5, fixed_test_nodes={0})
        pruned_edges = {frozenset(e) for e in pruned.edges()}
        assert pruned_edges.issubset(original_edges)

    def test_returns_new_graph_object(self, tiny_graph):
        pruned = prune_edges(tiny_graph, edge_threshold=0.5, fixed_test_nodes=set())
        assert pruned is not tiny_graph

    def test_preserves_full_node_set_even_if_isolated(self, tiny_graph):
        pruned = prune_edges(tiny_graph, edge_threshold=0.99, fixed_test_nodes=set())
        assert set(pruned.nodes()) == set(tiny_graph.nodes())

    def test_empty_graph_returns_unchanged(self):
        g = nx.Graph()
        g.add_nodes_from(range(5))
        result = prune_edges(g, edge_threshold=0.5, fixed_test_nodes=set())
        assert result is g  # explicit early-return path

    def test_threshold_zero_keeps_almost_all_edges(self, tiny_graph):
        pruned = prune_edges(tiny_graph, edge_threshold=0.0, fixed_test_nodes=set())
        assert pruned.number_of_edges() == tiny_graph.number_of_edges()

    def test_num_to_keep_subtracts_reserved_test_edges_from_the_budget(self):
        """Found via mutation testing: the `- len(test_edges_to_keep)` in the num_to_keep
        formula mutated to `+ len(test_edges_to_keep)` survived — every existing test used
        a small edge_threshold where the non_test_edges pool was smaller than either
        formula's result, so both produced the same final edge count. A 30-node ring
        (30 edges, distinct weights) with a large threshold makes the correct budget (9
        extra edges) and the buggy one (11) both fit within the available pool, so they
        produce different, directly comparable totals."""
        n = 30
        g = nx.Graph()
        for i in range(n):
            g.add_edge(i, (i + 1) % n, weight=1.0 - i / (2 * n))  # strictly decreasing weights
        pruned = prune_edges(g, edge_threshold=0.9, fixed_test_nodes={0})
        # 1 reserved test edge (node 0's single best edge) + correct budget of 9 = 10 total.
        assert pruned.number_of_edges() == 10


class TestApplyBlackHoleSparsification:
    @pytest.fixture
    def graph_and_df(self, tiny_graph):
        df = pd.DataFrame({"pld_category": CATEGORIES_10})
        return tiny_graph, df

    def test_metrics_has_expected_keys(self, graph_and_df):
        graph, df = graph_and_df
        result = apply_black_hole_sparsification(graph, df, (0.33, 0.33, 0.33), 0.3)
        expected_keys = {
            "nodes_before", "edges_before", "nodes_after", "edges_after",
            "node_retention_pct", "edge_retention_pct", "isolated_nodes_after",
            "density_before", "density_after", "num_communities",
            "num_fixed_test_nodes", "peak_memory_mb", "elapsed_seconds",
        }
        assert expected_keys.issubset(result["metrics"].keys())

    def test_never_increases_node_or_edge_count(self, graph_and_df):
        graph, df = graph_and_df
        result = apply_black_hole_sparsification(graph, df, (0.33, 0.33, 0.33), 0.5)
        m = result["metrics"]
        assert m["nodes_after"] <= m["nodes_before"]
        assert m["edges_after"] <= m["edges_before"]

    def test_retention_percentages_in_valid_range(self, graph_and_df):
        graph, df = graph_and_df
        result = apply_black_hole_sparsification(graph, df, (0.33, 0.33, 0.33), 0.3)
        m = result["metrics"]
        assert 0.0 <= m["node_retention_pct"] <= 100.0
        assert 0.0 <= m["edge_retention_pct"] <= 100.0

    def test_original_graph_untouched_by_repeated_calls(self, graph_and_df):
        """End-to-end regression test for the mutation bug: calling this twice on the
        SAME graph with different thresholds must not corrupt results."""
        graph, df = graph_and_df
        n_before = graph.number_of_nodes()
        apply_black_hole_sparsification(graph, df, (0.33, 0.33, 0.33), 0.3)
        assert graph.number_of_nodes() == n_before
        result_b = apply_black_hole_sparsification(graph, df, (0.33, 0.33, 0.33), 0.5)
        assert graph.number_of_nodes() == n_before
        assert result_b["metrics"]["nodes_before"] == n_before

    def test_fixed_test_nodes_present_in_pruned_graph(self, graph_and_df):
        graph, df = graph_and_df
        result = apply_black_hole_sparsification(graph, df, (0.33, 0.33, 0.33), 0.5)
        assert result["fixed_test_nodes"].issubset(set(result["graph"].nodes()))

    def test_deterministic_given_same_graph_and_weights(self, graph_and_df):
        """Found via mutation testing: mutating louvain_communities' `seed=42` to
        `seed=None` survived — nothing at this level asserted run-to-run determinism,
        only calculate_gravity_per_community's determinism given already-fixed
        communities. Community detection itself must also be seeded."""
        graph, df = graph_and_df
        result_a = apply_black_hole_sparsification(graph, df, (0.33, 0.33, 0.33), 0.3)
        result_b = apply_black_hole_sparsification(graph, df, (0.33, 0.33, 0.33), 0.3)
        assert result_a["fixed_test_nodes"] == result_b["fixed_test_nodes"]
        assert result_a["metrics"] == result_b["metrics"]

    def test_peak_memory_mb_is_a_rounded_float(self, graph_and_df):
        """Found via mutation testing: `round(peak_memory, 2)` mutated to
        `round(peak_memory, None)` survived — Python's round() with ndigits=None returns
        an int, silently changing the metric's type. Nothing checked the type/precision."""
        graph, df = graph_and_df
        result = apply_black_hole_sparsification(graph, df, (0.33, 0.33, 0.33), 0.3)
        peak = result["metrics"]["peak_memory_mb"]
        assert isinstance(peak, float)


class TestSaveSparsifiedGraph:
    def test_writes_expected_files(self, tiny_graph, tmp_path):
        df = pd.DataFrame({"pld_category": CATEGORIES_10})
        result = apply_black_hole_sparsification(tiny_graph, df, (0.33, 0.33, 0.33), 0.3)
        refcodes = [f"MOF{i:04d}" for i in range(10)]
        save_sparsified_graph(result, refcodes, "unittest", 0.3, output_dir=str(tmp_path))
        assert (tmp_path / "bh_graph_unittest_tau0.30.csv").exists()
        assert (tmp_path / "bh_metrics_unittest_tau0.30.json").exists()

    def test_saved_csv_and_json_contents_match_the_result(self, tiny_graph, tmp_path):
        """Found via mutation testing: many mutations inside save_sparsified_graph (column
        order, edge source/target, json.dump kwargs) survived because the previous test
        only checked that files exist, never their actual contents."""
        import json

        df = pd.DataFrame({"pld_category": CATEGORIES_10})
        result = apply_black_hole_sparsification(tiny_graph, df, (0.33, 0.33, 0.33), 0.3)
        refcodes = np.array([f"MOF{i:04d}" for i in range(10)])
        save_sparsified_graph(result, refcodes, "unittest", 0.3, output_dir=str(tmp_path))

        saved_edges = pd.read_csv(tmp_path / "bh_graph_unittest_tau0.30.csv")
        assert list(saved_edges.columns) == ["source", "target", "weight"]
        assert len(saved_edges) == result["graph"].number_of_edges()
        assert set(saved_edges["source"]) | set(saved_edges["target"]) <= set(refcodes)

        with open(tmp_path / "bh_metrics_unittest_tau0.30.json") as f:
            saved_metrics = json.load(f)
        assert saved_metrics == result["metrics"]
