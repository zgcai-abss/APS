"""Small deterministic unit tests, not paper experiments or model evaluations."""

import itertools
import unittest
from dataclasses import replace

import numpy as np

from aps_framework import APSConfig, APSRunner, CallableOracle, OracleResponse, StratificationResult
from aps_framework.audit import (
    AuditSelection, compute_audit_result, residual_scores_from_diagnostics, select_shadow_audits,
)
from aps_framework.graph import neighbor_summaries, validate_neighbors
from aps_framework.oracle import parse_decision
from aps_framework.response_surface import local_response_slope, propagate_from_prototypes
from aps_framework.schedules import (
    core_prototype_rate, core_strata_count, nominal_core_budget,
    residual_aware_allocation, shadow_audit_budget, singleton_tail_count,
)
from aps_framework.stratification import (
    MiniBatchKMeansStratifier, select_farthest_first, select_singleton_tail,
    validate_partition,
)


def configuration(**changes):
    # Test fixtures exercise code paths; they are not recommended run settings.
    config = APSConfig(
        seed=1, n_base=8, alpha_base=0.5, lambda_decay=0.5, core_strata_base=2,
        eta=0.5, tail_base_fraction=0.125, zeta=0.5, audit_gamma=0.5,
        audit_beta=0.5, audit_min=1, support_size=2, distance_epsilon=1e-6,
        tau=1e-6, mad_floor=1e-6, lambda_rho=1.0, lambda_e=1.0,
        lambda_r=1.0, rare_state_threshold=0.2, kl_floor=1e-12,
        initial_residual_score=1.0, kmeans_batch_size=8, kmeans_n_init=1,
        max_farthest_candidates=2, audit_design="uniform",
    )
    return replace(config, **changes)


class FixedPartition:
    def partition(self, features, config):
        labels = np.array([0, 0, 0, 0, 1, 1, 1, -1])
        return StratificationResult(labels, np.array([7]), {0: np.arange(4), 1: np.arange(4, 7)})


def run_inputs():
    return dict(
        profiles=[{"id": i} for i in range(8)],
        features=np.arange(8, dtype=float).reshape(-1, 1),
        initial_states=["left"] * 8,
        graph_neighbors=[[(i + 1) % 8] for i in range(8)],
        events=["step_1", "step_2"], options=["left", "right"],
    )


class AllocationTests(unittest.TestCase):
    def test_clipped_budget_is_redistributed(self):
        counts = residual_aware_allocation({0: np.arange(1), 1: np.arange(100)}, 10,
                                           {0: 1e8, 1: 0.0}, 1e-6, 1.0)
        self.assertEqual(counts, {0: 1, 1: 9})

    def test_capacity_and_budget_properties(self):
        rng = np.random.default_rng(4)
        for _ in range(150):
            sizes = rng.integers(0, 20, size=8)
            strata = {i: np.arange(n) for i, n in enumerate(sizes)}
            budget = int(rng.integers(0, 200))
            counts = residual_aware_allocation(strata, budget,
                {i: float(10 ** rng.uniform(-6, 6)) for i in strata}, 1e-6, 1.0)
            self.assertEqual(sum(counts.values()), min(budget, int(sizes.sum())))
            self.assertTrue(all(0 <= counts[i] <= sizes[i] for i in strata))
            if budget >= np.count_nonzero(sizes):
                self.assertTrue(all(counts[i] >= 1 for i in strata if sizes[i]))

    def test_empty_strata(self):
        self.assertEqual(residual_aware_allocation({}, 2, {}, 1e-6, 1.0), {})

    def test_zero_budget(self):
        self.assertEqual(residual_aware_allocation({1: np.arange(4)}, 0, {}, 1e-6, 1.0), {1: 0})

    def test_higher_residual_increases_allocation(self):
        strata = {0: np.arange(20), 1: np.arange(20)}
        equal = residual_aware_allocation(strata, 10, {0: 1.0, 1: 1.0}, 1e-6, 1.0)
        unequal = residual_aware_allocation(strata, 10, {0: 9.0, 1: 1.0}, 1e-6, 1.0)
        self.assertEqual(equal, {0: 5, 1: 5})
        self.assertGreater(unequal[0], equal[0])
        self.assertEqual(sum(unequal.values()), 10)

    def test_invalid_budget_and_scores(self):
        for budget in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                residual_aware_allocation({0: np.arange(3)}, budget, {}, 1e-6, 1.0)
        with self.assertRaises(ValueError):
            residual_aware_allocation({0: np.arange(3)}, 1, {0: float("nan")}, 1e-6, 1.0)

    def test_schedule_growth(self):
        cfg = configuration()
        self.assertEqual(core_strata_count(8, cfg), 2)
        self.assertEqual(core_strata_count(32, cfg), 4)
        self.assertEqual(singleton_tail_count(8, cfg), 1)
        self.assertEqual(singleton_tail_count(32, cfg), 2)
        self.assertEqual(core_prototype_rate(32, cfg), 0.25)
        self.assertEqual(nominal_core_budget(32, cfg), 8)
        self.assertEqual(shadow_audit_budget(32, cfg), 8)


class SelectionTests(unittest.TestCase):
    def test_candidate_limit_never_duplicates(self):
        chosen = select_farthest_first(np.arange(10).reshape(-1, 1), np.arange(10),
                                       5, np.random.default_rng(0), 3)
        self.assertEqual(len(chosen), 5)
        self.assertEqual(len(np.unique(chosen)), 5)

    def test_identical_features(self):
        chosen = select_farthest_first(np.zeros((8, 2)), np.arange(8), 6, np.random.default_rng(0), 2)
        self.assertEqual(len(np.unique(chosen)), 6)

    def test_selection_is_repeatable(self):
        values = np.arange(12).reshape(-1, 1)
        a = select_farthest_first(values, np.arange(12), 3, np.random.default_rng(9), 7)
        b = select_farthest_first(values, np.arange(12), 3, np.random.default_rng(9), 7)
        np.testing.assert_array_equal(a, b)

    def test_selection_all_and_none(self):
        values = np.arange(4).reshape(-1, 1)
        np.testing.assert_array_equal(select_farthest_first(values, np.arange(4), 9,
            np.random.default_rng(1), 1), np.arange(4))
        self.assertEqual(len(select_farthest_first(values, np.arange(4), 0,
            np.random.default_rng(1), 1)), 0)

    def test_duplicate_candidates_are_rejected(self):
        with self.assertRaises(ValueError):
            select_farthest_first(np.zeros((2, 1)), np.array([0, 0]), 1, np.random.default_rng(0), 2)

    def test_tail_selection(self):
        np.testing.assert_array_equal(select_singleton_tail(np.array([[0.], [0.], [100.]]), 1, 1e-6), [2])

    def test_kmeans_partition(self):
        data = np.array([[0.], [0.1], [0.2], [0.3], [10.], [10.1], [10.2], [30.]])
        partition = MiniBatchKMeansStratifier().partition(data, configuration())
        np.testing.assert_array_equal(partition.singleton_tail_indices, [7])
        combined = np.concatenate([partition.singleton_tail_indices, *partition.stratum_indices.values()])
        np.testing.assert_array_equal(np.sort(combined), np.arange(8))

    def test_custom_partition_accepts_empty_members(self):
        validate_partition(StratificationResult(np.array([0, 0]), np.array([]),
            {0: np.array([0, 1]), 1: np.array([])}), 2)

    def test_partition_rejects_overlap_and_mislabeled_members(self):
        for result in [
            StratificationResult(np.array([0, -1]), np.array([1]), {0: np.array([0, 1])}),
            StratificationResult(np.array([0, 1]), np.array([], dtype=int), {0: np.array([0, 1])}),
        ]:
            with self.assertRaises(ValueError):
                validate_partition(result, 2)


class AuditTests(unittest.TestCase):
    def test_ht_correction_matches_population_in_expectation(self):
        soft = np.array([[1., 0.], [0.8, 0.2], [0.6, 0.4], [0.3, 0.7]])
        truth = np.array([0, 1, 0, 1])
        estimates = []
        for pair in itertools.combinations([1, 2, 3], 2):
            selection = AuditSelection({0: np.array(pair)}, {i: 2 / 3 for i in pair})
            result = compute_audit_result(soft_ledger=soft, hard_states=soft.argmax(axis=1),
                correction_frames={0: np.array([1, 2, 3])}, audit_selection=selection,
                shadow_labels={i: int(truth[i]) for i in pair}, support_distances={},
                support_slopes={}, config=configuration())
            estimates.append(result.unprojected_distribution)
        np.testing.assert_allclose(np.mean(estimates, axis=0), [0.5, 0.5])

    def test_uniform_inclusion_probabilities(self):
        selection = select_shadow_audits({0: np.arange(6)}, {0: 2}, np.zeros(6, dtype=int),
            2, configuration(), np.random.default_rng(0))
        self.assertEqual(len(selection.all_indices), 2)
        self.assertTrue(all(p == 1 / 3 for p in selection.inclusion_probabilities.values()))

    def test_state_stratified_count_and_probabilities(self):
        states = np.array([0, 1, 1, 1, 1, 1])
        selection = select_shadow_audits({0: np.arange(6)}, {0: 4}, states,
            2, configuration(audit_design="state_stratified"), np.random.default_rng(2))
        self.assertEqual(len(selection.all_indices), 4)
        self.assertEqual(selection.inclusion_probabilities[0], 1.0)
        self.assertTrue(all(p == 0.6 for i, p in selection.inclusion_probabilities.items() if i != 0))

    def test_state_design_falls_back_to_uniform(self):
        selection = select_shadow_audits({0: np.arange(3)}, {0: 1}, np.arange(3),
            3, configuration(audit_design="state_stratified"), np.random.default_rng(1))
        self.assertEqual(list(selection.inclusion_probabilities.values()), [1 / 3])

    def test_full_state_stratified_audit(self):
        states = np.array([0, 1, 1, 1, 1, 1])
        selection = select_shadow_audits({0: np.arange(6)}, {0: 6}, states,
            2, configuration(audit_design="state_stratified"), np.random.default_rng(2))
        np.testing.assert_array_equal(selection.all_indices, np.arange(6))
        self.assertTrue(all(p == 1.0 for p in selection.inclusion_probabilities.values()))

    def test_zero_probability_stratum_rejected(self):
        with self.assertRaises(ValueError):
            select_shadow_audits({0: np.arange(3)}, {0: 0}, np.zeros(3, dtype=int),
                2, configuration(), np.random.default_rng(1))

    def test_invalid_inclusion_probability_rejected(self):
        with self.assertRaises(ValueError):
            compute_audit_result(soft_ledger=np.array([[1., 0.]]), hard_states=np.array([0]),
                correction_frames={0: np.array([0])}, audit_selection=AuditSelection({0: np.array([0])}, {0: 0.0}),
                shadow_labels={0: 1}, support_distances={}, support_slopes={}, config=configuration())

    def test_single_audit_does_not_invent_sample_variance(self):
        result = compute_audit_result(
            soft_ledger=np.array([[1., 0.], [1., 0.]]), hard_states=np.array([0, 0]),
            correction_frames={0: np.array([1])},
            audit_selection=AuditSelection({0: np.array([1])}, {1: 1.0}),
            shadow_labels={1: 1}, support_distances={1: 1.0},
            support_slopes={1: None}, config=configuration())
        self.assertIsNone(result.diagnostics[0]["residual_variance"])
        self.assertIsNone(result.diagnostics[0]["local_disagreement_slope"])
        self.assertEqual(result.diagnostics[0]["slope_count"], 0.0)
        self.assertTrue(np.isfinite(result.residual_scores[0]))
        self.assertGreater(result.residual_scores[0], 0.0)

    def test_residual_variance_is_sum_of_category_sample_variances(self):
        result = compute_audit_result(
            soft_ledger=np.array([[1., 0.], [1., 0.]]), hard_states=np.array([0, 0]),
            correction_frames={0: np.array([0, 1])},
            audit_selection=AuditSelection({0: np.array([0, 1])}, {0: 1.0, 1: 1.0}),
            shadow_labels={0: 0, 1: 1}, support_distances={},
            support_slopes={0: 2.0, 1: None}, config=configuration())
        self.assertEqual(result.diagnostics[0]["residual_variance"], 1.0)
        self.assertEqual(result.diagnostics[0]["local_disagreement_slope"], 2.0)
        self.assertEqual(result.diagnostics[0]["slope_count"], 1.0)

    def test_risk_score_matches_normalized_equation(self):
        scores = residual_scores_from_diagnostics({
            0: dict(residual_variance=0.5, avg_support_distance=1.0,
                    local_disagreement_slope=2.0, mismatch_rate=0.5, rare_state_recall=0.8),
            1: dict(residual_variance=1.0, avg_support_distance=2.0,
                    local_disagreement_slope=4.0, mismatch_rate=0.25, rare_state_recall=0.9),
        }, configuration())
        self.assertAlmostEqual(scores[0], 0.8525)
        self.assertAlmostEqual(scores[1], 2.0725)


class EngineTests(unittest.TestCase):
    def test_query_metadata_is_retained_separately_by_role(self):
        payload = {"retry_count": 1, "repair": {"applied": False}}
        runner = APSRunner(config=configuration(),
            oracle=CallableOracle(lambda q: OracleResponse("left", metadata=payload)),
            stratifier=FixedPartition())
        result = runner.run(**run_inputs())
        for record in result.rounds:
            for role, ids in [("core_prototype", record.prototype_indices),
                              ("singleton_tail", record.singleton_tail_indices),
                              ("shadow_audit", record.shadow_audit_indices)]:
                self.assertEqual(set(record.query_metadata[role]), set(ids))
                self.assertTrue(all(item == payload for item in record.query_metadata[role].values()))
        payload["repair"]["applied"] = True
        first_role = result.rounds[0].query_metadata["core_prototype"]
        self.assertTrue(all(not item["repair"]["applied"] for item in first_role.values()))

    def test_absent_metadata_is_not_fabricated(self):
        result = APSRunner(config=configuration(), oracle=CallableOracle(lambda q: "left"),
            stratifier=FixedPartition()).run(**run_inputs())
        self.assertEqual(result.rounds[0].query_metadata,
                         {"core_prototype": {}, "singleton_tail": {}, "shadow_audit": {}})

    def test_invalid_metadata_is_rejected(self):
        runner = APSRunner(config=configuration(),
            oracle=CallableOracle(lambda q: OracleResponse("left", metadata="invalid")),
            stratifier=FixedPartition())
        with self.assertRaisesRegex(ValueError, "metadata"):
            runner.run(**run_inputs())

    def test_rounds_receive_previous_rollout_state(self):
        queries = []
        def decide(query):
            queries.append(query)
            return "right"
        APSRunner(config=configuration(), oracle=CallableOracle(decide),
            stratifier=FixedPartition()).run(**run_inputs())
        first = [q for q in queries if q.round_index == 1]
        second = [q for q in queries if q.round_index == 2]
        self.assertTrue(first and second)
        self.assertTrue(all(q.previous_state == "left" for q in first))
        self.assertTrue(all(q.previous_state == "right" for q in second))
        self.assertTrue(all("right: 1/1" in q.neighbor_summary for q in second))

    def test_tail_agents_are_never_prototype_supports(self):
        result = APSRunner(config=configuration(), oracle=CallableOracle(lambda q: q.agent_id % 2),
            stratifier=FixedPartition()).run(**run_inputs())
        for record in result.rounds:
            for support in record.support_indices.values():
                self.assertTrue(set(support) <= set(record.prototype_indices))
                self.assertFalse(set(support) & set(record.singleton_tail_indices))

    def test_repeatable_runs_preserve_caller_inputs(self):
        inputs = run_inputs()
        original_features = inputs["features"].copy()
        original_states = list(inputs["initial_states"])
        runner = APSRunner(config=configuration(), oracle=CallableOracle(lambda q: q.agent_id % 2),
            stratifier=FixedPartition())
        first, second = runner.run(**inputs), runner.run(**inputs)
        for a, b in zip(first.rounds, second.rounds):
            np.testing.assert_array_equal(a.prototype_indices, b.prototype_indices)
            np.testing.assert_array_equal(a.shadow_audit_indices, b.shadow_audit_indices)
            np.testing.assert_allclose(a.projected_distribution, b.projected_distribution)
        np.testing.assert_array_equal(inputs["features"], original_features)
        self.assertEqual(inputs["initial_states"], original_states)

    def test_audits_do_not_overwrite_recurrent_states(self):
        queries = []
        def decide(query):
            queries.append(query)
            return "right" if query.role == "shadow_audit" else "left"
        result = APSRunner(config=configuration(), oracle=CallableOracle(decide), stratifier=FixedPartition()).run(**run_inputs())
        self.assertEqual(len(result.rounds), 2)
        for record in result.rounds:
            np.testing.assert_array_equal(record.hard_states, np.zeros(8, dtype=int))
            self.assertGreater(record.projected_distribution[1], 0)
            self.assertEqual(record.call_record.total, 8)
            self.assertEqual(sum(record.prototype_budgets.values()), 4)
            self.assertEqual(sum(record.audit_budgets.values()), 3)
            self.assertFalse(set(record.prototype_indices) & set(record.singleton_tail_indices))
            self.assertFalse(set(record.shadow_audit_indices) & set(record.prototype_indices))
            self.assertFalse(set(record.shadow_audit_indices) & set(record.singleton_tail_indices))
            self.assertEqual(set(record.shadow_labels), set(record.audit_inclusion_probabilities))
        self.assertEqual(result.total_online_calls, len(queries))
        self.assertTrue(all(q.previous_state == "left" for q in queries))
        self.assertTrue(all("left: 1/1" in q.neighbor_summary for q in queries))

    def test_core_budget_validation_precedes_oracle_calls(self):
        calls = []
        runner = APSRunner(config=configuration(alpha_base=0.01),
            oracle=CallableOracle(lambda q: calls.append(q)), stratifier=FixedPartition())
        with self.assertRaisesRegex(ValueError, "core budget"):
            runner.run(**run_inputs())
        self.assertEqual(calls, [])

    def test_audit_budget_validation_precedes_oracle_calls(self):
        calls = []
        runner = APSRunner(config=configuration(audit_gamma=0.0),
            oracle=CallableOracle(lambda q: calls.append(q)), stratifier=FixedPartition())
        with self.assertRaisesRegex(ValueError, "audit budget"):
            runner.run(**run_inputs())
        self.assertEqual(calls, [])

    def test_all_agents_directly_queried(self):
        runner = APSRunner(config=configuration(alpha_base=1.0, audit_gamma=0.0, audit_min=0),
            oracle=CallableOracle(lambda q: q.agent_id % 2), stratifier=FixedPartition())
        result = runner.run(**run_inputs())
        for record in result.rounds:
            self.assertEqual(record.call_record.total, 8)
            self.assertEqual(record.call_record.shadow_audit_calls, 0)
            np.testing.assert_allclose(record.projected_distribution, [0.5, 0.5])

    def test_all_tail_agents(self):
        runner = APSRunner(config=configuration(tail_base_fraction=1.0), oracle=CallableOracle(lambda q: "right"))
        result = runner.run(**run_inputs())
        for record in result.rounds:
            self.assertEqual(record.call_record.core_prototype_calls, 0)
            self.assertEqual(record.call_record.singleton_tail_calls, 8)
            np.testing.assert_allclose(record.projected_distribution, [0.0, 1.0])

    def test_invalid_response_is_not_silently_replaced(self):
        runner = APSRunner(config=configuration(), oracle=CallableOracle(lambda q: "unknown"), stratifier=FixedPartition())
        with self.assertRaises(RuntimeError):
            runner.run(**run_inputs())

    def test_response_count_checked(self):
        class IncompleteOracle:
            def batch_decide(self, queries):
                return []
        with self.assertRaises(RuntimeError):
            APSRunner(config=configuration(), oracle=IncompleteOracle(), stratifier=FixedPartition()).run(**run_inputs())

    def test_custom_partition_must_cover_every_agent(self):
        class InvalidPartition:
            def partition(self, features, config):
                return StratificationResult(np.zeros(8, dtype=int), np.array([], dtype=int), {0: np.array([0, 0])})
        with self.assertRaises(ValueError):
            APSRunner(config=configuration(), oracle=CallableOracle(lambda q: 0), stratifier=InvalidPartition()).run(**run_inputs())


class InputTests(unittest.TestCase):
    def test_local_slope_scales_with_feature_distance(self):
        features = np.array([[0.], [2.]])
        self.assertEqual(local_response_slope(features, np.eye(2), 1e-6), 1.0)
        self.assertEqual(local_response_slope(features * 10, np.eye(2), 1e-6), 0.1)

    def test_local_slope_handles_agreement_and_duplicate_features(self):
        self.assertEqual(local_response_slope(np.array([[0.], [2.]]),
            np.array([[1., 0.], [1., 0.]]), 1e-6), 0.0)
        self.assertEqual(local_response_slope(np.zeros((2, 1)), np.eye(2), 1e-6), 2e6)

    def test_single_support_has_no_slope_estimate(self):
        self.assertIsNone(local_response_slope(np.array([[0.]]), np.array([[1., 0.]]), 1e-6))

    def test_invalid_run_inputs_fail_before_queries(self):
        for changes in [dict(features=np.full((8, 1), np.nan)),
                        dict(options=["left", "left"]), dict(events=[]),
                        dict(initial_states=["unknown"] * 8), dict(initial_states=[])]:
            calls = []
            runner = APSRunner(config=configuration(), oracle=CallableOracle(lambda q: calls.append(q)),
                stratifier=FixedPartition())
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                runner.run(**(run_inputs() | changes))
            self.assertEqual(calls, [])

    def test_invalid_config(self):
        for changes in [dict(seed=-1), dict(support_size=0), dict(alpha_base=2),
                        dict(eta=1.0), dict(tau=float("nan")), dict(lambda_e=-1), dict(kmeans_n_init=0)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                configuration(**changes)

    def test_decision_index_contract(self):
        self.assertEqual(parse_decision(0, ["left", "right"]), 0)
        self.assertEqual(parse_decision("right", ["left", "right"]), 1)
        for invalid in [True, 1.0, -1, 2, "1", "LEFT"]:
            self.assertIsNone(parse_decision(invalid, ["left", "right"]))

    def test_graph_checks(self):
        for neighbors in [[[2], [0]], [[1, 1], [0]], [[1.5], [0]], [[]]]:
            with self.assertRaises(ValueError):
                validate_neighbors(neighbors, 2)

    def test_empty_neighbors(self):
        neighbors = validate_neighbors([[], []], 2)
        self.assertEqual(neighbor_summaries(neighbors, [0, 1], ["left", "right"]),
                         ["No neighbor-state information is available."] * 2)

    def test_weighted_propagation(self):
        result = propagate_from_prototypes(np.array([[0.], [2.], [1.]]), np.array([0, 1]),
            np.eye(2), np.array([2]), 2, 1e-6)
        np.testing.assert_allclose(result.soft_vectors, [[0.5, 0.5]])
        np.testing.assert_array_equal(result.hard_states, [0])


if __name__ == "__main__":
    unittest.main()
