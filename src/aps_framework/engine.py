from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from .audit import (
    allocate_shadow_audits,
    compute_audit_result,
    select_shadow_audits,
)
from .config import APSConfig
from .graph import neighbor_summaries, validate_neighbors
from .metrics import one_hot
from .oracle import LLMOracle, parse_decision
from .prompting import build_queries
from .response_surface import propagate_from_prototypes
from .schedules import nominal_core_budget, shadow_audit_budget, residual_aware_allocation
from .stratification import MiniBatchKMeansStratifier, Stratifier, select_farthest_first, validate_partition
from .types import APSResult, CallRecord, RoundRecord


class APSRunner:
    """Adaptive Prototype Simulation main loop."""

    def __init__(
        self,
        *,
        config: APSConfig,
        oracle: LLMOracle,
        stratifier: Stratifier | None = None,
    ) -> None:
        self.config = config
        self.oracle = oracle
        self.stratifier = stratifier or MiniBatchKMeansStratifier()

    def run(
        self,
        *,
        profiles: Sequence[Mapping[str, Any]],
        features: np.ndarray,
        events: Sequence[str],
        options: Sequence[str],
        initial_states: Sequence[str | int],
        graph_neighbors: Sequence[Sequence[int]],
    ) -> APSResult:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
            raise ValueError("features must be a finite N-by-D matrix with D > 0")
        n_agents = len(x)
        if n_agents != len(profiles):
            raise ValueError("profiles and features must have the same length")
        if n_agents == 0:
            raise ValueError("APS requires at least one agent")
        option_list = list(options)
        if not option_list or any(not isinstance(label, str) or not label for label in option_list):
            raise ValueError("options must be nonempty string labels")
        if len(set(option_list)) != len(option_list):
            raise ValueError("option labels must be unique")
        event_list = list(events)
        if not event_list or any(not isinstance(event, str) for event in event_list):
            raise ValueError("events must contain at least one string")

        profile_list = [dict(p) for p in profiles]
        rng = np.random.default_rng(self.config.seed)
        graph = validate_neighbors(graph_neighbors, n_agents)
        hard_states = self._initial_hard_states(initial_states, n_agents, option_list)

        structure = self.stratifier.partition(x, self.config)
        validate_partition(structure, n_agents)
        residual_scores = {
            int(k): float(self.config.initial_residual_score)
            for k in structure.stratum_indices
        }
        rounds: list[RoundRecord] = []

        for t, event in enumerate(event_list, start=1):
            summaries = neighbor_summaries(graph, hard_states, option_list)
            core_budget = nominal_core_budget(n_agents, self.config)
            nonempty_strata = sum(bool(len(ids)) for ids in structure.stratum_indices.values())
            if core_budget < nonempty_strata:
                raise ValueError("core budget must cover at least one prototype per nonempty stratum")
            stratum_budgets = residual_aware_allocation(
                structure.stratum_indices,
                core_budget,
                residual_scores,
                self.config.tau,
                self.config.initial_residual_score,
            )

            prototype_indices = self._select_prototypes(
                x,
                structure.stratum_indices,
                stratum_budgets,
                round_index=t,
            )
            tail_indices = structure.singleton_tail_indices
            prototype_set = set(int(i) for i in prototype_indices)
            correction_frames = {
                int(key): np.asarray([i for i in ids if int(i) not in prototype_set], dtype=np.int64)
                for key, ids in structure.stratum_indices.items()
            }
            audit_budget = shadow_audit_budget(n_agents, self.config)
            correction_strata = sum(bool(len(ids)) for ids in correction_frames.values())
            if audit_budget < correction_strata:
                raise ValueError("audit budget must cover every nonempty correction stratum")
            audit_counts = allocate_shadow_audits(
                correction_frames, audit_budget, residual_scores, self.config,
            )
            soft = np.zeros((n_agents, len(option_list)), dtype=float)
            new_hard = np.array(hard_states, copy=True)

            core_labels, core_metadata = self._query_and_parse(
                agent_ids=prototype_indices,
                profiles=profile_list,
                event=event,
                options=option_list,
                previous_states=hard_states,
                summaries=summaries,
                round_index=t,
                role="core_prototype",
            )
            tail_labels, tail_metadata = self._query_and_parse(
                agent_ids=tail_indices,
                profiles=profile_list,
                event=event,
                options=option_list,
                previous_states=hard_states,
                summaries=summaries,
                round_index=t,
                role="singleton_tail",
            )

            for agent_id, label in zip(prototype_indices, core_labels):
                soft[int(agent_id)] = one_hot(int(label), len(option_list))
                new_hard[int(agent_id)] = int(label)
            for agent_id, label in zip(tail_indices, tail_labels):
                soft[int(agent_id)] = one_hot(int(label), len(option_list))
                new_hard[int(agent_id)] = int(label)

            support_indices: dict[int, list[int]] = {}
            support_distances: dict[int, float] = {}
            support_slopes: dict[int, float | None] = {}
            for stratum_id, stratum in structure.stratum_indices.items():
                stratum = np.asarray(stratum, dtype=np.int64)
                stratum_prototypes = np.array([i for i in stratum if int(i) in prototype_set], dtype=np.int64)
                targets = np.array([i for i in stratum if int(i) not in prototype_set], dtype=np.int64)
                if len(targets) == 0:
                    continue
                propagation = propagate_from_prototypes(
                    x,
                    prototype_indices=stratum_prototypes,
                    prototype_soft=soft[stratum_prototypes],
                    target_indices=targets,
                    support_size=self.config.support_size,
                    epsilon=self.config.distance_epsilon,
                )
                soft[propagation.target_indices] = propagation.soft_vectors
                new_hard[propagation.target_indices] = propagation.hard_states
                support_indices.update(propagation.support_indices)
                support_distances.update(propagation.support_distances)
                support_slopes.update(propagation.support_slopes)

            audit_selection = select_shadow_audits(
                correction_frames,
                audit_counts,
                new_hard,
                len(option_list),
                self.config,
                rng,
            )
            audit_indices = audit_selection.all_indices
            audit_labels_list, audit_metadata = self._query_and_parse(
                agent_ids=audit_indices,
                profiles=profile_list,
                event=event,
                options=option_list,
                previous_states=hard_states,
                summaries=summaries,
                round_index=t,
                role="shadow_audit",
            )
            shadow_labels = {int(i): int(label) for i, label in zip(audit_indices, audit_labels_list)}
            audit_result = compute_audit_result(
                soft_ledger=soft,
                hard_states=new_hard,
                correction_frames=correction_frames,
                audit_selection=audit_selection,
                shadow_labels=shadow_labels,
                support_distances=support_distances,
                support_slopes=support_slopes,
                config=self.config,
            )
            residual_scores = audit_result.residual_scores or residual_scores

            call_record = CallRecord(
                round_index=t,
                core_prototype_calls=int(len(prototype_indices)),
                singleton_tail_calls=int(len(tail_indices)),
                shadow_audit_calls=int(len(audit_indices)),
            )
            rounds.append(
                RoundRecord(
                    round_index=t,
                    event=str(event),
                    prototype_indices=np.asarray(prototype_indices, dtype=np.int64),
                    singleton_tail_indices=np.asarray(tail_indices, dtype=np.int64),
                    shadow_audit_indices=np.asarray(audit_indices, dtype=np.int64),
                    hard_states=np.asarray(new_hard, dtype=np.int64),
                    soft_ledger=soft,
                    base_distribution=audit_result.base_distribution,
                    unprojected_distribution=audit_result.unprojected_distribution,
                    projected_distribution=audit_result.projected_distribution,
                    residual_scores=dict(residual_scores),
                    diagnostics=audit_result.diagnostics,
                    prototype_budgets=dict(stratum_budgets),
                    audit_budgets=dict(audit_counts),
                    audit_inclusion_probabilities=dict(audit_selection.inclusion_probabilities),
                    shadow_labels=shadow_labels,
                    support_indices=support_indices,
                    query_metadata={
                        "core_prototype": core_metadata,
                        "singleton_tail": tail_metadata,
                        "shadow_audit": audit_metadata,
                    },
                    call_record=call_record,
                )
            )
            hard_states = new_hard

        return APSResult(
            options=option_list,
            rounds=rounds,
            core_labels=structure.core_labels,
            singleton_tail_indices=structure.singleton_tail_indices,
            stratum_indices=structure.stratum_indices,
        )

    def _select_prototypes(
        self,
        features: np.ndarray,
        strata: Mapping[int, np.ndarray],
        budgets: Mapping[int, int],
        round_index: int,
    ) -> np.ndarray:
        parts = []
        for stratum_id in sorted(strata):
            count = int(budgets.get(int(stratum_id), 0))
            if count <= 0:
                continue
            rng = np.random.default_rng(self.config.seed + 1009 * round_index + int(stratum_id))
            chosen = select_farthest_first(
                features,
                strata[stratum_id],
                count=count,
                rng=rng,
                max_candidates=self.config.max_farthest_candidates,
            )
            if len(chosen):
                parts.append(chosen)
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.sort(np.concatenate(parts).astype(np.int64))

    def _query_and_parse(
        self,
        *,
        agent_ids: Sequence[int],
        profiles: Sequence[dict[str, Any]],
        event: str,
        options: Sequence[str],
        previous_states: Sequence[int],
        summaries: Sequence[str],
        round_index: int,
        role: str,
    ) -> tuple[list[int], dict[int, Mapping[str, Any]]]:
        ids = np.asarray(agent_ids, dtype=np.int64)
        if len(ids) == 0:
            return [], {}
        queries = build_queries(ids, profiles, event, options, previous_states, summaries, round_index, role)
        responses = self.oracle.batch_decide(queries)
        if len(responses) != len(ids):
            raise RuntimeError("Oracle returned a different number of responses than queries")
        labels: list[int] = []
        metadata: dict[int, Mapping[str, Any]] = {}
        for agent_id, response in zip(ids, responses):
            parsed = parse_decision(response.decision, options)
            if parsed is None:
                raise RuntimeError(f"Unparseable decision for agent {int(agent_id)}: {response.decision!r}")
            labels.append(int(parsed))
            if response.metadata is not None:
                if not isinstance(response.metadata, Mapping):
                    raise ValueError("Oracle response metadata must be a mapping")
                metadata[int(agent_id)] = deepcopy(dict(response.metadata))
        return labels, metadata

    @staticmethod
    def _initial_hard_states(
        initial_states: Sequence[str | int],
        n_agents: int,
        options: Sequence[str],
    ) -> np.ndarray:
        if len(initial_states) != n_agents:
            raise ValueError("initial_states must have length n_agents")
        parsed = []
        for value in initial_states:
            idx = parse_decision(value, options)
            if idx is None:
                raise ValueError(f"Could not parse initial state: {value!r}")
            parsed.append(idx)
        return np.asarray(parsed, dtype=np.int64)
