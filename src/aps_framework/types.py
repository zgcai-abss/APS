from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AgentQuery:
    agent_id: int
    profile: Mapping[str, Any]
    event: str
    options: Sequence[str]
    previous_state: str
    neighbor_summary: str
    round_index: int
    role: str


@dataclass(frozen=True)
class OracleResponse:
    """A parsed exact option label or zero-based integer option index."""
    decision: str | int
    reasoning: str = ""
    raw: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass
class CallRecord:
    round_index: int
    core_prototype_calls: int = 0
    singleton_tail_calls: int = 0
    shadow_audit_calls: int = 0

    @property
    def total(self) -> int:
        return self.core_prototype_calls + self.singleton_tail_calls + self.shadow_audit_calls


@dataclass
class RoundRecord:
    round_index: int
    event: str
    prototype_indices: np.ndarray
    singleton_tail_indices: np.ndarray
    shadow_audit_indices: np.ndarray
    hard_states: np.ndarray
    soft_ledger: np.ndarray
    base_distribution: np.ndarray
    unprojected_distribution: np.ndarray
    projected_distribution: np.ndarray
    residual_scores: dict[int, float]
    diagnostics: dict[int, Mapping[str, float | None]]
    prototype_budgets: dict[int, int]
    audit_budgets: dict[int, int]
    audit_inclusion_probabilities: dict[int, float]
    shadow_labels: dict[int, int]
    support_indices: dict[int, list[int]] = field(default_factory=dict)
    query_metadata: dict[str, dict[int, Mapping[str, Any]]] = field(default_factory=dict)
    call_record: CallRecord = field(default_factory=lambda: CallRecord(round_index=0))


@dataclass
class APSResult:
    options: list[str]
    rounds: list[RoundRecord]
    core_labels: np.ndarray
    singleton_tail_indices: np.ndarray
    stratum_indices: dict[int, np.ndarray]

    @property
    def total_online_calls(self) -> int:
        return sum(r.call_record.total for r in self.rounds)

    @property
    def final_hard_states(self) -> np.ndarray:
        if not self.rounds:
            return np.array([], dtype=np.int64)
        return self.rounds[-1].hard_states.copy()

    @property
    def final_distribution(self) -> np.ndarray:
        if not self.rounds:
            return np.array([], dtype=float)
        return self.rounds[-1].projected_distribution.copy()
