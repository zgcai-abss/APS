from __future__ import annotations

from typing import Sequence

from .types import AgentQuery


def build_queries(
    agent_ids: Sequence[int],
    profiles: Sequence[dict],
    event: str,
    options: Sequence[str],
    previous_states: Sequence[int],
    neighbor_summaries: Sequence[str],
    round_index: int,
    role: str,
) -> list[AgentQuery]:
    """Assemble context records; prompt wording belongs to the caller's oracle."""
    out: list[AgentQuery] = []
    for agent_id in agent_ids:
        idx = int(agent_id)
        prev_idx = int(previous_states[idx])
        prev_label = options[prev_idx] if 0 <= prev_idx < len(options) else str(prev_idx)
        out.append(
            AgentQuery(
                agent_id=idx,
                profile=profiles[idx],
                event=event,
                options=options,
                previous_state=prev_label,
                neighbor_summary=neighbor_summaries[idx],
                round_index=round_index,
                role=role,
            )
        )
    return out
