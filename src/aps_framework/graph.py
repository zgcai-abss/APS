from __future__ import annotations

from typing import Sequence

import numpy as np


def validate_neighbors(neighbors: Sequence[Sequence[int]], n_agents: int) -> tuple[np.ndarray, ...]:
    """Validate caller-supplied adjacency lists without constructing a graph."""
    if len(neighbors) != n_agents:
        raise ValueError("graph_neighbors must contain one row per agent")
    rows = []
    for row in neighbors:
        values = np.asarray(row)
        if values.ndim != 1 or (values.size and not np.issubdtype(values.dtype, np.integer)):
            raise ValueError("neighbor rows must contain integer agent IDs")
        values = values.astype(np.int64, copy=True)
        if np.any(values < 0) or np.any(values >= n_agents):
            raise ValueError("neighbor ID is out of range")
        if len(np.unique(values)) != len(values):
            raise ValueError("duplicate neighbors are not supported")
        rows.append(values)
    return tuple(rows)


def neighbor_summaries(
    neighbors: Sequence[np.ndarray],
    hard_states: Sequence[int],
    options: Sequence[str],
) -> list[str]:
    states = np.asarray(hard_states, dtype=int)
    summaries: list[str] = []
    for row in neighbors:
        if len(row) == 0:
            summaries.append("No neighbor-state information is available.")
            continue
        counts = np.bincount(states[row], minlength=len(options))
        parts = [f"{options[i]}: {int(count)}/{len(row)}" for i, count in enumerate(counts) if count]
        summaries.append("Previous-round neighbor states: " + "; ".join(parts) + ".")
    return summaries
