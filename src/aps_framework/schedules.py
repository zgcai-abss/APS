from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .config import APSConfig


def core_strata_count(n_agents: int, config: APSConfig) -> int:
    if n_agents <= 0:
        return 0
    if n_agents <= config.n_base:
        count = config.core_strata_base
    else:
        count = math.floor(config.core_strata_base * (n_agents / config.n_base) ** config.eta)
    return max(1, min(int(count), n_agents))


def singleton_tail_count(n_agents: int, config: APSConfig) -> int:
    if config.tail_base_fraction <= 0:
        return 0
    base_count = config.tail_base_fraction * config.n_base
    if n_agents <= config.n_base:
        count = math.ceil(base_count)
    else:
        count = math.ceil(base_count * (n_agents / config.n_base) ** config.zeta)
    return max(0, min(int(count), n_agents))


def core_prototype_rate(n_agents: int, config: APSConfig) -> float:
    if n_agents <= config.n_base:
        return float(config.alpha_base)
    return float(config.alpha_base * (config.n_base / n_agents) ** config.lambda_decay)


def nominal_core_budget(n_agents: int, config: APSConfig) -> int:
    return max(0, min(n_agents, int(math.ceil(core_prototype_rate(n_agents, config) * n_agents))))


def shadow_audit_budget(n_agents: int, config: APSConfig) -> int:
    if n_agents <= 0:
        return 0
    scaled = math.floor(
        config.audit_gamma
        * config.n_base
        * max(1.0, (n_agents / config.n_base) ** config.audit_beta)
    )
    return min(n_agents, max(int(config.audit_min), int(scaled)))


def residual_aware_allocation(
    strata: Mapping[int, np.ndarray],
    total_budget: int,
    residual_scores: Mapping[int, float],
    tau: float,
    default_residual_score: float,
) -> dict[int, int]:
    """Allocate integer budgets using |C_m| sqrt(R_m + tau).

    If the total budget permits, every nonempty stratum receives at least one
    slot before remaining slots are rounded from the continuous relaxation.
    """

    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if not math.isfinite(default_residual_score) or default_residual_score < 0:
        raise ValueError("default residual score must be finite and nonnegative")
    if isinstance(total_budget, bool) or not isinstance(total_budget, (int, np.integer)) or total_budget < 0:
        raise ValueError("total_budget must be a nonnegative integer")
    nonempty = {int(k): np.asarray(strata[k], dtype=int) for k in sorted(strata) if len(strata[k]) > 0}
    if total_budget <= 0 or not nonempty:
        return {int(k): 0 for k in strata}

    total_budget = min(int(total_budget), sum(len(v) for v in nonempty.values()))
    out = {int(k): 0 for k in strata}

    if total_budget >= len(nonempty):
        for key in nonempty:
            out[key] = 1
        remaining = total_budget - len(nonempty)
    else:
        remaining = total_budget

    keys = list(nonempty)
    scores = [float(residual_scores.get(k, default_residual_score)) for k in keys]
    if any(not math.isfinite(value) or value < 0 for value in scores):
        raise ValueError("residual scores must be finite and nonnegative")
    weights = np.array(
        [len(nonempty[k]) * math.sqrt(value + tau) for k, value in zip(keys, scores)],
        dtype=float,
    )
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        weights = np.array([len(nonempty[k]) for k in keys], dtype=float)
    # Reallocate capped slots until the feasible budget is exhausted.
    while remaining > 0:
        active = [i for i, key in enumerate(keys) if out[key] < len(nonempty[key])]
        active_weights = weights[active]
        active_weights = active_weights / active_weights.max()
        raw = remaining * active_weights / active_weights.sum()
        floors = np.floor(raw).astype(int)
        for position, count in zip(active, floors):
            key = keys[position]
            take = min(int(count), len(nonempty[key]) - out[key], remaining)
            out[key] += take
            remaining -= take
        if remaining:
            for local in np.argsort(-(raw - floors), kind="stable"):
                key = keys[active[int(local)]]
                if out[key] < len(nonempty[key]):
                    out[key] += 1
                    remaining -= 1
                if remaining == 0:
                    break
    return out
