from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .config import APSConfig
from .metrics import categorical_distribution, js_divergence, one_hot, project_to_simplex_by_clipping
from .schedules import residual_aware_allocation


@dataclass
class AuditSelection:
    selected: dict[int, np.ndarray]
    inclusion_probabilities: dict[int, float]

    @property
    def all_indices(self) -> np.ndarray:
        if not self.selected:
            return np.empty(0, dtype=np.int64)
        parts = [v for v in self.selected.values() if len(v) > 0]
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.sort(np.concatenate(parts).astype(np.int64))


@dataclass
class AuditResult:
    base_distribution: np.ndarray
    unprojected_distribution: np.ndarray
    projected_distribution: np.ndarray
    diagnostics: dict[int, dict[str, float | None]]
    residual_scores: dict[int, float]


def allocate_shadow_audits(
    correction_frames: Mapping[int, np.ndarray],
    total_budget: int,
    residual_scores: Mapping[int, float],
    config: APSConfig,
) -> dict[int, int]:
    return residual_aware_allocation(
        correction_frames,
        total_budget,
        residual_scores,
        config.tau,
        config.initial_residual_score,
    )


def select_shadow_audits(
    correction_frames: Mapping[int, np.ndarray],
    counts: Mapping[int, int],
    predicted_states: Sequence[int],
    n_options: int,
    config: APSConfig,
    rng: np.random.Generator,
) -> AuditSelection:
    selected: dict[int, np.ndarray] = {}
    psi: dict[int, float] = {}
    states = np.asarray(predicted_states, dtype=int)
    for stratum_id, frame in correction_frames.items():
        frame = np.asarray(frame, dtype=np.int64)
        count = int(counts.get(int(stratum_id), 0))
        if count < 0 or count > len(frame):
            raise ValueError("audit count is outside the correction-frame capacity")
        if len(frame) and count == 0:
            raise ValueError("every nonempty correction stratum needs positive audit coverage")
        if count == 0:
            selected[int(stratum_id)] = np.empty(0, dtype=np.int64)
            continue
        if config.audit_design == "uniform":
            chosen, probs = _uniform_without_replacement(frame, count, rng)
        else:
            chosen, probs = _state_stratified_without_replacement(frame, count, states, n_options, rng)
        selected[int(stratum_id)] = chosen
        psi.update(probs)
    return AuditSelection(selected=selected, inclusion_probabilities=psi)


def compute_audit_result(
    *,
    soft_ledger: np.ndarray,
    hard_states: np.ndarray,
    correction_frames: Mapping[int, np.ndarray],
    audit_selection: AuditSelection,
    shadow_labels: Mapping[int, int],
    support_distances: Mapping[int, float],
    support_slopes: Mapping[int, float | None],
    config: APSConfig,
) -> AuditResult:
    n_agents, n_options = soft_ledger.shape
    selected_ids = set(map(int, audit_selection.all_indices))
    if selected_ids != set(shadow_labels):
        raise ValueError("shadow labels must match the selected audit IDs exactly")
    base = soft_ledger.mean(axis=0)
    correction = np.zeros(n_options, dtype=float)

    for agent_id, label in shadow_labels.items():
        psi = audit_selection.inclusion_probabilities.get(int(agent_id))
        if psi is None or not np.isfinite(psi) or not 0 < psi <= 1:
            raise ValueError(f"Missing positive inclusion probability for audited agent {agent_id}")
        correction += (one_hot(int(label), n_options) - soft_ledger[int(agent_id)]) / float(psi)

    unprojected = base + correction / max(1, n_agents)
    projected = project_to_simplex_by_clipping(unprojected)

    diagnostics = _compute_diagnostics(
        soft_ledger=soft_ledger,
        hard_states=hard_states,
        correction_frames=correction_frames,
        audit_selection=audit_selection,
        shadow_labels=shadow_labels,
        support_distances=support_distances,
        support_slopes=support_slopes,
        n_options=n_options,
        rare_state_threshold=config.rare_state_threshold,
        kl_floor=config.kl_floor,
    )
    scores = residual_scores_from_diagnostics(diagnostics, config)
    return AuditResult(base, unprojected, projected, diagnostics, scores)


def residual_scores_from_diagnostics(
    diagnostics: Mapping[int, Mapping[str, float | None]],
    config: APSConfig,
) -> dict[int, float]:
    if not diagnostics:
        return {}
    keys = sorted(int(k) for k in diagnostics)

    def values(name: str) -> np.ndarray:
        return np.asarray([
            float(value) if (value := diagnostics[k].get(name)) is not None else 0.0
            for k in keys
        ], dtype=float)

    v = _normalize_by_max(values("residual_variance"))
    rho = _normalize_by_max(values("avg_support_distance"))
    local_slope = _normalize_by_max(values("local_disagreement_slope"))
    mismatch = np.clip(values("mismatch_rate"), 0.0, 1.0)
    recall = np.clip(values("rare_state_recall"), 0.0, 1.0)

    scores: dict[int, float] = {}
    for pos, key in enumerate(keys):
        score = (
            float(v[pos])
            + config.lambda_rho * float(local_slope[pos] ** 2) * float(rho[pos] ** 2)
            + config.lambda_e * float(mismatch[pos] ** 2)
            + config.lambda_r * float((1.0 - recall[pos]) ** 2)
        )
        scores[key] = max(score, 0.0)
    return scores


def _uniform_without_replacement(
    frame: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[int, float]]:
    chosen = np.sort(rng.choice(frame, size=count, replace=False).astype(np.int64))
    prob = count / len(frame)
    return chosen, {int(i): float(prob) for i in chosen}


def _state_stratified_without_replacement(
    frame: np.ndarray,
    count: int,
    states: np.ndarray,
    n_options: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[int, float]]:
    groups = {
        state: frame[states[frame] == state]
        for state in range(n_options)
        if np.any(states[frame] == state)
    }
    if count < len(groups):
        return _uniform_without_replacement(frame, count, rng)

    ordered_states = sorted(groups)
    alloc = residual_aware_allocation(groups, count, {}, tau=1.0, default_residual_score=0.0)

    chosen_parts = []
    psi: dict[int, float] = {}
    for state in ordered_states:
        cell = groups[state]
        take = min(alloc[state], len(cell))
        if take <= 0:
            continue
        chosen = np.sort(rng.choice(cell, size=take, replace=False).astype(np.int64))
        chosen_parts.append(chosen)
        prob = take / len(cell)
        for agent_id in chosen:
            psi[int(agent_id)] = float(prob)
    if not chosen_parts:
        return np.empty(0, dtype=np.int64), {}
    return np.sort(np.concatenate(chosen_parts).astype(np.int64)), psi


def _compute_diagnostics(
    *,
    soft_ledger: np.ndarray,
    hard_states: np.ndarray,
    correction_frames: Mapping[int, np.ndarray],
    audit_selection: AuditSelection,
    shadow_labels: Mapping[int, int],
    support_distances: Mapping[int, float],
    support_slopes: Mapping[int, float | None],
    n_options: int,
    rare_state_threshold: float,
    kl_floor: float,
) -> dict[int, dict[str, float | None]]:
    diagnostics: dict[int, dict[str, float | None]] = {}
    full_pred_dist = categorical_distribution(hard_states, n_options)
    threshold = rare_state_threshold
    rare_states = set(int(i) for i, p in enumerate(full_pred_dist) if float(p) <= threshold)

    for stratum_id, frame in correction_frames.items():
        selected = audit_selection.selected.get(int(stratum_id), np.empty(0, dtype=np.int64))
        if len(selected) == 0:
            diagnostics[int(stratum_id)] = {
                "audit_count": 0.0,
                "mismatch_rate": 0.0,
                "residual_variance": None,
                "monitoring_jsd": 0.0,
                "avg_support_distance": 0.0,
                "local_disagreement_slope": None,
                "slope_count": 0.0,
                "rare_state_recall": 1.0,
            }
            continue

        labels = np.asarray([int(shadow_labels[int(i)]) for i in selected], dtype=int)
        pred = np.asarray([int(hard_states[int(i)]) for i in selected], dtype=int)
        residuals = np.vstack([one_hot(label, n_options) - soft_ledger[int(i)] for i, label in zip(selected, labels)])
        if len(selected) > 1:
            residual_var = float(np.var(residuals, axis=0, ddof=1).sum())
        else:
            residual_var = None

        soft_dist = soft_ledger[selected].mean(axis=0)
        shadow_dist = categorical_distribution(labels, n_options)
        rare_mask = np.asarray([label in rare_states for label in labels], dtype=bool)
        if rare_mask.any():
            rare_recall = float(np.mean(pred[rare_mask] == labels[rare_mask]))
        else:
            rare_recall = 1.0

        slopes = [value for i in selected
                  if (value := support_slopes.get(int(i))) is not None]

        diagnostics[int(stratum_id)] = {
            "audit_count": float(len(selected)),
            "correction_frame_size": float(len(frame)),
            "mismatch_rate": float(np.mean(pred != labels)),
            "residual_variance": residual_var,
            "monitoring_jsd": float(js_divergence(soft_dist, shadow_dist, kl_floor)),
            "avg_support_distance": float(np.mean([support_distances.get(int(i), 0.0) for i in selected])),
            "local_disagreement_slope": float(np.mean(slopes)) if slopes else None,
            "slope_count": float(len(slopes)),
            "rare_state_recall": rare_recall,
        }
    return diagnostics


def _normalize_by_max(values: np.ndarray) -> np.ndarray:
    values = np.where(np.isfinite(values), values, 0.0)
    max_value = float(np.max(np.abs(values))) if len(values) else 0.0
    if max_value <= 0:
        return np.zeros_like(values, dtype=float)
    return values / max_value
