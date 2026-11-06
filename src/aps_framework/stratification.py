from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import APSConfig
from .schedules import core_strata_count, singleton_tail_count


@dataclass
class StratificationResult:
    core_labels: np.ndarray
    singleton_tail_indices: np.ndarray
    stratum_indices: dict[int, np.ndarray]
    centers: np.ndarray | None = None


class Stratifier(Protocol):
    def partition(self, features: np.ndarray, config: APSConfig) -> StratificationResult:
        ...


def validate_partition(result: StratificationResult, n_agents: int) -> None:
    labels = np.asarray(result.core_labels)
    tails = np.asarray(result.singleton_tail_indices)
    if labels.shape != (n_agents,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("core_labels must contain one integer label per agent")
    all_indices = []
    for ids in [tails, *result.stratum_indices.values()]:
        ids = np.asarray(ids)
        if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
            raise ValueError("partition members must be one-dimensional integer IDs")
        all_indices.extend(int(i) for i in ids)
    if sorted(all_indices) != list(range(n_agents)):
        raise ValueError("core strata and tails must be disjoint and cover every agent exactly once")
    if tails.size and not np.all(labels[tails] == -1):
        raise ValueError("tail labels must be -1")
    for key, ids in result.stratum_indices.items():
        ids = np.asarray(ids, dtype=np.int64)
        if not isinstance(key, (int, np.integer)) or key < 0 or not np.all(labels[ids] == key):
            raise ValueError("stratum IDs must agree with core_labels")


class MiniBatchKMeansStratifier:
    def partition(self, features: np.ndarray, config: APSConfig) -> StratificationResult:
        from sklearn.cluster import MiniBatchKMeans

        x = np.asarray(features, dtype=np.float32)
        n_agents = len(x)
        if n_agents == 0:
            return StratificationResult(
                core_labels=np.empty(0, dtype=np.int32),
                singleton_tail_indices=np.empty(0, dtype=np.int64),
                stratum_indices={},
                centers=None,
            )

        tail_n = singleton_tail_count(n_agents, config)
        tail = select_singleton_tail(x, tail_n, mad_floor=config.mad_floor)
        tail_set = set(int(i) for i in tail)
        core_indices = np.array([i for i in range(n_agents) if i not in tail_set], dtype=np.int64)
        labels = np.full(n_agents, -1, dtype=np.int32)
        if len(core_indices) == 0:
            return StratificationResult(labels, tail, {}, None)

        k = min(core_strata_count(n_agents, config), len(core_indices))
        if k <= 1:
            labels[core_indices] = 0
            return StratificationResult(labels, tail, {0: core_indices}, x[core_indices].mean(axis=0, keepdims=True))

        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=config.seed,
            batch_size=min(config.kmeans_batch_size, max(100, len(core_indices))),
            n_init=config.kmeans_n_init,
        )
        core_labels = km.fit_predict(x[core_indices])
        labels[core_indices] = core_labels.astype(np.int32)
        strata = {
            int(c): np.sort(core_indices[core_labels == c]).astype(np.int64)
            for c in range(k)
            if np.any(core_labels == c)
        }
        return StratificationResult(labels, tail, strata, km.cluster_centers_.astype(np.float32))


def robust_mad_tail_scores(features: np.ndarray, mad_floor: float) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0)
    mad = np.maximum(mad, mad_floor)
    standardized = (x - median) / mad
    return np.linalg.norm(standardized, axis=1)


def select_singleton_tail(features: np.ndarray, count: int, mad_floor: float) -> np.ndarray:
    n_agents = len(features)
    count = max(0, min(int(count), n_agents))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    scores = robust_mad_tail_scores(features, mad_floor=mad_floor)
    order = np.lexsort((np.arange(n_agents), -scores))
    return np.sort(order[:count]).astype(np.int64)


def select_farthest_first(
    features: np.ndarray,
    candidate_indices: np.ndarray,
    count: int,
    rng: np.random.Generator,
    max_candidates: int,
) -> np.ndarray:
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.ndim != 1 or len(np.unique(candidates)) != len(candidates):
        raise ValueError("prototype candidates must be a one-dimensional array of unique IDs")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if np.any(candidates < 0) or np.any(candidates >= len(features)):
        raise ValueError("prototype candidate ID is out of range")
    count = max(0, min(int(count), len(candidates)))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count >= len(candidates):
        return np.sort(candidates)

    candidate_limit = max(int(max_candidates), count)
    if len(candidates) > candidate_limit:
        candidates = np.sort(rng.choice(candidates, size=candidate_limit, replace=False).astype(np.int64))

    x = np.asarray(features, dtype=np.float32)
    sub = np.asarray(x[candidates], dtype=np.float64)
    centroid = sub.mean(axis=0)
    first = int(np.argmin(np.sum((sub - centroid) ** 2, axis=1)))
    selected_positions = [first]
    min_sq = np.sum((sub - sub[first]) ** 2, axis=1)
    min_sq[first] = -1.0
    while len(selected_positions) < count:
        pos = int(np.argmax(min_sq))
        selected_positions.append(pos)
        dist_sq = np.sum((sub - sub[pos]) ** 2, axis=1)
        min_sq = np.minimum(min_sq, dist_sq)
        min_sq[selected_positions] = -1.0
    return np.sort(candidates[np.asarray(selected_positions, dtype=int)]).astype(np.int64)
