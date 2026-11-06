from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PropagationResult:
    target_indices: np.ndarray
    soft_vectors: np.ndarray
    hard_states: np.ndarray
    support_indices: dict[int, list[int]]
    support_distances: dict[int, float]
    support_slopes: dict[int, float | None]


def propagate_from_prototypes(
    features: np.ndarray,
    prototype_indices: np.ndarray,
    prototype_soft: np.ndarray,
    target_indices: np.ndarray,
    support_size: int,
    epsilon: float,
) -> PropagationResult:
    from sklearn.neighbors import NearestNeighbors

    prototypes = np.asarray(prototype_indices, dtype=np.int64)
    targets = np.asarray(target_indices, dtype=np.int64)
    if len(targets) == 0:
        n_options = prototype_soft.shape[1] if prototype_soft.ndim == 2 else 0
        return PropagationResult(targets, np.empty((0, n_options)), np.empty(0, dtype=np.int64), {}, {}, {})
    if len(prototypes) == 0:
        raise ValueError("At least one prototype is required for propagation")

    k = min(max(1, int(support_size)), len(prototypes))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(features[prototypes])
    distances, local_support = nn.kneighbors(features[targets], return_distance=True)
    weights = 1.0 / (distances + epsilon)
    weights = weights / weights.sum(axis=1, keepdims=True)
    soft = np.einsum("ij,ijk->ik", weights, prototype_soft[local_support])
    hard = np.argmax(soft, axis=1).astype(np.int64)

    support_indices: dict[int, list[int]] = {}
    support_distances: dict[int, float] = {}
    support_slopes: dict[int, float | None] = {}
    for row, agent_id in enumerate(targets):
        support = prototypes[local_support[row]]
        support_indices[int(agent_id)] = [int(x) for x in support]
        support_distances[int(agent_id)] = float(np.average(distances[row], weights=weights[row]))
        support_slopes[int(agent_id)] = local_response_slope(
            np.asarray(features[support], dtype=np.float64),
            prototype_soft[local_support[row]],
            epsilon,
        )

    return PropagationResult(
        target_indices=targets,
        soft_vectors=soft,
        hard_states=hard,
        support_indices=support_indices,
        support_distances=support_distances,
        support_slopes=support_slopes,
    )


def local_response_slope(
    support_features: np.ndarray,
    support_responses: np.ndarray,
    distance_floor: float,
) -> float | None:
    """Maximum pairwise L1 response difference per Euclidean feature distance.

    This is an empirical slope on queried support, not a bound on the LLM kernel.
    A single support point provides no slope estimate.
    """
    if len(support_features) < 2:
        return None
    left, right = np.triu_indices(len(support_features), k=1)
    distances = np.linalg.norm(support_features[left] - support_features[right], axis=1)
    differences = np.abs(support_responses[left] - support_responses[right]).sum(axis=1)
    return float(np.max(differences / np.maximum(distances, distance_floor)))
