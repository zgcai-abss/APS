from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def one_hot(index: int, n_options: int) -> np.ndarray:
    out = np.zeros(n_options, dtype=float)
    out[int(index)] = 1.0
    return out


def categorical_distribution(states: Sequence[int], n_options: int) -> np.ndarray:
    counts = np.bincount(np.asarray(states, dtype=int), minlength=n_options).astype(float)
    total = float(counts.sum())
    if total <= 0:
        return np.ones(n_options, dtype=float) / n_options
    return counts / total


def js_divergence(p: Sequence[float], q: Sequence[float], kl_floor: float) -> float:
    p_arr = _normalize(np.asarray(p, dtype=float))
    q_arr = _normalize(np.asarray(q, dtype=float))
    m = 0.5 * (p_arr + q_arr)
    return 0.5 * _kl(p_arr, m, kl_floor) + 0.5 * _kl(q_arr, m, kl_floor)


def project_to_simplex_by_clipping(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        return np.ones_like(arr, dtype=float) / len(arr)
    return arr / total


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0:
        return np.ones_like(values, dtype=float) / len(values)
    return values / total


def _kl(a: np.ndarray, b: np.ndarray, floor: float) -> float:
    total = 0.0
    for x, y in zip(a, b):
        if x > 0:
            total += float(x) * math.log(float(x) / max(float(y), floor))
    return total
