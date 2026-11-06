from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral


@dataclass(frozen=True)
class APSConfig:
    """Configuration schema for Adaptive Prototype Simulation.

    This class defines the parameters required by the method implementation,
    but intentionally does not encode concrete default values.
    Callers must provide parameter values from outside this package.
    """

    seed: int

    n_base: int
    alpha_base: float
    lambda_decay: float
    core_strata_base: int
    eta: float
    tail_base_fraction: float
    zeta: float
    audit_gamma: float
    audit_beta: float
    audit_min: int

    support_size: int
    distance_epsilon: float
    tau: float
    mad_floor: float

    lambda_rho: float
    lambda_e: float
    lambda_r: float
    rare_state_threshold: float
    kl_floor: float
    initial_residual_score: float

    kmeans_batch_size: int
    kmeans_n_init: int
    max_farthest_candidates: int

    audit_design: str

    def __post_init__(self) -> None:
        for name in ("seed", "n_base", "core_strata_base", "audit_min", "support_size",
                     "kmeans_batch_size", "kmeans_n_init", "max_farthest_candidates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            minimum = 0 if name in {"seed", "audit_min"} else 1
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        for name, value in vars(self).items():
            if name != "audit_design" and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in ("lambda_decay", "eta", "zeta", "audit_beta"):
            value = getattr(self, name)
            if not (0.0 < value < 1.0):
                raise ValueError(f"{name} must be in (0, 1)")
        if not 0 < self.alpha_base <= 1:
            raise ValueError("alpha_base must be in (0, 1]")
        if not 0 <= self.tail_base_fraction <= 1:
            raise ValueError("tail_base_fraction must be in [0, 1]")
        if self.audit_gamma < 0:
            raise ValueError("audit_gamma must be nonnegative")
        if self.distance_epsilon <= 0 or self.tau <= 0 or self.mad_floor <= 0:
            raise ValueError("distance_epsilon, tau, and mad_floor must be positive")
        if not 0 <= self.rare_state_threshold <= 1:
            raise ValueError("rare_state_threshold must be in [0, 1]")
        if self.kl_floor <= 0:
            raise ValueError("kl_floor must be positive")
        if self.initial_residual_score < 0:
            raise ValueError("initial_residual_score must be nonnegative")
        if min(self.lambda_rho, self.lambda_e, self.lambda_r) < 0:
            raise ValueError("diagnostic weights must be nonnegative")
        if self.audit_design not in {"uniform", "state_stratified"}:
            raise ValueError("audit_design must be 'uniform' or 'state_stratified'")
