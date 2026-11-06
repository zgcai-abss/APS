# Method Parameters

All `APSConfig` fields must be supplied by the caller. This package does not
select an operating point or provide paper-experiment presets.

| Field | Meaning |
|---|---|
| `seed` | Nonnegative seed for partitioning and sampling |
| `n_base` | Population scale at which the base schedules are defined |
| `alpha_base` | Base core-prototype rate in (0, 1] |
| `lambda_decay` | Decay exponent of the core-prototype rate |
| `core_strata_base` | Base number of core strata |
| `eta` | Growth exponent of the number of core strata |
| `tail_base_fraction` | Base tail count divided by `n_base` |
| `zeta` | Growth exponent of the tail count |
| `audit_gamma` | Coefficient of the audit schedule |
| `audit_beta` | Growth exponent of the audit budget |
| `audit_min` | Minimum requested audit count before population capacity limits |
| `support_size` | Maximum number of prototype neighbors per propagated agent |
| `distance_epsilon` | Positive stabilizer in interpolation weights and denominator floor for local support slopes |
| `tau` | Positive residual-score stabilizer in allocation |
| `mad_floor` | Positive denominator floor for median/MAD tail scores |
| `lambda_rho` | Weight of squared local disagreement slope times squared support distance |
| `lambda_e` | Weight of squared audit mismatch |
| `lambda_r` | Weight of squared missed rare-state recall |
| `rare_state_threshold` | Maximum predicted population frequency of a rare state |
| `kl_floor` | Positive numerical floor for diagnostic JSD |
| `initial_residual_score` | Initial score assigned to each core stratum |
| `kmeans_batch_size` | MiniBatch K-means batch size |
| `kmeans_n_init` | MiniBatch K-means initialization count |
| `max_farthest_candidates` | Prototype-selection candidate limit, expanded if needed to meet the requested unique count |
| `audit_design` | `uniform` or `state_stratified` |

The four growth/decay exponents are in (0, 1). Tail counts and audit counts are
bounded by population capacity. Actual core allocation is bounded by the remaining
core population after tails have been selected. Configuration validation checks
numeric ranges and finiteness; run-time validation checks graph, labels, partition
coverage and feasible per-stratum budgets before querying.
