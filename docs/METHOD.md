# Mapping to the APS Method

| Paper element | Implementation |
|---|---|
| Section 3.1: hard recurrent states and soft response vectors | `RoundRecord` and `APSRunner.run` |
| Section 3.2: core strata, prototype queries and propagation | `MiniBatchKMeansStratifier`, `select_farthest_first`, `propagate_from_prototypes` |
| Eq. (2): local response surface and hard argmax | Inverse-distance normalized interpolation; ties select the first option |
| Eq. (3): scale-dependent prototype rate | `core_prototype_rate`, `nominal_core_budget` |
| Eq. (4): residual-aware allocation | `residual_aware_allocation` with weights proportional to group size times the square root of score plus tau |
| Section 3.3: singleton-tail routing | Median/MAD tail scores and direct queries; tails are excluded from core supports |
| Eq. (5): shadow-audit correction | Soft-ledger mean plus inclusion-probability-weighted residuals |
| Eq. (6): next-round risk | Normalized residual variance, support distance, local disagreement slope, mismatch and rare-state recall |
| Section 3.4: query categories | `CallRecord` counts prototypes, tails and shadow audits separately |

## Round Sequence

1. Build neighborhood summaries from the previous hard states.
2. Allocate the prototype budget and select unique prototypes in each core stratum.
3. Check that the audit budget can cover the nonempty correction strata.
4. Query prototypes and tails using the pre-round context.
5. Interpolate prototype response vectors and take hard argmax decisions.
6. Sample nonprototype core agents and obtain shadow labels using the same
   pre-round context and designated LLM pipeline.
7. Correct the aggregate report, compute diagnostics and update allocation scores.
8. Return records and advance only the hard rollout states to the next round.

## Sampling and Integer Budgets

The allocation first gives one slot to each nonempty stratum when the budget
permits, then assigns remaining slots by the residual-aware weights and largest
remainders. Capacity-capped slots are redistributed until the feasible budget is
used. The engine rejects a core budget smaller than the number of nonempty strata
because propagation would otherwise have no prototype support in some groups.

The prototype candidate limit is a working-set limit, not a call limit. When the
requested prototype count exceeds it, the candidate pool expands to at least that
count. All selected IDs are unique, including tied-feature cases.

Uniform audit sampling uses probability `sample_count / frame_size` within each
correction stratum. State-stratified sampling allocates at least one sample per
represented predicted state when possible, then distributes the remaining budget
by state-cell sizes. If that budget is too small to cover state cells, sampling
falls back to uniform sampling within the correction stratum. Inclusion
probabilities are recorded for every sampled agent.

Positive coverage of every nonempty correction stratum is required here. This
keeps the known-probability correction well defined without silently ignoring a
whole group's residuals. The package raises when the supplied audit budget cannot
meet that requirement.

## Correction and Diagnostics

The unprojected report applies the design-weighted correction on the fixed
nonprototype core frame. Directly queried agents enter through their one-hot
vectors. The projected report clips negative coordinates and renormalizes; that
nonlinear projection does not itself inherit an unbiasedness guarantee.

The residual variance is the sum of category-wise sample variances (`ddof=1`).
With fewer than two audits, it is recorded as `None`, not as zero variance or a
squared residual. Its unavailable term is omitted from the allocation score;
other available diagnostics still contribute. This omission is a small-sample
implementation rule, not evidence that uncertainty is absent.

For each audited agent's queried-prototype support, the local disagreement slope
is the largest pairwise L1 response-vector difference divided by Euclidean
feature distance. The denominator is floored at `distance_epsilon`. The group
diagnostic averages these slopes over audited agents with at least two supports.
With fewer than two supports, the slope is `None`; `slope_count` records how many
audits support this diagnostic. If no slope is available, its score term is
omitted. This is an empirical support-slope calculation, not a guaranteed
Lipschitz bound or a second-derivative curvature estimate.

The other diagnostics are hard-state mismatch with shadow labels, mean weighted
prototype-support distance, and rare-state recall. Rare states are defined by
predicted population frequency at or below `rare_state_threshold`. If the audit
sample has no rare labels, recall is set to one to avoid adding an unsupported
rare-state penalty. Variance, distance and slope are divided by their respective
maximum across groups before Eq. (6) is applied. Monitoring JSD is a separate
diagnostic and uses natural logarithms.

## Reference Implementation Choices

The paper specifies the core workflow and equations but leaves some operational
rules open. This package chooses centroid-initialized farthest-first prototypes,
an optional candidate-pool limit, capacity-aware largest-remainder allocation,
the support-slope formula above, and the stated small-sample handling. The audit
budget is allocated across correction frames using the same size-and-risk rule.
Both uniform and predicted-state-stratified audit designs are available. These
are explicit choices for this reference implementation, not new APS components
or claims about the original experiment code.

Query responses may include metadata about retries or repairs, as described in
Algorithm 2. The engine retains supplied metadata in `RoundRecord.query_metadata`
without changing decisions or treating retries as extra successful agent queries.
The caller implements provider access, prompt formatting and retries. No network
client, persistent logging, experiment configuration or benchmark is included.

The partitioner and LLM oracle are replaceable through their interfaces. The
default partitioner uses MiniBatch K-means; no particular population, feature
schema, graph model, scenario, provider, or experiment budget is built in.
