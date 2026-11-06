# Method Package Assembly

## 0.2.0 - 2026-11-06

- Replace response dispersion (`1 - max(h_i)`) with an explicit pairwise local
  response-disagreement slope on queried prototype support.
- Report unavailable single-audit sample variance as `None`, rather than using
  a squared residual under a variance label. Omit unavailable diagnostic terms
  from allocation while retaining other terms.
- Record the number of audits supporting a slope estimate.
- Preserve caller-supplied retry/repair metadata by query role and agent.
- Document implementation choices, diagnostic normalization, and small-sample
  handling without claiming they generated the historical paper experiments.
- Add regression tests for the diagnostics, risk equation, and metadata handling.

## 0.1.0 - 2026-11-06

- Extract the APS method modules from the existing framework working copy.
- Remove baselines, scenario files, experiment launchers, experiment I/O, fixed
  experiment-budget overrides, graph-generation rules and provider configuration.
- Require caller-supplied initial states, graph, events, options and oracle.
- Keep query context assembly separate from application-specific prompt wording.
- Redistribute capacity-capped allocation slots instead of dropping the budget.
- Expand prototype candidate pools when needed to avoid duplicate selections.
- Use the capacity-aware allocation for predicted-state audit cells as well.
- Reject missing prototype/audit coverage before issuing that round's queries.
- Record audit inclusion probabilities and shadow labels in the returned state.
- Use exact labels or zero-based integer decisions consistently at the API boundary.
- Add deterministic offline tests covering method invariants and edge cases.
- Apply the MIT license chosen for this method package.

The original framework directory is not changed. The accompanying manuscript's
Checklist is separately updated to describe a method-only release, not an
experiment reproduction package; methods, results, and references are unchanged.
This version is a reference implementation, not a claim that the original paper
experiments used these corrected source files.
