# APS Method Framework

A method-only reference implementation of Adaptive Prototype Simulation (APS).
The package implements the main simulation loop described in Sections 3.1--3.3:
adaptive prototypes, local response propagation, singleton-tail routing, shadow
audits, and separate recurrent and reporting states.

It includes no population dataset, application scenario, experiment preset,
baseline, benchmark runner, paper result, model endpoint, or credential. All
simulation inputs and the designated LLM pipeline are supplied by the caller.

## Installation

Requires Python 3.10 or newer, NumPy and scikit-learn.

```bash
python -m pip install .
```

## Integration

The following shows the interface; every value is supplied by your application.
`method_parameters` contains the fields of `APSConfig`, which has no run defaults.

```python
from aps_framework import APSConfig, APSRunner, CallableOracle

config = APSConfig(**method_parameters)

# decide_with_your_llm(query) formats the prompt, calls the designated LLM,
# and returns an exact option label or a zero-based integer option index.
oracle = CallableOracle(decide_with_your_llm)

result = APSRunner(config=config, oracle=oracle).run(
    profiles=profiles,
    features=features,
    initial_states=initial_states,
    graph_neighbors=graph_neighbors,
    events=events,
    options=options,
)

hard_states = result.final_hard_states
corrected_distribution = result.final_distribution
call_count = result.total_online_calls
```

For batching, implement `LLMOracle.batch_decide(queries)` and return one
`OracleResponse` per query in the same order. `AgentQuery` carries the profile,
pre-round own state, pre-round neighbor summary, current event, options, round
index, and query role. The role is accounting metadata; keep the same behavioral
prompt and LLM pipeline across prototype, tail, and audit queries.
Optional `OracleResponse.metadata` is retained by agent and query role in each
round. An adapter may supply retry or response-repair details there. The framework
does not invent metadata when the adapter supplies none.

## Inputs

| Input | Contract |
|---|---|
| `profiles` | One caller-defined mapping per agent |
| `features` | Finite N-by-D numeric array; scaling is performed by the caller |
| `initial_states` | N exact option labels or zero-based integer option indices |
| `graph_neighbors` | N adjacency lists of valid agent IDs; empty lists are allowed |
| `events` | Nonempty sequence of caller-defined event strings |
| `options` | Unique, nonempty decision labels |
| `oracle` | Caller-supplied designated LLM pipeline and parser |

There is no default graph, initial opinion, event, option set, or provider.
Integer decisions are **zero-based**. Numeric strings are accepted only when
they are actual option labels. Convert a provider's one-based answer in your
adapter before passing it to APS.

## Modules

| Module | Responsibility |
|---|---|
| `engine.py` | Multi-round execution and ledger separation |
| `stratification.py` | Tail scoring, core partitioning, prototype selection |
| `schedules.py` | Sublinear schedules and capacity-aware budget allocation |
| `response_surface.py` | Nearest-prototype interpolation and hard-state assignment |
| `audit.py` | Audit sampling, inclusion probabilities, correction and diagnostics |
| `graph.py` | Validation and summaries of the supplied fixed graph |
| `prompting.py` | Query-context records, without prompt templates |
| `oracle.py` | Provider-independent oracle interface and callable adapter |
| `config.py`, `types.py`, `metrics.py` | Method parameters, records and numeric helpers |

See [the method mapping](docs/METHOD.md) and [parameter reference](docs/PARAMETERS.md).

## Returned State and Accounting

Each `RoundRecord` holds the hard recurrent states, soft vectors, base distribution,
unprojected and projected corrected distributions, diagnostics, selected IDs,
allocation counts, shadow labels, and audit inclusion probabilities. These are
returned in memory, together with supplied query metadata; the package writes no
logs, caches, or result files.

Shadow audits never overwrite the hard recurrent states. All three query roles
use the pre-round state and neighborhood context. Counters report successful
agent-decision queries by role. A caller's transport retries, cache hits and API
costs require separate accounting in that adapter.

The configuration must fund at least one prototype in each nonempty core stratum
and at least one audit in each nonempty correction stratum. An invalid budget
raises before that round's LLM queries; the framework does not silently exceed it.
Audits can be zero when every agent is already queried directly.

## Tests

From the installed source package:

```bash
python -m unittest discover -s tests -v
```

The tests use small deterministic arrays and stub decision functions. They test
allocation, selection, propagation, audit correction, context use and input
validation. They are not application scenarios or reproductions of paper results.

The verified environment is Python 3.11.14, NumPy 2.2.6 and scikit-learn 1.8.0.

## Scope and Provenance

This is a newly assembled method reference package, adapted from the authors'
working APS modules with the corrections listed in [CHANGELOG.md](CHANGELOG.md).
Version 0.2.0 makes the local disagreement-slope calculation explicit and marks
sample variance as unavailable when only one agent is audited in a group.
It is not the archived code snapshot used for the paper's reported runs. It holds
per-agent states in memory and does not claim the throughput or memory behavior
of the paper's large-scale pipeline. Profiles and core strata remain fixed within
a run, as in the main method; dynamic-memory extensions are not included.
The package's concrete choices for selection, diagnostics and rounding are listed
in [docs/METHOD.md](docs/METHOD.md); they are not asserted to be the historical
settings used to generate the paper's tables.

## License

MIT; see [LICENSE](LICENSE). NumPy and scikit-learn are external dependencies and
retain their own licenses. No third-party library source is vendored here.

For academic attribution, refer to the APS paper when using the method. This
request is separate from the MIT license conditions.
