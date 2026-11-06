from __future__ import annotations

from numbers import Integral
from typing import Callable, Protocol, Sequence

from .types import AgentQuery, OracleResponse


class LLMOracle(Protocol):
    """Return one parsed decision per query, in the same order.

    Model selection, prompt wording, transport, credentials, and retries are
    supplied by the application, not by this method package.
    """

    def batch_decide(self, queries: Sequence[AgentQuery]) -> list[OracleResponse]:
        ...


class CallableOracle:
    """Adapt a caller-provided single-query function to the batch interface."""

    def __init__(self, decide: Callable[[AgentQuery], OracleResponse | str | int]) -> None:
        if not callable(decide):
            raise TypeError("decide must be callable")
        self.decide = decide

    def batch_decide(self, queries: Sequence[AgentQuery]) -> list[OracleResponse]:
        responses = []
        for query in queries:
            value = self.decide(query)
            responses.append(value if isinstance(value, OracleResponse) else OracleResponse(value))
        return responses


def parse_decision(decision: str | int, options: Sequence[str]) -> int | None:
    """Accept exact option labels or zero-based integer indices, never guesses."""
    if isinstance(decision, bool):
        return None
    if isinstance(decision, Integral):
        return int(decision) if 0 <= int(decision) < len(options) else None
    if isinstance(decision, str) and decision in options:
        return options.index(decision)
    return None
