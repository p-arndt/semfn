from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .decision import Decision, UncertainDecision

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from .function import SemanticFunction

_CONCURRENCY = 8


@dataclass(frozen=True, slots=True)
class Case:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    expected: Any


def case(*args: Any, expected: Any, **kwargs: Any) -> Case:
    """One labeled example: the call arguments and the value it should return."""
    return Case(args, kwargs, expected)


@dataclass(frozen=True, slots=True)
class CaseResult:
    case: Case
    decision: Decision[Any]
    uncertain: bool
    """The function raised `UncertainDecision` instead of returning a value."""

    @property
    def passed(self) -> bool:
        return not self.uncertain and self.decision.value == self.case.expected


@dataclass(frozen=True, slots=True)
class EvalReport:
    name: str
    results: tuple[CaseResult, ...]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.results else 0.0

    @property
    def failures(self) -> tuple[CaseResult, ...]:
        return tuple(result for result in self.results if not result.passed)


class Evaluation:
    """Cases registered on a function. Await it to run them right away."""

    __slots__ = ("_cases", "_function")

    def __init__(
        self, function: SemanticFunction[..., Any], cases: Sequence[Case]
    ) -> None:
        self._function = function
        self._cases = cases

    def __await__(self) -> Generator[Any, None, EvalReport]:
        return run(self._function, self._cases).__await__()


async def run(
    function: SemanticFunction[..., Any], cases: Sequence[Case] | None = None
) -> EvalReport:
    """Run `cases`, or every case registered on `function`, through its policy.

    The confidence policy stays active on purpose: a report should describe
    what callers of the function actually get back.
    """
    limit = asyncio.Semaphore(_CONCURRENCY)

    async def check(item: Case) -> CaseResult:
        async with limit:
            try:
                decision = await function.result(*item.args, **item.kwargs)
            except UncertainDecision as error:
                return CaseResult(item, error.decision, True)
        return CaseResult(item, decision, False)

    selected = function.cases if cases is None else cases
    results = await asyncio.gather(*(check(item) for item in selected))
    return EvalReport(function.__name__, tuple(results))
