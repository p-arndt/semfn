from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Decision[T]:
    """A semantic result together with how sure the backend was.

    `confidence` is always the probability of the selected option, so one
    threshold means the same thing for every return type. `distribution` lists
    every option with its probability, in declaration order.
    """

    value: T
    confidence: float
    distribution: tuple[tuple[Any, float], ...]

    def probability(self, option: object) -> float:
        for candidate, probability in self.distribution:
            if candidate is option or candidate == option:
                return probability
        raise KeyError(option)


class UncertainDecision(Exception):
    def __init__(self, decision: Decision[Any], minimum: float) -> None:
        self.decision = decision
        self.minimum = minimum
        super().__init__(
            f"Decision confidence {decision.confidence:.3f} is below {minimum:.3f}"
        )
