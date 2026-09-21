from __future__ import annotations

import asyncio
import copy
import inspect
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, Never, Protocol, get_type_hints, overload

from .decision import Decision, UncertainDecision
from .evals import Case, Evaluation
from .runtime import Backend, SemanticRuntime, current_runtime
from .schema import NONE_LABEL, Schema, choice_schema, plan_for, serialize

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping

_MISSING: Any = object()


class SemanticCall[R]:
    """One pending semantic call. Await it, or pass several to `gather`."""

    __slots__ = ("_arguments", "_function", "_wants_decision")

    def __init__(
        self,
        function: SemanticFunction[..., Any],
        arguments: dict[str, Any],
        wants_decision: bool,
    ) -> None:
        self._function = function
        self._arguments = arguments
        self._wants_decision = wants_decision

    def __await__(self) -> Generator[Any, None, R]:
        return self._run().__await__()

    async def _run(self) -> R:
        (result,) = await gather(self)
        return result

    def _finish(self, schema: Schema, answer: Mapping[str, Any]) -> Any:
        decision = self._function._apply_policy(schema.decode(answer))
        return decision if self._wants_decision else decision.value


class SemanticFunction[**P, T]:
    __name__: str
    __qualname__: str
    __wrapped__: Callable[P, Any]

    def __init__(
        self,
        function: Callable[P, Any],
        *,
        min_confidence: float | None = None,
        uncertain: Any = _MISSING,
        backend: Backend | None = None,
    ) -> None:
        name = getattr(function, "__qualname__", type(function).__name__)
        self.instructions = inspect.cleandoc(function.__doc__ or "")
        if not self.instructions:
            raise TypeError(f"{name} needs a semantic description in its docstring")
        hints = get_type_hints(function, include_extras=True)
        if "return" not in hints:
            raise TypeError(f"{name} needs a return annotation")
        if min_confidence is not None and not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"{name}: min_confidence must be between 0 and 1")
        if uncertain is not _MISSING and min_confidence is None:
            raise TypeError(f"{name}: uncertain has no effect without min_confidence")
        self.signature = inspect.signature(function)
        self.min_confidence = min_confidence
        self.uncertain = uncertain
        self._plan = plan_for(name, hints, self.instructions)
        self._own_runtime = SemanticRuntime(backend) if backend is not None else None
        self._bound_arguments: tuple[Any, ...] = ()
        self.cases: list[Case] = []
        update_wrapper(self, function)

    def __get__(
        self, instance: object, owner: type | None = None
    ) -> SemanticFunction[..., T]:
        if instance is None:
            return self
        bound = copy.copy(self)
        bound._bound_arguments = (instance,)
        return bound

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> SemanticCall[T]:
        """The decided value, once awaited."""
        return SemanticCall(self, self._arguments(*args, **kwargs), False)

    def result(self, *args: P.args, **kwargs: P.kwargs) -> SemanticCall[Decision[T]]:
        """The full `Decision` with confidence and distribution, once awaited."""
        return SemanticCall(self, self._arguments(*args, **kwargs), True)

    def eval(self, cases: Iterable[Case]) -> Evaluation:
        """Register labeled cases for `semfn eval`. Await the result to run them."""
        added = list(cases)
        self.cases.extend(added)
        return Evaluation(self, added)

    def _arguments(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        bound = self.signature.bind(*self._bound_arguments, *args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    def _runtime(self) -> SemanticRuntime:
        return current_runtime(self._own_runtime)

    def _prepare(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], Schema]:
        plan = self._plan
        schema = plan.schema
        if schema is None:
            choices = arguments[plan.choices_parameter or ""]
            values = list(choices) if choices is not None else []
            if not values and not plan.allows_none:
                raise ValueError(
                    f"{plan.choices_parameter} must contain at least one choice"
                )
            labels = [f"option_{index}" for index in range(len(values))]
            if plan.allows_none:
                values.append(None)
                labels.append(NONE_LABEL)
            schema = choice_schema(self.instructions, values, labels)
        state = {
            name: serialize(value)
            for name, value in arguments.items()
            if name != plan.choices_parameter
        }
        return state, schema

    def _apply_policy(self, decision: Decision[Any]) -> Decision[Any]:
        if (
            self.min_confidence is not None
            and decision.confidence < self.min_confidence
        ):
            if self.uncertain is _MISSING:
                raise UncertainDecision(decision, self.min_confidence)
            return Decision(self.uncertain, decision.confidence, decision.distribution)
        return decision


@overload
async def gather[R1](call1: SemanticCall[R1], /) -> tuple[R1]: ...
@overload
async def gather[R1, R2](
    call1: SemanticCall[R1], call2: SemanticCall[R2], /
) -> tuple[R1, R2]: ...
@overload
async def gather[R1, R2, R3](
    call1: SemanticCall[R1], call2: SemanticCall[R2], call3: SemanticCall[R3], /
) -> tuple[R1, R2, R3]: ...
@overload
async def gather[R1, R2, R3, R4](
    call1: SemanticCall[R1],
    call2: SemanticCall[R2],
    call3: SemanticCall[R3],
    call4: SemanticCall[R4],
    /,
) -> tuple[R1, R2, R3, R4]: ...
@overload
async def gather[R1, R2, R3, R4, R5](
    call1: SemanticCall[R1],
    call2: SemanticCall[R2],
    call3: SemanticCall[R3],
    call4: SemanticCall[R4],
    call5: SemanticCall[R5],
    /,
) -> tuple[R1, R2, R3, R4, R5]: ...
@overload
async def gather(*calls: SemanticCall[Any]) -> tuple[Any, ...]: ...


async def gather(*calls: SemanticCall[Any]) -> tuple[Any, ...]:
    """Resolve several calls, sharing one backend request per distinct state.

    Backends encode the state once per request, so questions about the same
    arguments are much cheaper together than one by one.
    """
    prepared = [call._function._prepare(call._arguments) for call in calls]
    groups: list[tuple[SemanticRuntime, dict[str, Any], list[int]]] = []
    for index, (call, (state, _)) in enumerate(zip(calls, prepared, strict=True)):
        runtime = call._function._runtime()
        for group_runtime, group_state, members in groups:
            if group_runtime is runtime and group_state == state:
                members.append(index)
                break
        else:
            groups.append((runtime, state, [index]))

    results: list[Any] = [None] * len(calls)

    async def resolve(
        runtime: SemanticRuntime, state: dict[str, Any], members: list[int]
    ) -> None:
        questions = {f"question_{i}": prepared[i][1].question for i in members}
        answers = await runtime.evaluate(state, questions)
        for i in members:
            results[i] = calls[i]._finish(prepared[i][1], answers[f"question_{i}"])

    await asyncio.gather(*(resolve(*group) for group in groups))
    return tuple(results)


class _Decorator[U](Protocol):
    def __call__[**P, T](
        self, function: Callable[P, T], /
    ) -> SemanticFunction[P, T | U]: ...


@overload
def semantic[**P, T](function: Callable[P, T], /) -> SemanticFunction[P, T]: ...
@overload
def semantic[U](
    *, min_confidence: float, uncertain: U, backend: Backend | None = None
) -> _Decorator[U]: ...
@overload
def semantic(
    *, min_confidence: float | None = None, backend: Backend | None = None
) -> _Decorator[Never]: ...


def semantic(
    function: Callable[..., Any] | None = None,
    /,
    *,
    min_confidence: float | None = None,
    uncertain: Any = _MISSING,
    backend: Backend | None = None,
) -> Any:
    """Turn a typed, documented function into a semantic function.

    The docstring is the question, the arguments are the state, and the return
    annotation defines the possible answers. The body never runs.

    Below `min_confidence` a call raises `UncertainDecision`, or returns
    `uncertain` when given. `backend` pins this function to its own backend.
    """

    def decorate(target: Callable[..., Any]) -> SemanticFunction[..., Any]:
        return SemanticFunction(
            target, min_confidence=min_confidence, uncertain=uncertain, backend=backend
        )

    return decorate(function) if function is not None else decorate
