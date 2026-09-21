from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from types import UnionType
from typing import (
    Any,
    Literal,
    Protocol,
    Self,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

T = TypeVar("T")
_MISSING = object()


class Backend(Protocol):
    def evaluate(
        self, state: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any] | Awaitable[Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class Decision[T]:
    value: T
    confidence: float
    distribution: Mapping[Any, float]


class UncertainDecision(Exception):
    def __init__(self, decision: Decision[Any], minimum: float):
        self.decision = decision
        self.minimum = minimum
        super().__init__(
            f"Decision confidence {decision.confidence:.3f} is below {minimum:.3f}"
        )


@dataclass(frozen=True, slots=True)
class _ScoreType:
    levels: tuple[str, ...]


class Score[*Levels]:
    def __class_getitem__(cls, levels: object) -> _ScoreType:
        if get_origin(levels) is Literal:
            levels = get_args(levels)
        elif isinstance(levels, str) or not isinstance(levels, tuple):
            levels = (levels,)
        if not levels or not all(isinstance(level, str) and level for level in levels):
            raise TypeError("Score levels must be non-empty strings")
        return _ScoreType(cast(tuple[str, ...], levels))


@dataclass(slots=True)
class _Schema:
    question: dict[str, Any]
    decode: Callable[[Mapping[str, Any]], Decision[Any]]


def _optional(annotation: Any) -> tuple[Any, bool]:
    if get_origin(annotation) in (Union, UnionType):
        args = get_args(annotation)
        remaining = tuple(arg for arg in args if arg is not type(None))
        if len(remaining) == 1 and len(remaining) != len(args):
            return remaining[0], True
    return annotation, False


def _serialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialize(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: _serialize(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _describe(value: Any) -> str:
    serialized = _serialize(value)
    return (
        serialized
        if isinstance(serialized, str)
        else json.dumps(serialized, ensure_ascii=False)
    )


def _probabilities(answer: Mapping[str, Any]) -> dict[str, float]:
    raw = answer.get("probabilities", answer.get("distribution", {}))
    return {str(key): float(value) for key, value in raw.items()}


def _distribution_key(value: Any, fallback: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    candidate = getattr(value, "id", fallback)
    try:
        hash(candidate)
    except TypeError:
        return fallback
    return candidate


def _choice_schema(instructions: str, values: list[Any], labels: list[str]) -> _Schema:
    question = {
        "type": "choice",
        "instructions": instructions,
        "criteria": {label: _describe(value) for label, value in zip(labels, values)},
    }

    def decode(answer: Mapping[str, Any]) -> Decision[Any]:
        distribution = _probabilities(answer)
        selected = str(
            answer.get("choice", max(distribution, key=lambda key: distribution[key]))
        )
        if selected not in labels:
            raise ValueError(f"Backend returned unknown choice {selected!r}")
        value = values[labels.index(selected)]
        confidence = float(answer.get("confidence", distribution.get(selected, 0.0)))
        public_distribution = {
            _distribution_key(item, label): distribution.get(label, 0.0)
            for item, label in zip(values, labels)
        }
        return Decision(value, confidence, public_distribution)

    return _Schema(question, decode)


def _schema_for(
    annotation: Any, instructions: str, arguments: Mapping[str, Any]
) -> tuple[_Schema, str | None]:
    annotation, allows_none = _optional(annotation)

    if annotation is bool and not allows_none:

        def decode_bool(answer: Mapping[str, Any]) -> Decision[bool]:
            probability = float(answer["noul"])
            value = probability >= 0.5
            return Decision(
                value,
                probability if value else 1 - probability,
                {True: probability, False: 1 - probability},
            )

        return _Schema(
            {"type": "noul", "instructions": instructions}, decode_bool
        ), None

    if isinstance(annotation, _ScoreType):
        levels = annotation.levels

        def decode(answer: Mapping[str, Any]) -> Decision[str]:
            distribution = _probabilities(answer)
            score = float(answer["score"])
            index = min(len(levels) - 1, max(0, round(score)))
            confidence = float(
                answer.get("confidence", max(distribution.values(), default=0.0))
            )
            mapped = {
                level: distribution.get(str(i), distribution.get(level, 0.0))
                for i, level in enumerate(levels)
            }
            return Decision(levels[index], confidence, mapped)

        return _Schema(
            {"type": "score", "instructions": instructions, "criteria": list(levels)},
            decode,
        ), None

    if get_origin(annotation) is Literal:
        values = list(get_args(annotation))
        if allows_none:
            values.append(None)
        labels = [
            value if isinstance(value, str) else f"option_{index}"
            for index, value in enumerate(values)
        ]
        return _choice_schema(instructions, values, labels), None

    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        values: list[Any] = list(annotation)
        if allows_none:
            values.append(None)
        labels = [
            str(value.value) if isinstance(value, Enum) else "none" for value in values
        ]
        return _choice_schema(instructions, values, labels), None

    for name, value in arguments.items():
        if isinstance(value, list) and all(
            isinstance(item, annotation) for item in value
        ):
            if not value and not allows_none:
                raise ValueError(f"{name} must contain at least one choice")
            values = list(value)
            labels = [f"option_{index}" for index in range(len(values))]
            if allows_none:
                values.append(None)
                labels.append("none")
            return _choice_schema(instructions, values, labels), name

    if annotation is bool and allows_none:
        return _choice_schema(
            instructions, [True, False, None], ["true", "false", "none"]
        ), None
    raise TypeError(f"Unsupported semantic return type: {annotation!r}")


class SemanticRuntime:
    def __init__(self, backend: Backend):
        self.backend = backend

    async def evaluate(
        self, state: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        response = self.backend.evaluate(state, questions)
        if inspect.isawaitable(response):
            response = await response
        return response.get("answers", response)  # type: ignore[return-value]

    async def warmup(self) -> None:
        """Load backend resources before serving the first request."""
        warmup = getattr(self.backend, "warmup", None)
        if warmup is not None:
            result = warmup()
            if inspect.isawaitable(result):
                await result


_runtime: SemanticRuntime | None = None


def configure(
    *,
    backend: str | Backend = "laya",
    model: str = "convaiinnovations/laya-multilingual",
    quiet: bool = True,
) -> SemanticRuntime:
    global _runtime
    if backend == "laya":
        from .laya import LayaBackend

        backend = LayaBackend(model, quiet=quiet)
    elif isinstance(backend, str):
        raise ValueError(f"Unknown backend {backend!r}")
    _runtime = SemanticRuntime(backend)
    return _runtime


def _configured_runtime() -> SemanticRuntime:
    global _runtime
    if _runtime is None:
        _runtime = configure()
    return _runtime


@dataclass(slots=True)
class _Call:
    function: SemanticFunction[Any]
    arguments: dict[str, Any]
    wants_decision: bool


class Batch:
    def __init__(self):
        self._calls: list[_Call] = []
        self._token: Any = None

    async def __aenter__(self) -> Self:
        self._token = _current_batch.set(self)
        return self

    async def __aexit__(self, *_: object) -> None:
        _current_batch.reset(self._token)

    async def resolve(self) -> tuple[Any, ...]:
        if not self._calls:
            return ()
        prepared = [call.function._prepare(call.arguments) for call in self._calls]
        states = [item[0] for item in prepared]
        if any(state != states[0] for state in states[1:]):
            raise ValueError("All calls in a batch must serialize to the same state")
        questions = {
            f"question_{i}": item[1].question for i, item in enumerate(prepared)
        }
        answers = await _configured_runtime().evaluate(states[0], questions)
        results = []
        for i, (call, (_, schema)) in enumerate(zip(self._calls, prepared)):
            decision = call.function._apply_policy(
                schema.decode(answers[f"question_{i}"])
            )
            results.append(decision if call.wants_decision else decision.value)
        return tuple(results)


_current_batch: ContextVar[Batch | None] = ContextVar("semfn_batch", default=None)


class _BatchFactory:
    def __call__(self) -> Batch:
        return Batch()


batch = _BatchFactory()


class SemanticFunction[T]:
    def __init__(
        self,
        function: Callable[..., T],
        *,
        min_confidence: float | None,
        uncertain: Any,
    ):
        self.function = function
        self.signature = inspect.signature(function)
        self.hints = get_type_hints(function)
        self.instructions = inspect.cleandoc(function.__doc__ or "")
        function_name = getattr(function, "__name__", type(function).__name__)
        if not self.instructions:
            raise TypeError(
                f"{function_name} needs a semantic description in its docstring"
            )
        if "return" not in self.hints:
            raise TypeError(f"{function_name} needs a return annotation")
        self.min_confidence = min_confidence
        self.uncertain = uncertain
        self.__name__ = function_name
        self.__doc__ = function.__doc__
        self.__wrapped__ = function

    def _arguments(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    def _prepare(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], _Schema]:
        schema, choices_argument = _schema_for(
            self.hints["return"], self.instructions, arguments
        )
        state = {
            name: _serialize(value)
            for name, value in arguments.items()
            if name != choices_argument
        }
        return state, schema

    def _apply_policy(self, decision: Decision[T]) -> Decision[T]:
        if (
            self.min_confidence is not None
            and decision.confidence < self.min_confidence
        ):
            if self.uncertain is _MISSING:
                raise UncertainDecision(decision, self.min_confidence)
            return Decision(self.uncertain, decision.confidence, decision.distribution)
        return decision

    def __call__(self, *args: Any, **kwargs: Any) -> Awaitable[T]:
        return cast(
            Awaitable[T], self._schedule(self._arguments(*args, **kwargs), False)
        )

    def result(self, *args: Any, **kwargs: Any) -> Awaitable[Decision[T]]:
        return cast(
            Awaitable[Decision[T]],
            self._schedule(self._arguments(*args, **kwargs), True),
        )

    def _schedule(self, arguments: dict[str, Any], wants_decision: bool) -> Any:
        if active := _current_batch.get():
            active._calls.append(_Call(self, arguments, wants_decision))
            return None
        return self._evaluate(arguments, wants_decision)

    async def _evaluate(self, arguments: dict[str, Any], wants_decision: bool) -> Any:
        state, schema = self._prepare(arguments)
        answers = await _configured_runtime().evaluate(
            state, {"decision": schema.question}
        )
        decision = self._apply_policy(schema.decode(answers["decision"]))
        return decision if wants_decision else decision.value


@overload
def _semantic[T](
    function: Callable[..., T],
    /,
    *,
    min_confidence: float | None = None,
    uncertain: Any = _MISSING,
) -> SemanticFunction[T]: ...


@overload
def _semantic[T](
    function: None = None,
    /,
    *,
    min_confidence: float | None = None,
    uncertain: Any = _MISSING,
) -> Callable[[Callable[..., T]], SemanticFunction[T]]: ...


def _semantic[T](
    function: Callable[..., T] | None = None,
    *,
    min_confidence: float | None = None,
    uncertain: Any = _MISSING,
) -> SemanticFunction[T] | Callable[[Callable[..., T]], SemanticFunction[T]]:
    def decorate(target: Callable[..., T]) -> SemanticFunction[T]:
        return SemanticFunction(
            target, min_confidence=min_confidence, uncertain=uncertain
        )

    return decorate(function) if function is not None else decorate


class _SemanticDecorator(Protocol):
    batch: _BatchFactory

    @overload
    def __call__[T](
        self,
        function: Callable[..., T],
        /,
        *,
        min_confidence: float | None = None,
        uncertain: Any = _MISSING,
    ) -> SemanticFunction[T]: ...

    @overload
    def __call__[T](
        self,
        function: None = None,
        /,
        *,
        min_confidence: float | None = None,
        uncertain: Any = _MISSING,
    ) -> Callable[[Callable[..., T]], SemanticFunction[T]]: ...


semantic: _SemanticDecorator = cast(_SemanticDecorator, _semantic)
semantic.batch = batch
