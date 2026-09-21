from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
)

from .decision import Decision

NONE_LABEL = "none"


class _Ordinal:
    pass


type Score[Levels: str] = Annotated[Levels, _Ordinal()]
"""Ordered levels from lowest to highest, e.g. `Score[Literal["low", "high"]]`."""


@dataclass(frozen=True, slots=True)
class Schema:
    question: dict[str, Any]
    decode: Callable[[Mapping[str, Any]], Decision[Any]]


def _unalias(annotation: Any) -> Any:
    while isinstance(annotation, TypeAliasType):
        annotation = annotation.__value__
    return annotation


def _optional(annotation: Any) -> tuple[Any, bool]:
    annotation = _unalias(annotation)
    if get_origin(annotation) in (Union, UnionType):
        args = get_args(annotation)
        remaining = tuple(arg for arg in args if arg is not type(None))
        if len(remaining) == 1 and len(remaining) != len(args):
            return _unalias(remaining[0]), True
    return annotation, False


def serialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): serialize(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [serialize(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: serialize(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _describe(value: Any) -> str:
    """Render an option as plain text for the model.

    Measured against Laya: JSON syntax and field names drown out the content
    (the right project scored 0.41 as JSON, 0.74 as bare values), and any
    wording for the none option other than "none" attracts most of the mass.
    """
    return _plain(serialize(value))


def _plain(serialized: Any) -> str:
    if serialized is None:
        return NONE_LABEL
    if isinstance(serialized, str):
        return serialized
    if isinstance(serialized, Mapping):
        return " - ".join(_plain(item) for item in serialized.values())
    if isinstance(serialized, list):
        return ", ".join(_plain(item) for item in serialized)
    return json.dumps(serialized)


def _describe_member(member: Enum) -> str:
    # A bare non-string value like `1` tells the model nothing; the member
    # name carries the meaning.
    if isinstance(member.value, str):
        return member.value
    return f"{member.name} ({_describe(member.value)})"


def _probabilities(answer: Mapping[str, Any]) -> dict[str, float]:
    raw = answer.get("probabilities", answer.get("distribution", {}))
    return {str(key): float(value) for key, value in raw.items()}


def _fallback_confidence(answer: Mapping[str, Any]) -> float:
    # A backend that reports no probabilities gave a deterministic answer.
    return float(answer.get("confidence", 1.0))


def choice_schema(
    instructions: str,
    values: Sequence[Any],
    labels: Sequence[str],
    descriptions: Sequence[str] | None = None,
) -> Schema:
    if len(set(labels)) != len(labels):
        raise TypeError(f"Choice labels must be unique, got {list(labels)!r}")
    if descriptions is None:
        descriptions = [_describe(value) for value in values]
    question = {
        "type": "choice",
        "instructions": instructions,
        "criteria": dict(zip(labels, descriptions, strict=True)),
    }

    def decode(answer: Mapping[str, Any]) -> Decision[Any]:
        probabilities = _probabilities(answer)
        if "choice" in answer:
            selected = str(answer["choice"])
        elif probabilities:
            selected = max(probabilities, key=lambda label: probabilities[label])
        else:
            raise ValueError("Backend answer has neither a choice nor probabilities")
        if selected not in labels:
            raise ValueError(f"Backend returned unknown choice {selected!r}")
        if not probabilities:
            probabilities = {selected: _fallback_confidence(answer)}
        distribution = tuple(
            (value, probabilities.get(label, 0.0))
            for value, label in zip(values, labels, strict=True)
        )
        return Decision(
            values[labels.index(selected)], probabilities[selected], distribution
        )

    return Schema(question, decode)


def _bool_schema(instructions: str) -> Schema:
    def decode(answer: Mapping[str, Any]) -> Decision[bool]:
        probability = float(answer["noul"])
        value = probability >= 0.5
        return Decision(
            value,
            probability if value else 1 - probability,
            ((True, probability), (False, 1 - probability)),
        )

    return Schema({"type": "noul", "instructions": instructions}, decode)


def _score_schema(instructions: str, levels: tuple[str, ...]) -> Schema:
    def decode(answer: Mapping[str, Any]) -> Decision[str]:
        probabilities = _probabilities(answer)
        by_level = [
            probabilities.get(str(index), probabilities.get(level, 0.0))
            for index, level in enumerate(levels)
        ]
        if probabilities:
            # The most likely level, not the rounded expectation: a split
            # distribution would otherwise select a level nobody voted for.
            selected = max(range(len(levels)), key=lambda index: by_level[index])
        else:
            rounded = math.floor(float(answer["score"]) + 0.5)
            selected = min(len(levels) - 1, max(0, rounded))
            by_level[selected] = _fallback_confidence(answer)
        return Decision(
            levels[selected],
            by_level[selected],
            tuple(zip(levels, by_level, strict=True)),
        )

    return Schema(
        {"type": "score", "instructions": instructions, "criteria": list(levels)},
        decode,
    )


def _score_levels(annotation: Any) -> tuple[str, ...] | None:
    if get_origin(annotation) is not Score:
        return None
    levels = _unalias(get_args(annotation)[0])
    values = get_args(levels) if get_origin(levels) is Literal else ()
    if len(values) < 2 or not all(isinstance(v, str) and v for v in values):
        raise TypeError("Score needs a Literal of at least two non-empty strings")
    return values


def _is_sequence_of(annotation: Any, item: Any) -> bool:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (list, Sequence):
        return len(args) == 1 and _unalias(args[0]) == item
    return (
        origin is tuple
        and len(args) == 2
        and (_unalias(args[0]), args[1]) == (item, ...)
    )


@dataclass(frozen=True, slots=True)
class Plan:
    """How a return annotation maps to a backend question.

    Either the options are known from the annotation (`schema`), or they arrive
    per call in the `choices_parameter` argument.
    """

    schema: Schema | None = None
    choices_parameter: str | None = None
    allows_none: bool = False


def plan_for(name: str, hints: Mapping[str, Any], instructions: str) -> Plan:
    annotation, allows_none = _optional(hints["return"])

    if annotation is bool:
        if allows_none:
            return Plan(
                choice_schema(
                    instructions, [True, False, None], ["true", "false", NONE_LABEL]
                )
            )
        return Plan(_bool_schema(instructions))

    if (levels := _score_levels(annotation)) is not None:
        if allows_none:
            raise TypeError(
                f"{name}: Score cannot be optional; use min_confidence with "
                "uncertain=None instead"
            )
        return Plan(_score_schema(instructions, levels))

    if get_origin(annotation) is Literal:
        values = [*get_args(annotation)]
        labels = [
            value if isinstance(value, str) else f"option_{index}"
            for index, value in enumerate(values)
        ]
        if allows_none:
            values.append(None)
            labels.append(NONE_LABEL)
        return Plan(choice_schema(instructions, values, labels))

    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        members: list[Enum | None] = list(annotation)
        labels = [
            member.value if isinstance(member.value, str) else member.name
            for member in annotation
        ]
        descriptions = [_describe_member(member) for member in annotation]
        if allows_none:
            members.append(None)
            labels.append(NONE_LABEL)
            descriptions.append(_describe(None))
        return Plan(choice_schema(instructions, members, labels, descriptions))

    candidates = [
        parameter
        for parameter, hint in hints.items()
        if parameter != "return" and _is_sequence_of(_optional(hint)[0], annotation)
    ]
    if len(candidates) == 1:
        return Plan(choices_parameter=candidates[0], allows_none=allows_none)
    if candidates:
        raise TypeError(
            f"{name}: several parameters could hold the choices: {candidates!r}"
        )
    raise TypeError(
        f"{name}: unsupported semantic return type {annotation!r}; return bool, "
        "Literal, an Enum, Score[...], or T together with a list[T] parameter"
    )
