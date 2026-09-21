"""Static guarantees of the public API; `ty check` is the real assertion here."""

from typing import Literal, assert_type

import semfn

type Kind = Literal["bug", "feature"]
Level = Literal["low", "high"]


@semfn.semantic
def classify(text: str) -> Kind:
    """Classify the request."""
    raise NotImplementedError


@semfn.semantic(min_confidence=0.8, uncertain=None)
def lenient(text: str) -> bool:
    """Is this certain?"""
    raise NotImplementedError


@semfn.semantic(min_confidence=0.8)
def strict(text: str) -> bool:
    """Is this certain?"""
    raise NotImplementedError


@semfn.semantic
def level(text: str) -> semfn.Score[Level]:
    """How much?"""
    raise NotImplementedError


async def check_types() -> None:
    assert_type(await classify("x"), Kind)
    assert_type(await classify.result("x"), semfn.Decision[Kind])
    assert_type(await lenient("x"), bool | None)
    assert_type(await strict("x"), bool)
    assert_type(await level("x"), Level)
    kind, decision = await semfn.gather(classify("x"), strict.result("x"))
    assert_type(kind, Kind)
    assert_type(decision, semfn.Decision[bool])
    classify(1)  # ty: ignore[invalid-argument-type]
