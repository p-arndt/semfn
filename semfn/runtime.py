from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator, Mapping


class Backend(Protocol):
    """Answers a set of questions about one shared state.

    `evaluate` returns one answer per question id, optionally nested under an
    `"answers"` key. Expected answer fields per question type:

    - `noul`: `{"noul": <probability of yes>}`
    - `choice`: `{"choice": <label>}` and/or `{"probabilities": {<label>: p}}`
    - `score`: `{"probabilities": {<level index>: p}}` and/or `{"score": <float>}`
    """

    def evaluate(
        self, state: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any] | Awaitable[Mapping[str, Any]]: ...


class SemanticRuntime:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    async def evaluate(
        self, state: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Mapping[str, Any]]:
        response = self.backend.evaluate(state, questions)
        if inspect.isawaitable(response):
            response = await response
        return cast(
            "Mapping[str, Mapping[str, Any]]", response.get("answers", response)
        )

    async def warmup(self) -> None:
        """Load backend resources before serving the first request."""
        warmup = getattr(self.backend, "warmup", None)
        if warmup is not None:
            result = warmup()
            if inspect.isawaitable(result):
                await result


class _Defaults:
    runtime: SemanticRuntime | None = None


_scoped_runtime: ContextVar[SemanticRuntime | None] = ContextVar(
    "semfn_runtime", default=None
)


def configure(
    *,
    backend: str | Backend = "laya",
    model: str | None = None,
    quiet: bool = True,
) -> SemanticRuntime:
    """Set the process-wide default backend."""
    if backend == "laya":
        from .laya import LayaBackend

        backend = LayaBackend(
            model or "convaiinnovations/laya-multilingual", quiet=quiet
        )
    elif backend == "jev":
        from .jev import JevBackend

        backend = JevBackend(model or "jev-latest")
    elif isinstance(backend, str):
        raise ValueError(f"Unknown backend {backend!r}")
    _Defaults.runtime = SemanticRuntime(backend)
    return _Defaults.runtime


@contextmanager
def using(backend: Backend) -> Iterator[SemanticRuntime]:
    """Route every semantic call in this context to `backend`.

    Scoped to the current task, so tests and request handlers can swap the
    backend without touching the process-wide default.
    """
    runtime = SemanticRuntime(backend)
    token = _scoped_runtime.set(runtime)
    try:
        yield runtime
    finally:
        _scoped_runtime.reset(token)


def current_runtime(own: SemanticRuntime | None) -> SemanticRuntime:
    """The runtime for a call: `using` scope first, then the function's own."""
    return _scoped_runtime.get() or own or _Defaults.runtime or configure()
