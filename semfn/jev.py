from __future__ import annotations

import asyncio
from importlib import import_module
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


class JevBackend:
    """Hosted TypeSafe Jev backend.

    The optional ``typesafe-sdk`` dependency reads ``TYPESAFE_API_KEY`` from
    the environment. A client can be supplied for tests or custom SDK config.
    """

    def __init__(self, model: str = "jev-latest", *, client: Any = None) -> None:
        self.model = model
        self._client = client
        self._load_lock = Lock()

    def _load(self) -> Any:
        if self._client is None:
            with self._load_lock:
                if self._client is None:
                    try:
                        client_type = import_module("typesafe_sdk").TypeSafeClient
                    except ImportError as error:
                        raise ImportError(
                            'Jev support requires the "jev" extra: '
                            'pip install "semfn[jev]"'
                        ) from error
                    self._client = client_type()
        return self._client

    def _evaluate(
        self, state: dict[str, Any], questions: dict[str, Any]
    ) -> Mapping[str, Any]:
        response = self._load().system_one(
            state=state, questions=questions, model=self.model
        )
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return response

    async def evaluate(
        self, state: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        return await asyncio.to_thread(self._evaluate, dict(state), dict(questions))
