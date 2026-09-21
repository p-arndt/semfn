from __future__ import annotations

import asyncio
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


class LayaBackend:
    def __init__(
        self,
        model: str = "convaiinnovations/laya-multilingual",
        *,
        quiet: bool = True,
    ) -> None:
        self.model = model
        self.quiet = quiet
        self._agent: Any = None
        self._load_lock = Lock()
        self._predict_lock = Lock()

    def _load(self) -> Any:
        if self._agent is None:
            with self._load_lock:
                if self._agent is not None:
                    return self._agent

                import laya
                from huggingface_hub.utils import (
                    are_progress_bars_disabled,
                    disable_progress_bars,
                    enable_progress_bars,
                )

                progress_was_disabled = are_progress_bars_disabled()
                if self.quiet:
                    disable_progress_bars()
                try:
                    self._agent = laya.load(self.model)
                finally:
                    if self.quiet and not progress_was_disabled:
                        enable_progress_bars()
        return self._agent

    def _predict(
        self, state: dict[str, Any], questions: dict[str, Any]
    ) -> Mapping[str, Any]:
        agent = self._load()
        # One torch model is shared by all callers, and laya's predict
        # reassigns the agent's device when it falls back after an OOM.
        with self._predict_lock:
            return agent.predict(state, questions)

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    async def evaluate(
        self, state: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        return await asyncio.to_thread(self._predict, dict(state), dict(questions))
