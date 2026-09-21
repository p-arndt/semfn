from __future__ import annotations

import asyncio
from collections.abc import Mapping
from threading import Lock
from typing import Any


class LayaBackend:
    def __init__(
        self,
        model: str = "convaiinnovations/laya-multilingual",
        *,
        quiet: bool = True,
    ):
        self.model = model
        self.quiet = quiet
        self._agent: Any = None
        self._load_lock = Lock()

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

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    async def evaluate(
        self, state: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        return await asyncio.to_thread(
            self._load().predict, dict(state), dict(questions)
        )
