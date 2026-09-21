from __future__ import annotations

import asyncio
import sys
import threading
import time
import unittest
from types import ModuleType
from typing import Any
from unittest.mock import patch

from huggingface_hub.utils import are_progress_bars_disabled, enable_progress_bars

from semfn.laya import LayaBackend


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self.active = 0
        self.max_active = 0
        self._counter_lock = threading.Lock()

    def predict(self, state: Any, questions: Any) -> dict[str, Any]:
        with self._counter_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        # Long enough for unserialized callers to overlap.
        time.sleep(0.02)
        with self._counter_lock:
            self.active -= 1
            self.calls.append((state, questions))
        return {"answers": dict.fromkeys(questions, "yes")}


class FakeLaya(ModuleType):
    def __init__(self) -> None:
        super().__init__("laya")
        self.agent = FakeAgent()
        self.load_models: list[str] = []
        self.load_threads: list[int] = []
        self.progress_disabled_during_load: list[bool] = []

    def load(self, model: str) -> FakeAgent:
        self.load_models.append(model)
        self.load_threads.append(threading.get_ident())
        self.progress_disabled_during_load.append(are_progress_bars_disabled())
        # Long enough for concurrent callers to race into the load.
        time.sleep(0.05)
        return self.agent


class LayaBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.laya = FakeLaya()
        modules = patch.dict(sys.modules, {"laya": self.laya})
        modules.start()
        self.addCleanup(modules.stop)
        enable_progress_bars()
        self.addCleanup(enable_progress_bars)

    async def test_evaluate_loads_off_the_event_loop_thread(self) -> None:
        backend = LayaBackend()

        await backend.evaluate({"a": 1}, {"q": {"type": "bool"}})

        self.assertEqual(len(self.laya.load_threads), 1)
        self.assertNotEqual(self.laya.load_threads[0], threading.get_ident())

    async def test_evaluate_keeps_the_event_loop_responsive_while_loading(
        self,
    ) -> None:
        backend = LayaBackend()
        ticks = 0

        async def tick() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.005)

        ticker = asyncio.create_task(tick())
        try:
            await backend.evaluate({}, {"q": {}})
        finally:
            ticker.cancel()

        self.assertGreater(ticks, 1)

    async def test_concurrent_evaluate_loads_the_model_once(self) -> None:
        backend = LayaBackend(model="some/model")

        await asyncio.gather(*(backend.evaluate({"i": i}, {"q": {}}) for i in range(5)))

        self.assertEqual(self.laya.load_models, ["some/model"])
        self.assertEqual(len(self.laya.agent.calls), 5)

    async def test_concurrent_evaluate_serializes_predict(self) -> None:
        backend = LayaBackend()

        await asyncio.gather(*(backend.evaluate({"i": i}, {"q": {}}) for i in range(5)))

        self.assertEqual(self.laya.agent.max_active, 1)

    async def test_warmup_loads_the_model(self) -> None:
        backend = LayaBackend()

        await backend.warmup()
        await backend.evaluate({}, {"q": {}})

        self.assertEqual(len(self.laya.load_models), 1)
        self.assertNotEqual(self.laya.load_threads[0], threading.get_ident())

    async def test_evaluate_passes_plain_dicts_and_returns_predict_result(
        self,
    ) -> None:
        backend = LayaBackend()
        state = {"name": "demo"}
        questions = {"urgent": {"type": "bool"}}

        result = await backend.evaluate(state, questions)

        self.assertEqual(result, {"answers": {"urgent": "yes"}})
        passed_state, passed_questions = self.laya.agent.calls[0]
        self.assertIs(type(passed_state), dict)
        self.assertIs(type(passed_questions), dict)
        self.assertEqual(passed_state, state)
        self.assertEqual(passed_questions, questions)
        self.assertIsNot(passed_state, state)
        self.assertIsNot(passed_questions, questions)

    async def test_quiet_load_restores_progress_bars(self) -> None:
        backend = LayaBackend(quiet=True)

        await backend.warmup()

        self.assertEqual(self.laya.progress_disabled_during_load, [True])
        self.assertFalse(are_progress_bars_disabled())

    async def test_quiet_load_restores_progress_bars_when_load_fails(self) -> None:
        backend = LayaBackend(quiet=True)

        with (
            patch.object(self.laya, "load", side_effect=RuntimeError("boom")),
            self.assertRaises(RuntimeError),
        ):
            await backend.warmup()

        self.assertFalse(are_progress_bars_disabled())

    async def test_non_quiet_load_leaves_progress_bars_enabled(self) -> None:
        backend = LayaBackend(quiet=False)

        await backend.warmup()

        self.assertEqual(self.laya.progress_disabled_during_load, [False])


if __name__ == "__main__":
    unittest.main()
