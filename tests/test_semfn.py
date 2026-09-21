import unittest
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import semfn


class FakeBackend:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def evaluate(self, state, questions):
        self.calls.append((state, questions))
        answers = self.answers[: len(questions)]
        del self.answers[: len(questions)]
        return {"answers": dict(zip(questions, answers))}


class Severity(str, Enum):
    LOW = "low"
    HIGH = "high"


@dataclass
class Project:
    id: str
    name: str


class SemfnTests(unittest.IsolatedAsyncioTestCase):
    async def test_bool_value_and_result(self):
        backend = FakeBackend([{"noul": 0.91}, {"noul": 0.2}])
        semfn.configure(backend=backend)

        @semfn.semantic
        def blocking(text: str) -> bool:
            """Is the user blocked?"""
            raise NotImplementedError

        self.assertTrue(await blocking("Cannot log in"))
        result = await blocking.result("Works now")
        self.assertFalse(result.value)
        self.assertAlmostEqual(result.confidence, 0.8)
        self.assertEqual(backend.calls[0][0], {"text": "Cannot log in"})

    async def test_literal_enum_and_optional_object(self):
        backend = FakeBackend(
            [
                {"choice": "feature", "probabilities": {"bug": 0.1, "feature": 0.9}},
                {"choice": "high", "probabilities": {"low": 0.2, "high": 0.8}},
                {
                    "choice": "option_1",
                    "probabilities": {"option_0": 0.1, "option_1": 0.85, "none": 0.05},
                },
            ]
        )
        semfn.configure(backend=backend)

        @semfn.semantic
        def classify(text: str) -> Literal["bug", "feature"]:
            """Classify the request."""
            raise NotImplementedError

        @semfn.semantic
        def severity(text: str) -> Severity:
            """How severe is this?"""
            raise NotImplementedError

        @semfn.semantic
        def project_for(text: str, projects: list[Project]) -> Project | None:
            """Which project fits best?"""

        self.assertEqual(await classify("Please add dark mode"), "feature")
        self.assertEqual(await severity("Data loss"), Severity.HIGH)
        projects = [Project("a", "Alpha"), Project("b", "Beta")]
        self.assertIs(await project_for("Beta task", projects), projects[1])
        state, questions = backend.calls[2]
        self.assertEqual(state, {"text": "Beta task"})
        self.assertIn('"name": "Beta"', questions["decision"]["criteria"]["option_1"])

    async def test_confidence_policy(self):
        backend = FakeBackend([{"noul": 0.6}, {"noul": 0.6}])
        semfn.configure(backend=backend)

        @semfn.semantic(min_confidence=0.8)
        def strict(text: str) -> bool:
            """Is this certain?"""
            raise NotImplementedError

        @semfn.semantic(min_confidence=0.8, uncertain=None)
        def lenient(text: str) -> bool:
            """Is this certain?"""
            raise NotImplementedError

        with self.assertRaises(semfn.UncertainDecision):
            await strict("maybe")
        self.assertIsNone(await lenient("maybe"))

    async def test_score(self):
        backend = FakeBackend([{"score": 2.7, "confidence": 0.75}])
        semfn.configure(backend=backend)

        @semfn.semantic
        def urgency(
            text: str,
        ) -> semfn.Score[Literal["irrelevant", "later", "soon", "immediate"]]:
            """How urgently does this require action?"""
            raise NotImplementedError

        self.assertEqual(await urgency("Production is down"), "immediate")

    async def test_batch_uses_one_backend_call(self):
        backend = FakeBackend(
            [
                {"choice": "bug", "probabilities": {"bug": 0.9, "feature": 0.1}},
                {"noul": 0.8},
            ]
        )
        semfn.configure(backend=backend)

        @semfn.semantic
        def classify(text: str) -> Literal["bug", "feature"]:
            """Classify the request."""
            raise NotImplementedError

        @semfn.semantic
        def blocking(text: str) -> bool:
            """Is the user blocked?"""
            raise NotImplementedError

        async with semfn.semantic.batch() as pending:
            classify("Login loops")
            blocking("Login loops")

        self.assertEqual(await pending.resolve(), ("bug", True))
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(backend.calls[0][1]), 2)


if __name__ == "__main__":
    unittest.main()
