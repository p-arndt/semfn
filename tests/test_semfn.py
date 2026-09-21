import asyncio
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Literal

import semfn


class FakeBackend:
    """Hands out queued answers in question order and records every request."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def evaluate(self, state, questions):
        self.calls.append((state, questions))
        answers = self.answers[: len(questions)]
        del self.answers[: len(questions)]
        return {"answers": dict(zip(questions, answers, strict=True))}


class Severity(StrEnum):
    LOW = "low"
    HIGH = "high"


class Priority(Enum):
    LATER = 1
    NOW = 2


@dataclass
class Project:
    id: str
    name: str


@semfn.semantic
def classify(text: str) -> Literal["bug", "feature"]:
    """Classify the request."""
    raise NotImplementedError


@semfn.semantic
def blocking(text: str) -> bool:
    """Is the user blocked?"""
    raise NotImplementedError


@semfn.semantic
def project_for(text: str, projects: list[Project]) -> Project | None:
    """Which project fits best?"""
    raise NotImplementedError


@semfn.semantic
def urgency(text: str) -> semfn.Score[Literal["later", "soon", "immediate"]]:
    """How urgently does this require action?"""
    raise NotImplementedError


class ReturnTypeTests(unittest.IsolatedAsyncioTestCase):
    async def test_bool_value_and_result(self):
        backend = FakeBackend({"noul": 0.91}, {"noul": 0.2})
        with semfn.using(backend):
            self.assertIs(await blocking("Cannot log in"), True)
            result = await blocking.result("Works now")
        self.assertIs(result.value, False)
        self.assertAlmostEqual(result.confidence, 0.8)
        self.assertAlmostEqual(result.probability(True), 0.2)
        self.assertEqual(backend.calls[0][0], {"text": "Cannot log in"})
        self.assertEqual(backend.calls[0][1]["question_0"]["type"], "noul")

    async def test_optional_bool_offers_none(self):
        @semfn.semantic
        def approved(text: str) -> bool | None:
            """Was the request approved?"""
            raise NotImplementedError

        backend = FakeBackend({"choice": "none", "probabilities": {"none": 0.7}})
        with semfn.using(backend):
            self.assertIsNone(await approved("No decision yet"))
        criteria = backend.calls[0][1]["question_0"]["criteria"]
        self.assertEqual(list(criteria), ["true", "false", "none"])

    async def test_literal(self):
        backend = FakeBackend(
            {"choice": "feature", "probabilities": {"bug": 0.1, "feature": 0.9}}
        )
        with semfn.using(backend):
            result = await classify.result("Please add dark mode")
        self.assertEqual(result.value, "feature")
        self.assertEqual(result.distribution, (("bug", 0.1), ("feature", 0.9)))

    async def test_confidence_is_the_selected_probability(self):
        # Laya reports an entropy-based confidence for choices; a threshold
        # must still mean the same thing as it does for bool.
        backend = FakeBackend(
            {
                "choice": "bug",
                "probabilities": {"bug": 0.8, "feature": 0.2},
                "confidence": 0.28,
            }
        )
        with semfn.using(backend):
            result = await classify.result("Login loops")
        self.assertAlmostEqual(result.confidence, 0.8)

    async def test_choice_without_probabilities(self):
        with semfn.using(FakeBackend({"choice": "bug"})):
            result = await classify.result("Login loops")
        self.assertEqual(result.value, "bug")
        self.assertEqual(result.confidence, 1.0)

    async def test_probabilities_without_choice(self):
        with semfn.using(FakeBackend({"probabilities": {"bug": 0.3, "feature": 0.7}})):
            self.assertEqual(await classify("Add dark mode"), "feature")

    async def test_unknown_choice_is_rejected(self):
        with (
            semfn.using(FakeBackend({"choice": "spam"})),
            self.assertRaisesRegex(ValueError, "unknown choice"),
        ):
            await classify("x")

    async def test_string_enum(self):
        @semfn.semantic
        def severity(text: str) -> Severity:
            """How severe is this?"""
            raise NotImplementedError

        backend = FakeBackend(
            {"choice": "high", "probabilities": {"low": 0.2, "high": 0.8}}
        )
        with semfn.using(backend):
            self.assertIs(await severity("Data loss"), Severity.HIGH)

    async def test_non_string_enum_is_described_by_name(self):
        @semfn.semantic
        def priority(text: str) -> Priority:
            """When should this be handled?"""
            raise NotImplementedError

        backend = FakeBackend({"choice": "NOW"})
        with semfn.using(backend):
            self.assertIs(await priority("Production is down"), Priority.NOW)
        criteria = backend.calls[0][1]["question_0"]["criteria"]
        self.assertEqual(criteria, {"LATER": "LATER (1)", "NOW": "NOW (2)"})

    async def test_score_picks_the_most_likely_level(self):
        backend = FakeBackend(
            {"score": 1.0, "probabilities": {"0": 0.5, "1": 0.0, "2": 0.5001}}
        )
        with semfn.using(backend):
            result = await urgency.result("Production is down")
        self.assertEqual(result.value, "immediate")
        self.assertAlmostEqual(result.confidence, 0.5001)
        self.assertEqual(
            result.distribution, (("later", 0.5), ("soon", 0.0), ("immediate", 0.5001))
        )

    async def test_score_without_probabilities_rounds_half_up(self):
        with semfn.using(FakeBackend({"score": 0.5}, {"score": 7})):
            self.assertEqual(await urgency("a"), "soon")
            self.assertEqual(await urgency("b"), "immediate")


class ChoiceParameterTests(unittest.IsolatedAsyncioTestCase):
    async def test_selects_the_object_and_keeps_choices_out_of_the_state(self):
        backend = FakeBackend(
            {
                "choice": "option_1",
                "probabilities": {"option_0": 0.1, "option_1": 0.85, "none": 0.05},
            }
        )
        projects = [Project("a", "Alpha"), Project("b", "Beta")]
        with semfn.using(backend):
            result = await project_for.result("Beta task", projects)
        self.assertIs(result.value, projects[1])
        self.assertEqual(result.probability(projects[0]), 0.1)
        self.assertEqual(result.probability(None), 0.05)
        state, questions = backend.calls[0]
        self.assertEqual(state, {"text": "Beta task"})
        self.assertEqual(
            questions["question_0"]["criteria"],
            {"option_0": "a - Alpha", "option_1": "b - Beta", "none": "none"},
        )

    async def test_choices_come_from_the_annotated_parameter(self):
        @semfn.semantic
        def tag_for(text: str, history: list[int], tags: tuple[str, ...]) -> str:
            """Which tag fits best?"""
            raise NotImplementedError

        backend = FakeBackend({"choice": "option_1"})
        with semfn.using(backend):
            self.assertEqual(await tag_for("x", [], ("a", "b")), "b")
        self.assertEqual(backend.calls[0][0], {"text": "x", "history": []})

    async def test_empty_choices(self):
        @semfn.semantic
        def tag_for(text: str, tags: list[str]) -> str:
            """Which tag fits best?"""
            raise NotImplementedError

        with semfn.using(FakeBackend({"choice": "none"})):
            with self.assertRaisesRegex(ValueError, "tags must contain"):
                await tag_for("x", [])
            self.assertIsNone(await project_for("x", []))


class DecorationTests(unittest.TestCase):
    def test_rejects_bad_signatures_at_decoration_time(self):
        def no_docstring(text: str) -> bool:
            raise NotImplementedError

        def no_return(text: str):
            """Is it?"""

        def unsupported(text: str) -> int:
            """How many?"""
            raise NotImplementedError

        def ambiguous(a: list[str], b: list[str]) -> str:
            """Which one?"""
            raise NotImplementedError

        def optional_score(text: str) -> semfn.Score[Literal["a", "b"]] | None:
            """How much?"""
            raise NotImplementedError

        def clashing_labels(text: str) -> Literal["none"] | None:
            """Which?"""
            raise NotImplementedError

        cases: list[tuple[Callable[..., object], str]] = [
            (no_docstring, "docstring"),
            (no_return, "return annotation"),
            (unsupported, "unsupported semantic return type"),
            (ambiguous, "several parameters"),
            (optional_score, "Score cannot be optional"),
            (clashing_labels, "unique"),
        ]
        for function, message in cases:
            with (
                self.subTest(message),
                self.assertRaisesRegex(TypeError, message),
            ):
                semfn.SemanticFunction(function)

    def test_rejects_bad_policies(self):
        def certain(text: str) -> bool:
            """Is it?"""
            raise NotImplementedError

        with self.assertRaisesRegex(TypeError, "without min_confidence"):
            semfn.SemanticFunction(certain, uncertain=None)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            semfn.semantic(min_confidence=1.5)(certain)

    def test_keeps_function_metadata(self):
        self.assertEqual(classify.__name__, "classify")
        self.assertEqual(classify.__doc__, "Classify the request.")
        self.assertEqual(classify.__module__, __name__)

    def test_rejects_wrong_arguments_immediately(self):
        with self.assertRaises(TypeError):
            classify("x", unknown=1)  # ty: ignore[unknown-argument]


class PolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_raises_below_min_confidence(self):
        @semfn.semantic(min_confidence=0.8)
        def strict(text: str) -> bool:
            """Is this certain?"""
            raise NotImplementedError

        with semfn.using(FakeBackend({"noul": 0.6}, {"noul": 0.9})):
            with self.assertRaises(semfn.UncertainDecision) as caught:
                await strict("maybe")
            self.assertIs(await strict("surely"), True)
        self.assertAlmostEqual(caught.exception.decision.confidence, 0.6)
        self.assertEqual(caught.exception.minimum, 0.8)

    async def test_returns_uncertain_fallback(self):
        @semfn.semantic(min_confidence=0.8, uncertain=None)
        def lenient(text: str) -> bool:
            """Is this certain?"""
            raise NotImplementedError

        with semfn.using(FakeBackend({"noul": 0.6}, {"noul": 0.6})):
            self.assertIsNone(await lenient("maybe"))
            result = await lenient.result("maybe")
        self.assertIsNone(result.value)
        self.assertAlmostEqual(result.confidence, 0.6)


class GatherTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_state_shares_one_backend_call(self):
        backend = FakeBackend(
            {"choice": "bug", "probabilities": {"bug": 0.9, "feature": 0.1}},
            {"noul": 0.8},
        )
        with semfn.using(backend):
            kind, decision = await semfn.gather(
                classify("Login loops"), blocking.result("Login loops")
            )
        self.assertEqual(kind, "bug")
        self.assertIs(decision.value, True)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(backend.calls[0][1]), 2)

    async def test_different_states_are_split_and_keep_their_order(self):
        backend = FakeBackend({"noul": 0.9}, {"noul": 0.1}, {"noul": 0.7})
        with semfn.using(backend):
            results = await semfn.gather(blocking("a"), blocking("b"), blocking("a"))
        # The two "a" questions share the first request, "b" gets the second.
        self.assertEqual(results, (True, True, False))
        states = [state for state, _ in backend.calls]
        self.assertEqual(states, [{"text": "a"}, {"text": "b"}])

    async def test_empty_gather(self):
        self.assertEqual(await semfn.gather(), ())

    async def test_a_call_can_be_awaited_later(self):
        with semfn.using(FakeBackend({"noul": 0.9})):
            pending = blocking("Cannot log in")
            await asyncio.sleep(0)
            self.assertIs(await pending, True)


class BackendSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_function_backend_beats_the_default(self):
        default, own = FakeBackend(), FakeBackend({"noul": 0.9})

        @semfn.semantic(backend=own)
        def pinned(text: str) -> bool:
            """Is it?"""
            raise NotImplementedError

        semfn.configure(backend=default)
        self.assertIs(await pinned("x"), True)
        self.assertEqual(len(default.calls), 0)

    async def test_using_beats_the_function_backend_and_is_scoped(self):
        own, scoped = FakeBackend({"noul": 0.9}), FakeBackend({"noul": 0.1})

        @semfn.semantic(backend=own)
        def pinned(text: str) -> bool:
            """Is it?"""
            raise NotImplementedError

        with semfn.using(scoped):
            self.assertIs(await pinned("x"), False)
        self.assertIs(await pinned("x"), True)

    async def test_async_backend_and_unwrapped_answers(self):
        class AsyncBackend:
            async def evaluate(self, state, questions):
                return {question: {"noul": 0.9} for question in questions}

        with semfn.using(AsyncBackend()):
            self.assertIs(await blocking("x"), True)

    async def test_warmup_is_forwarded(self):
        class WarmBackend(FakeBackend):
            warmed = False

            async def warmup(self):
                self.warmed = True

        backend = WarmBackend()
        await semfn.configure(backend=backend).warmup()
        self.assertTrue(backend.warmed)
        await semfn.configure(backend=FakeBackend()).warmup()

    def test_unknown_backend_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            semfn.configure(backend="gpt")


class MethodTests(unittest.IsolatedAsyncioTestCase):
    async def test_methods_use_the_instance_as_state(self):
        @dataclass
        class Ticket:
            body: str

            @semfn.semantic
            def is_blocking(self) -> bool:
                """Is the user blocked?"""
                raise NotImplementedError

        backend = FakeBackend({"noul": 0.9})
        with semfn.using(backend):
            self.assertIs(await Ticket("Cannot log in").is_blocking(), True)
        self.assertEqual(backend.calls[0][0], {"self": {"body": "Cannot log in"}})
        self.assertIsInstance(Ticket.is_blocking, semfn.SemanticFunction)


if __name__ == "__main__":
    unittest.main()
