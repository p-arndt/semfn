import contextlib
import io
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Literal

import semfn
from semfn import cli


class FakeBackend:
    """Answers every choice question with `label` at `confidence`."""

    def __init__(self, label, confidence=0.9):
        self.label = label
        self.confidence = confidence

    def evaluate(self, state, questions):
        answer = {"choice": self.label, "confidence": self.confidence}
        return dict.fromkeys(questions, answer)


def make_classify(**options):
    @semfn.semantic(**options)
    def classify(text: str) -> Literal["bug", "feature"]:
        """Classify the request."""
        raise NotImplementedError

    return classify


class EvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_report_counts_passes_and_failures(self):
        classify = make_classify()
        evaluation = classify.eval(
            [
                semfn.case("Crashes on save", expected="bug"),
                semfn.case(text="Add dark mode", expected="feature"),
            ]
        )
        with semfn.using(FakeBackend("bug")):
            report = await evaluation
        self.assertEqual((report.name, report.passed, report.total), ("classify", 1, 2))
        self.assertAlmostEqual(report.accuracy, 0.5)
        (failure,) = report.failures
        self.assertEqual(failure.case.expected, "feature")
        self.assertEqual(failure.decision.value, "bug")

    async def test_eval_registers_cases_across_calls(self):
        classify = make_classify()
        classify.eval([semfn.case("a", expected="bug")])
        classify.eval([semfn.case("b", expected="bug")])
        self.assertEqual([item.args for item in classify.cases], [("a",), ("b",)])

    async def test_uncertain_raise_counts_as_failure(self):
        classify = make_classify(min_confidence=0.8)
        evaluation = classify.eval([semfn.case("Crashes", expected="bug")])
        with semfn.using(FakeBackend("bug", confidence=0.5)):
            report = await evaluation
        self.assertEqual(report.passed, 0)
        self.assertTrue(report.results[0].uncertain)
        self.assertAlmostEqual(report.results[0].decision.confidence, 0.5)

    async def test_uncertain_value_can_be_the_expected_answer(self):
        classify = make_classify(min_confidence=0.8, uncertain=None)
        evaluation = classify.eval([semfn.case("???", expected=None)])
        with semfn.using(FakeBackend("bug", confidence=0.5)):
            report = await evaluation
        self.assertEqual(report.passed, 1)

    async def test_empty_report_has_zero_accuracy(self):
        self.assertEqual(semfn.EvalReport("empty", ()).accuracy, 0.0)


EVAL_FILE = """
    from dataclasses import dataclass
    from typing import Literal

    import semfn


    @dataclass
    class Ticket:
        body: str


    @semfn.semantic
    def classify(ticket: Ticket) -> Literal["bug", "feature"]:
        \"\"\"Classify the request.\"\"\"


    @semfn.semantic
    def without_cases(text: str) -> bool:
        \"\"\"Is this ignored?\"\"\"


    classify.eval(
        [
            semfn.case(Ticket("Crashes on save"), expected="bug"),
            semfn.case(Ticket("Add dark mode"), expected="feature"),
        ]
    )
"""


class CliTests(unittest.TestCase):
    def run_cli(self, *arguments, files=None):
        with tempfile.TemporaryDirectory() as directory:
            for name, content in (files or {"triage.py": EVAL_FILE}).items():
                Path(directory, name).write_text(textwrap.dedent(content))
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                semfn.using(FakeBackend("bug")),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = cli.main(["eval", directory, *arguments])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_reports_each_function_and_fails_below_threshold(self):
        code, output, _ = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(output, "classify    1/2  ← below 80%\n")

    def test_passes_when_accuracy_meets_threshold(self):
        code, output, _ = self.run_cli("--min-accuracy", "0.5")
        self.assertEqual(code, 0)
        self.assertEqual(output, "classify    1/2\n")

    def test_verbose_lists_failed_cases(self):
        _, output, _ = self.run_cli("-v")
        self.assertIn("Add dark mode", output)
        self.assertIn("expected 'feature'\n", output)
        self.assertIn("got      'bug' at 0.90\n", output)

    def test_no_cases_found(self):
        code, _, errors = self.run_cli(files={"empty.py": "import semfn\n"})
        self.assertEqual(code, 2)
        self.assertIn("No eval cases found", errors)

    def test_missing_path(self):
        with self.assertRaises(FileNotFoundError):
            cli.main(["eval", "does/not/exist"])
