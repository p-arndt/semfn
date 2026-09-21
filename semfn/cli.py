from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import evals
from .function import SemanticFunction
from .runtime import configure

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from .evals import CaseResult, EvalReport


def _files(paths: Sequence[str]) -> Iterator[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(path.rglob("*.py"))
        elif path.is_file():
            yield path
        else:
            raise FileNotFoundError(f"No such file or directory: {raw}")


def _load(path: Path) -> dict[str, Any]:
    name = f"semfn_evals_{path.stem}_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses with postponed annotations look their module up in sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return vars(module)


def discover(paths: Sequence[str]) -> list[SemanticFunction[..., Any]]:
    """Every semantic function with registered cases, in file order."""
    found: dict[int, SemanticFunction[..., Any]] = {}
    for path in _files(paths):
        for value in _load(path).values():
            if isinstance(value, SemanticFunction) and value.cases:
                found.setdefault(id(value), value)
    return list(found.values())


def _shorten(text: str, limit: int = 100) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _describe_failure(result: CaseResult) -> str:
    arguments = [repr(value) for value in result.case.args]
    arguments += [f"{key}={value!r}" for key, value in result.case.kwargs.items()]
    got = "uncertain" if result.uncertain else repr(result.decision.value)
    lines = [f"    - {_shorten(argument)}" for argument in arguments]
    lines.append(f"      expected {_shorten(repr(result.case.expected))}")
    lines.append(f"      got      {_shorten(got)} at {result.decision.confidence:.2f}")
    return "\n".join(lines)


def _print(reports: Sequence[EvalReport], min_accuracy: float, verbose: bool) -> None:
    width = max(len(report.name) for report in reports)
    for report in reports:
        line = f"{report.name:<{width}}  {report.passed:>3}/{report.total}"
        if report.accuracy < min_accuracy:
            line += f"  ← below {min_accuracy:.0%}"
        print(line)
        if verbose:
            for failure in report.failures:
                print(_describe_failure(failure))


async def _run(functions: Sequence[SemanticFunction[..., Any]]) -> list[EvalReport]:
    # One function at a time keeps the output order stable and the load on a
    # local model predictable; cases within a function already run concurrently.
    return [await evals.run(function) for function in functions]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="semfn")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("eval", help="run the cases registered with .eval()")
    run.add_argument("paths", nargs="*", default=["evals"], help="files or directories")
    run.add_argument("--backend", choices=["laya", "jev"])
    run.add_argument("--model")
    run.add_argument("--min-accuracy", type=float, default=0.8, metavar="FRACTION")
    run.add_argument("-v", "--verbose", action="store_true", help="show failed cases")
    options = parser.parse_args(argv)

    # Eval files import the application code they test, relative to where the
    # command is run, like pytest's rootdir.
    if (root := str(Path.cwd())) not in sys.path:
        sys.path.insert(0, root)
    if options.backend or options.model:
        configure(backend=options.backend or "laya", model=options.model)

    functions = discover(options.paths)
    if not functions:
        print(f"No eval cases found in {', '.join(options.paths)}", file=sys.stderr)
        return 2
    reports = asyncio.run(_run(functions))
    _print(reports, options.min_accuracy, options.verbose)
    return 1 if any(r.accuracy < options.min_accuracy for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
