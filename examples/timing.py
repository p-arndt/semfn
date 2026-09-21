import asyncio
from collections.abc import Awaitable
from time import perf_counter
from typing import Any, Literal

import semfn


@semfn.semantic
def classify(text: str) -> Literal["bug", "feature", "question", "other"]:
    """What kind of request is this?"""
    raise NotImplementedError


async def measure(label: str, awaitable: Awaitable[Any]) -> Any:
    started = perf_counter()
    result = await awaitable
    elapsed = perf_counter() - started
    print(f"{label}: {elapsed:.3f}s -> {result}")
    return result


async def main() -> None:
    runtime = semfn.configure()

    started = perf_counter()
    await runtime.warmup()
    print(f"Model load: {perf_counter() - started:.3f}s")

    await measure("First inference", classify("Login redirects forever"))
    await measure("Warm inference", classify("Please add dark mode"))


if __name__ == "__main__":
    asyncio.run(main())
