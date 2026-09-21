import asyncio
from typing import Literal

import semfn


@semfn.semantic
def classify(text: str) -> Literal["bug", "feature", "question", "other"]:
    """What kind of request is this?"""
    raise NotImplementedError


async def main() -> None:
    decision = await classify.result("Login redirects forever")
    print(decision)


if __name__ == "__main__":
    asyncio.run(main())
