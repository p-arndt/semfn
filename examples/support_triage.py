import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import semfn


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Ticket:
    body: str
    priority: str = "normal"


@semfn.semantic
def intent(text: str) -> Literal["bug", "feature", "question", "other"]:
    """What does the user want?"""
    raise NotImplementedError


@semfn.semantic
def is_blocking(text: str) -> bool:
    """Is the user unable to continue because of this problem?"""
    raise NotImplementedError


@semfn.semantic
def severity(text: str) -> Severity:
    """How severe is the reported problem?"""
    raise NotImplementedError


async def main() -> None:
    ticket = Ticket(
        "Every login redirects back to the login page. Our whole support team "
        "is locked out and cannot answer customers."
    )

    async with semfn.semantic.batch() as decisions:
        intent(ticket.body)
        is_blocking(ticket.body)
        severity(ticket.body)

    kind, blocking, impact = await decisions.resolve()
    if kind == "bug" and blocking and impact in {Severity.HIGH, Severity.CRITICAL}:
        ticket.priority = "urgent"

    print(kind, blocking, impact.value, ticket.priority)

    # Second ticket
    ticket2 = Ticket("We would like a dark mode feature for the dashboard.")

    async with semfn.semantic.batch() as decisions2:
        intent(ticket2.body)
        is_blocking(ticket2.body)
        severity(ticket2.body)

    kind2, blocking2, impact2 = await decisions2.resolve()
    if kind2 == "bug" and blocking2 and impact2 in {Severity.HIGH, Severity.CRITICAL}:
        ticket2.priority = "urgent"

    print(kind2, blocking2, impact2.value, ticket2.priority)


if __name__ == "__main__":
    asyncio.run(main())
