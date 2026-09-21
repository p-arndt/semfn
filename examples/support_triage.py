import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import semfn


class Severity(StrEnum):
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


async def triage(ticket: Ticket) -> None:
    # One gather over the same text is a single model pass for all three questions.
    kind, blocking, impact = await semfn.gather(
        intent(ticket.body), is_blocking(ticket.body), severity(ticket.body)
    )
    if kind == "bug" and blocking and impact in {Severity.HIGH, Severity.CRITICAL}:
        ticket.priority = "urgent"
    print(kind, blocking, impact.value, ticket.priority)


async def main() -> None:
    await triage(
        Ticket(
            "Every login redirects back to the login page. Our whole support team "
            "is locked out and cannot answer customers."
        )
    )
    await triage(Ticket("We would like a dark mode feature for the dashboard."))


if __name__ == "__main__":
    asyncio.run(main())
