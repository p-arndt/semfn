import asyncio
from dataclasses import dataclass, field

import semfn


@dataclass
class Error:
    service: str
    message: str
    trace: str = ""


@dataclass
class Incident:
    title: str
    errors: list[Error] = field(default_factory=list)


@semfn.semantic
def same_incident(existing: Error, incoming: Error) -> bool:
    """Are these probably symptoms of the same underlying software problem?"""
    raise NotImplementedError


async def attach_to_incident(
    error: Error, incidents: list[Incident]
) -> Incident | None:
    for incident in incidents:
        match = await same_incident.result(incident.errors[-1], error)
        if match.value and match.confidence > 0.9:
            incident.errors.append(error)
            return incident
    return None


async def main() -> None:
    incidents = [
        Incident(
            "Expired authentication sessions",
            [
                Error(
                    "web",
                    "Redirect loop after login",
                    "SessionExpired at middleware.py:84",
                )
            ],
        ),
        Incident(
            "Search database unavailable",
            [Error("search", "Query timed out", "ConnectionTimeout at index.py:31")],
        ),
    ]
    incoming = Error(
        "api", "Login succeeds, then sends me back to sign-in", "InvalidSession"
    )

    incident = await attach_to_incident(incoming, incidents)
    print(incident.title if incident else "Create a new incident")


if __name__ == "__main__":
    asyncio.run(main())
