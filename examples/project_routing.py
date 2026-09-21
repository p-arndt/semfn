import asyncio
from dataclasses import dataclass

import semfn


@dataclass
class Project:
    id: str
    name: str
    description: str


@semfn.semantic(min_confidence=0.65, uncertain=None)
def project_for(filename: str, content: str, projects: list[Project]) -> Project | None:
    """Which project is this document most closely related to?"""
    raise NotImplementedError


async def main() -> None:
    projects = [
        Project("infra", "Launchpad", "Virtual machines and isolated agent runtimes"),
        Project("auth", "Gatekeeper", "Login, sessions, permissions, and identity"),
        Project("billing", "Ledger", "Invoices, payments, refunds, and subscriptions"),
    ]

    decision = await project_for.result(
        "firecracker-notes.md",
        "Evaluate Firecracker as an isolation boundary for running customer agents.",
        projects,
    )

    if decision.value is None:
        print(f"No confident match ({decision.confidence:.0%})")
    else:
        print(f"{decision.value.name} ({decision.confidence:.0%})")
        for project, probability in decision.distribution:
            print(f"  {project.name if project else 'none'}: {probability:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
