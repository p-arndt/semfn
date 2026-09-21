import asyncio
from dataclasses import dataclass

import semfn


@dataclass
class Command:
    name: str
    description: str

    async def execute(self) -> None:
        print(f"Executing: {self.name}")


@semfn.semantic(min_confidence=0.7, uncertain=None)
def best_match(query: str, commands: list[Command]) -> Command | None:
    """Which command best satisfies this request?"""


async def main() -> None:
    commands = [
        Command("restart-staging-api", "Restart the API service in staging"),
        Command("deploy-production-web", "Deploy the web application to production"),
        Command("tail-worker-logs", "Stream recent background worker logs"),
        Command("open-status-page", "Open the public system status page"),
    ]

    command = await best_match("bounce the test API", commands)
    if command:
        await command.execute()
    else:
        print("No command matched with enough confidence")


if __name__ == "__main__":
    asyncio.run(main())
