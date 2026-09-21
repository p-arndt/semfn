from dataclasses import dataclass

import semfn


@dataclass
class Command:
    name: str
    description: str


@semfn.semantic
def best_match(query: str, commands: list[Command]) -> Command | None:
    """Which command best satisfies this request?"""
    raise NotImplementedError


restart = Command("restart-staging-api", "Restart the API service in staging")
deploy = Command("deploy-production-web", "Deploy the web application to production")
logs = Command("tail-worker-logs", "Stream recent background worker logs")
status = Command("open-status-page", "Open the public system status page")
commands = [restart, deploy, logs, status]

best_match.eval(
    [
        semfn.case("bounce the test API", commands, expected=restart),
        semfn.case("staging api is stuck, kick it", commands, expected=restart),
        semfn.case("reboot the api on staging", commands, expected=restart),
        semfn.case("ship the website to prod", commands, expected=deploy),
        semfn.case("release the frontend", commands, expected=deploy),
        semfn.case("push the new web build live", commands, expected=deploy),
        semfn.case("show me what the workers are doing", commands, expected=logs),
        semfn.case(
            "why are background jobs failing? let me see output",
            commands,
            expected=logs,
        ),
        semfn.case("follow the job queue logs", commands, expected=logs),
        semfn.case("are we down?", commands, expected=status),
        semfn.case("show the uptime page customers see", commands, expected=status),
        semfn.case("order pizza for the team", commands, expected=None),
        semfn.case("what's the weather in Berlin", commands, expected=None),
        semfn.case("delete the production database", commands, expected=None),
    ]
)
