from dataclasses import dataclass

import semfn


@dataclass
class Project:
    id: str
    name: str
    description: str


@semfn.semantic
def project_for(filename: str, content: str, projects: list[Project]) -> Project | None:
    """Which project is this document most closely related to?"""
    raise NotImplementedError


infra = Project("infra", "Launchpad", "Virtual machines and isolated agent runtimes")
auth = Project("auth", "Gatekeeper", "Login, sessions, permissions, and identity")
billing = Project("billing", "Ledger", "Invoices, payments, refunds, and subscriptions")
projects = [infra, auth, billing]

project_for.eval(
    [
        semfn.case(
            "firecracker-notes.md",
            "Evaluate Firecracker as an isolation boundary for customer agents.",
            projects,
            expected=infra,
        ),
        semfn.case(
            "vm-pool.md",
            "Keep a warm pool of virtual machines so sandboxes start within a second.",
            projects,
            expected=infra,
        ),
        semfn.case(
            "snapshot-restore.txt",
            "Restoring a runtime from a memory snapshot fails on ARM hosts.",
            projects,
            expected=infra,
        ),
        semfn.case(
            "oauth-migration.md",
            "Move social sign-in to OAuth 2.1 with PKCE and rotate refresh tokens.",
            projects,
            expected=auth,
        ),
        semfn.case(
            "rbac.md",
            "Introduce roles so admins can restrict who may delete workspaces.",
            projects,
            expected=auth,
        ),
        semfn.case(
            "session-timeouts.txt",
            "Users complain they are logged out after ten minutes of inactivity.",
            projects,
            expected=auth,
        ),
        semfn.case(
            "refund-policy.md",
            "Partial refunds should be prorated to the day for annual plans.",
            projects,
            expected=billing,
        ),
        semfn.case(
            "vat.md",
            "Invoices for EU customers need the reverse-charge VAT note.",
            projects,
            expected=billing,
        ),
        semfn.case(
            "dunning.txt",
            "Retry failed card payments three times before pausing the subscription.",
            projects,
            expected=billing,
        ),
        semfn.case(
            "offsite.md",
            "Agenda and hotel options for the team offsite in Lisbon.",
            projects,
            expected=None,
        ),
        semfn.case(
            "recipe.txt",
            "Sourdough starter feeding schedule and hydration ratios.",
            projects,
            expected=None,
        ),
        semfn.case(
            "hiring.md",
            "Interview questions for the senior designer role.",
            projects,
            expected=None,
        ),
    ]
)
