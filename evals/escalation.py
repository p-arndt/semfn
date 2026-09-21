from dataclasses import dataclass

import semfn


@dataclass
class Customer:
    plan: str
    seats: int
    note: str = ""


@semfn.semantic
def should_escalate(message: str, customer: Customer) -> bool:
    """Should this message be escalated to a human account manager right away?

    Escalate when a paying customer threatens to cancel, reports data loss or a
    security problem, or is completely blocked from working. Routine questions
    and minor issues are not escalated.
    """
    raise NotImplementedError


enterprise = Customer("enterprise", 450, "Renewal due next month")
team = Customer("team", 12)
free = Customer("free", 1)

should_escalate.eval(
    [
        semfn.case(
            "If this is not fixed by Friday we are moving to a competitor.",
            enterprise,
            expected=True,
        ),
        semfn.case(
            "All our project data from last week is gone. Where is it?",
            team,
            expected=True,
        ),
        semfn.case(
            "We can see another company's invoices in our account.",
            enterprise,
            expected=True,
        ),
        semfn.case(
            "Nobody in our company can log in since this morning.",
            enterprise,
            expected=True,
        ),
        semfn.case(
            "I want to cancel our subscription, this tool keeps failing us.",
            team,
            expected=True,
        ),
        semfn.case(
            "Someone logged into my admin account from a country I have never been to.",
            team,
            expected=True,
        ),
        semfn.case("How do I change my profile picture?", enterprise, expected=False),
        semfn.case("Is there a keyboard shortcut for search?", team, expected=False),
        semfn.case(
            "The logo in the footer looks slightly blurry.", free, expected=False
        ),
        semfn.case("Thanks, that solved it!", enterprise, expected=False),
        semfn.case("Can you send me last month's invoice again?", team, expected=False),
        semfn.case("Do you have a student discount?", free, expected=False),
        semfn.case(
            "A tooltip is misspelled on the settings page.", enterprise, expected=False
        ),
    ]
)
