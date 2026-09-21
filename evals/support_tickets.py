from typing import Literal

import semfn


@semfn.semantic
def classify_ticket(text: str) -> Literal["bug", "feature", "question"]:
    """What kind of support request is this?"""
    raise NotImplementedError


classify_ticket.eval(
    [
        semfn.case("Login redirects forever after the last update", expected="bug"),
        semfn.case("The export button throws a 500 error", expected="bug"),
        semfn.case("App crashes when I rotate my phone", expected="bug"),
        semfn.case("Totals on the invoice PDF are off by one cent", expected="bug"),
        semfn.case(
            "Since yesterday notifications arrive twice for every comment",
            expected="bug",
        ),
        semfn.case("Search returns nothing even for exact titles", expected="bug"),
        semfn.case("Please add a dark mode to the dashboard", expected="feature"),
        semfn.case("It would be great to export reports as CSV", expected="feature"),
        semfn.case("Can you support single sign-on with Okta?", expected="feature"),
        semfn.case(
            "We need an API endpoint for bulk user creation", expected="feature"
        ),
        semfn.case(
            "I wish I could pin my favorite projects to the sidebar",
            expected="feature",
        ),
        semfn.case("How do I reset my password?", expected="question"),
        semfn.case("Where can I find my invoices?", expected="question"),
        semfn.case(
            "What is the difference between the Pro and Team plan?", expected="question"
        ),
        semfn.case("Is my data stored in the EU?", expected="question"),
        semfn.case(
            "How many seats are included in our current subscription?",
            expected="question",
        ),
    ]
)
