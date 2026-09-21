from dataclasses import dataclass

import semfn


@dataclass
class Error:
    service: str
    message: str
    trace: str = ""


@semfn.semantic
def same_incident(a: Error, b: Error) -> bool:
    """Are these symptoms of the same underlying problem?"""
    raise NotImplementedError


session_loop = Error(
    "web", "Redirect loop after login", "SessionExpired at middleware.py:84"
)
session_api = Error(
    "api", "Login succeeds, then sends me back to sign-in", "InvalidSession"
)
session_mobile = Error("mobile", "User is logged out immediately after signing in")
search_timeout = Error("search", "Query timed out", "ConnectionTimeout at index.py:31")
search_pool = Error("search", "Could not get a database connection from the pool")
search_slow = Error("web", "Search page never finishes loading")
disk_full = Error("worker", "No space left on device", "OSError at upload.py:12")
upload_failed = Error("api", "Saving the uploaded file failed: disk quota exceeded")
payment_declined = Error("billing", "Card was declined by the payment provider")
payment_webhook = Error("billing", "Stripe webhook signature verification failed")
cert_expired = Error("gateway", "TLS handshake failed: certificate has expired")
cert_client = Error("mobile", "SSL error: the server certificate is not valid anymore")

same_incident.eval(
    [
        semfn.case(session_loop, session_api, expected=True),
        semfn.case(session_loop, session_mobile, expected=True),
        semfn.case(session_api, session_mobile, expected=True),
        semfn.case(search_timeout, search_pool, expected=True),
        semfn.case(search_timeout, search_slow, expected=True),
        semfn.case(disk_full, upload_failed, expected=True),
        semfn.case(cert_expired, cert_client, expected=True),
        semfn.case(session_loop, search_timeout, expected=False),
        semfn.case(session_api, disk_full, expected=False),
        semfn.case(search_pool, payment_declined, expected=False),
        semfn.case(payment_declined, payment_webhook, expected=False),
        semfn.case(disk_full, cert_expired, expected=False),
        semfn.case(upload_failed, session_mobile, expected=False),
        semfn.case(cert_client, search_slow, expected=False),
    ]
)
