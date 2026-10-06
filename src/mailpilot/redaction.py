"""Secret redaction for logs and the audit trail (Phase 5.1).

MailPilot holds three kinds of credential at runtime -- the Gemini API key,
the Gmail OAuth tokens, and the OAuth client secret -- and none of them may
reach a log line or an audit record. Nothing logs them on purpose; this
module is the defense in depth for the accidental cases: an exception
message that echoes a request, a debug line from a client library, an email
body that happens to contain a key, a tool argument the model copied from
one.

Two kinds of match:

- **By shape**, for the credential formats MailPilot actually handles
  (`AIza...` API keys, `ya29.` access tokens, `1//` refresh tokens,
  `GOCSPX-` client secrets, JWTs, `Bearer` headers), wherever they appear.
- **By name**, for values labelled as secrets (`access_token=...`,
  `{"client_secret": ...}`, `password: ...`), whatever they look like.

There is deliberately no generic "long random-looking string" rule: Gmail
message and draft ids and Gemini thought signatures all look like that, and
mangling them would make the logs useless for debugging while implying a
guarantee this module can't give. A secret in a shape not listed here, and
not labelled with one of the names below, is NOT caught.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"

# Each pattern matches only the secret itself, so the surrounding text survives.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),  # Google API key (GOOGLE_API_KEY)
    re.compile(r"ya29\.[0-9A-Za-z_.-]{20,}"),  # Google OAuth access token
    re.compile(r"1//[0-9A-Za-z_-]{20,}"),  # Google OAuth refresh token
    re.compile(r"GOCSPX-[0-9A-Za-z_-]{20,}"),  # Google OAuth client secret
    re.compile(r"eyJ[0-9A-Za-z_-]{10,}\.eyJ[0-9A-Za-z_-]{10,}\.[0-9A-Za-z_-]{10,}"),  # JWT, e.g. an id_token
)

# The minimum length keeps prose such as "bearer tokens" intact.
_BEARER = re.compile(r"(?i)\b(bearer\s+)[0-9A-Za-z._~+/=-]{16,}")

# Dict keys (compared lower-cased, with "-" read as "_") whose value is a
# secret whatever it looks like.
SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "refresh_token",
        "id_token",
        "client_secret",
        "api_key",
        "apikey",
        "google_api_key",
        "x_goog_api_key",
        "authorization",
        "password",
        "secret",
        "token",
    }
)

# The same idea inside free text: `access_token=...`, `"client_secret": "..."`,
# `password: ...`. Narrower than SENSITIVE_KEYS -- a bare "token:" or
# "secret:" in prose is too common to treat as a label.
_ASSIGNMENT_KEYS = (
    "access_token",
    "refresh_token",
    "id_token",
    "client_secret",
    "google_api_key",
    "api_key",
    "apikey",
    "password",
)
_ASSIGNMENT = re.compile(
    r"(?i)\b(" + "|".join(_ASSIGNMENT_KEYS) + r")"
    r"(\\?[\"']?\s*[:=]\s*\\?[\"']?)"
    r"(?!" + re.escape(REDACTED) + r")"
    r"([^\s\"'\\&,;}\]]+)"
)


def redact_text(text: str) -> str:
    """Return `text` with every recognised secret replaced by `REDACTED`."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    text = _BEARER.sub(rf"\g<1>{REDACTED}", text)
    return _ASSIGNMENT.sub(rf"\g<1>\g<2>{REDACTED}", text)


def is_sensitive_key(key: Any) -> bool:
    return isinstance(key, str) and key.lower().replace("-", "_") in SENSITIVE_KEYS


def redact_value(value: Any) -> Any:
    """Redact a JSON-like structure: every string is scrubbed with
    `redact_text`, and the value under a `SENSITIVE_KEYS` key is replaced
    outright (left alone when empty, so "the key isn't set" stays visible).
    Lists and tuples come back as lists; other values are returned as-is."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            key: REDACTED if is_sensitive_key(key) and item not in (None, "") else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_value(item) for item in value]
    return value
