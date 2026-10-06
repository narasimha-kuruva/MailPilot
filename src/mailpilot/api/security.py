"""Who may call the API.

MailPilot can read the mailbox and approve sends, so the agent, context and
metrics routes are never open to the network by default:

- With `MAILPILOT_API_KEY` set, every request must present it, as
  `X-API-Key: <key>` or `Authorization: Bearer <key>`.
- Without a key, only requests from this machine (a loopback address) are
  accepted. That covers `python -m mailpilot.main` on a laptop, and still
  protects a server mistakenly bound to `0.0.0.0`. Anything else -- another
  machine, a Docker container's published port -- needs the key.

`/health` and the API docs stay open: neither reveals mail. A reverse proxy
on the same machine reaches MailPilot from loopback, so without a key the
proxy itself must authenticate its users.
"""

from __future__ import annotations

import hmac
import ipaddress

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from mailpilot.api.deps import SettingsDep

# auto_error=False: a missing header is decided below, not rejected outright,
# because loopback requests need no key when none is configured. Declaring
# both schemes gives the /docs page its "Authorize" button.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="The MAILPILOT_API_KEY value.")
_bearer = HTTPBearer(auto_error=False, description="The MAILPILOT_API_KEY value, as a bearer token.")


def _is_loopback(host: str | None) -> bool:
    try:
        return host is not None and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


async def require_api_access(
    request: Request,
    settings: SettingsDep,
    api_key: str | None = Security(_api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    configured = settings.mailpilot_api_key
    presented = api_key or (bearer.credentials if bearer else None)
    if configured:
        # Constant-time comparison, so response timing doesn't leak the key.
        if presented and hmac.compare_digest(presented.encode(), configured.encode()):
            return
        raise HTTPException(
            status_code=401,
            detail="Missing or wrong API key. Send it as `X-API-Key: <key>` or `Authorization: Bearer <key>`.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if _is_loopback(request.client.host if request.client else None):
        return
    raise HTTPException(
        status_code=403,
        detail=(
            "MailPilot has no API key configured, so it only accepts requests from this machine. "
            "Set MAILPILOT_API_KEY to allow other clients."
        ),
    )
