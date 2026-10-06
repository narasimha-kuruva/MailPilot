"""Who may call the API.

MailPilot can read the mailbox and approve sends, so the agent, context and
metrics routes are never open to the network by default:

- With `MAILPILOT_API_KEY` set, every request must present it, as
  `X-API-Key: <key>` or `Authorization: Bearer <key>`.
- Without a key, only requests from this machine (a loopback address) are
  accepted. That covers `python -m mailpilot.main` on a laptop, and still
  protects a server mistakenly bound to `0.0.0.0`. Anything else -- another
  machine, a Docker container's published port -- needs the key.

  Coming from loopback isn't enough on its own: any web page the user has
  open runs in a browser on this machine. Such a page could post to
  127.0.0.1 directly, or rebind its own hostname to 127.0.0.1 (DNS
  rebinding) and then read the answers as if they were its own. So without
  a key, a request must also be addressed to this machine (the `Host`
  header) and, if a browser sent it, come from a page this machine served
  (the `Origin` header). A request with the key needs neither: a page
  can't know the key.

`/health` and the API docs stay open: neither reveals mail. Behind a reverse
proxy, set a key and have the proxy send it.
"""

from __future__ import annotations

import hmac
import ipaddress
from urllib.parse import urlsplit

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


def _names_this_machine(hostname: str | None) -> bool:
    """`localhost` or a loopback address -- names a remote page can't rebind."""
    return hostname is not None and (hostname.lower() == "localhost" or _is_loopback(hostname))


def _hostname(url: str) -> str | None:
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


def _local_browser_context(request: Request) -> bool:
    host = request.headers.get("host")
    if host is None or not _names_this_machine(_hostname(f"//{host}")):
        return False
    origin = request.headers.get("origin")
    # Browsers send Origin on every cross-origin request and on same-origin
    # POSTs; "null" (a sandboxed frame, a file:// page) has no hostname.
    return origin is None or _names_this_machine(_hostname(origin))


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
    if not _is_loopback(request.client.host if request.client else None):
        raise HTTPException(
            status_code=403,
            detail=(
                "MailPilot has no API key configured, so it only accepts requests from this machine. "
                "Set MAILPILOT_API_KEY to allow other clients."
            ),
        )
    if not _local_browser_context(request):
        raise HTTPException(
            status_code=403,
            detail=(
                "MailPilot has no API key configured, so it only answers requests addressed to localhost, "
                "127.0.0.1 or [::1], and only from its own pages. Set MAILPILOT_API_KEY to allow other clients."
            ),
        )
