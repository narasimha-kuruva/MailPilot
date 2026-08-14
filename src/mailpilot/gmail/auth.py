"""Gmail OAuth 2.0 authentication.

Credentials are never hardcoded: the OAuth client secrets file and the
cached user token are both read from paths configured via environment
variables (`GOOGLE_OAUTH_CLIENT_SECRETS_FILE`, `GOOGLE_OAUTH_TOKEN_FILE`;
see `mailpilot.config.Settings`). On first use this runs an interactive,
local-server OAuth consent flow and caches the resulting token; on every
later use it loads and, if needed, silently refreshes that cached token.
"""

from __future__ import annotations

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from mailpilot.config import Settings
from mailpilot.logging_config import get_logger

logger = get_logger(__name__)

# gmail.modify covers all read/write operations used by this app (search,
# read, labels, drafts, send) except permanent deletion, which MailPilot
# never performs.
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def load_credentials(settings: Settings) -> Credentials:
    """Return valid Gmail OAuth credentials, authorizing or refreshing as needed."""
    token_path = Path(settings.google_oauth_token_file)
    creds: Credentials | None = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), GMAIL_SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        logger.info("Refreshing expired Gmail OAuth token")
        creds.refresh(Request())
        _save_credentials(creds, token_path)
        return creds

    secrets_path = Path(settings.google_oauth_client_secrets_file)
    if not secrets_path.exists():
        raise FileNotFoundError(
            f"Gmail OAuth client secrets file not found at '{secrets_path}'. "
            "Download it from Google Cloud Console (OAuth client ID, "
            "Desktop app type) and set GOOGLE_OAUTH_CLIENT_SECRETS_FILE, "
            "then restart and complete the consent flow."
        )

    logger.info("Starting interactive Gmail OAuth consent flow")
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), GMAIL_SCOPES)
    creds = flow.run_local_server(port=0)
    _save_credentials(creds, token_path)
    return creds


def _save_credentials(creds: Credentials, token_path: Path) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")
