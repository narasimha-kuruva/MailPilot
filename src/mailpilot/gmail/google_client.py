"""`GmailClient` implementation backed by the real Gmail API.

`googleapiclient` is synchronous, so every operation is offloaded to a
worker thread via `asyncio.to_thread`, keeping this class's public
interface async to match `GmailClient`. The underlying `googleapiclient`
service and OAuth credentials are built lazily on first use, so importing
or constructing this class never requires network access or a completed
OAuth flow — only calling one of its methods does.

Every read/write call except `send_email` is retried with bounded,
classified backoff (`mailpilot.resilience`) on transient errors (rate
limits, 5xx, timeouts). `send_email` is deliberately excluded: a network
error on a send doesn't tell you whether the email actually went out, so
retrying it automatically risks a duplicate send. See
`mailpilot.safety.idempotency` for how a duplicate send is prevented at
the layer that decides *whether* to call `send_email` at all.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, TypeVar

from googleapiclient.discovery import build

from mailpilot.config import Settings
from mailpilot.gmail.auth import load_credentials
from mailpilot.gmail.client import GmailClient
from mailpilot.gmail.mime_utils import build_raw_draft, parse_message
from mailpilot.logging_config import get_logger
from mailpilot.resilience import with_retries
from mailpilot.schemas.email import (
    CreatedDraft,
    DraftEmail,
    EmailMessage,
    EmailThread,
    Label,
)

logger = get_logger(__name__)

T = TypeVar("T")


class GoogleGmailClient(GmailClient):
    """Gmail read/write operations via the official `google-api-python-client`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._service: Any = None

    def _get_service(self) -> Any:
        if self._service is None:
            creds = load_credentials(self._settings)
            self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return self._service

    async def _run(self, fn: Callable[[], T]) -> T:
        """Run a sync Gmail API call in a worker thread, with bounded retries on transient errors."""
        return await with_retries(
            lambda: asyncio.to_thread(fn),
            max_retries=self._settings.gmail_max_retries,
            base_delay_seconds=self._settings.gmail_retry_base_delay_seconds,
        )

    async def search_messages(self, query: str, max_results: int = 25) -> list[EmailMessage]:
        service = self._get_service()

        def _list_ids() -> list[str]:
            response = (
                service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )
            return [item["id"] for item in response.get("messages", [])]

        def _get_one(message_id: str) -> EmailMessage:
            raw = service.users().messages().get(userId="me", id=message_id, format="full").execute()
            return parse_message(raw)

        message_ids = await self._run(_list_ids)
        if not message_ids:
            return []

        # Fetch each message's full body concurrently (across the default
        # thread pool). A single message that fails to fetch (e.g. deleted
        # between the list and get calls) is dropped with a warning rather
        # than failing the whole search.
        results = await asyncio.gather(
            *(self._run(lambda mid=mid: _get_one(mid)) for mid in message_ids), return_exceptions=True
        )
        messages: list[EmailMessage] = []
        for message_id, result in zip(message_ids, results):
            if isinstance(result, BaseException):
                logger.warning(
                    "Skipping message that failed to fetch during search",
                    extra={"extra_fields": {"message_id": message_id, "error": str(result)}},
                )
                continue
            messages.append(result)
        return messages

    async def get_message(self, message_id: str) -> EmailMessage:
        service = self._get_service()

        def _get() -> EmailMessage:
            raw = service.users().messages().get(userId="me", id=message_id, format="full").execute()
            return parse_message(raw)

        return await self._run(_get)

    async def get_thread(self, thread_id: str) -> EmailThread:
        service = self._get_service()

        def _get() -> EmailThread:
            raw = service.users().threads().get(userId="me", id=thread_id, format="full").execute()
            messages = [parse_message(message) for message in raw.get("messages", [])]
            subject = messages[0].subject if messages else ""
            labels = sorted({label for message in messages for label in message.labels})
            return EmailThread(thread_id=thread_id, subject=subject, messages=messages, labels=labels)

        return await self._run(_get)

    async def list_labels(self) -> list[Label]:
        service = self._get_service()

        def _list() -> list[Label]:
            response = service.users().labels().list(userId="me").execute()
            return [
                Label(label_id=item["id"], name=item["name"], type=item.get("type", "user"))
                for item in response.get("labels", [])
            ]

        return await self._run(_list)

    async def apply_label(self, message_id: str, label_id: str) -> None:
        service = self._get_service()

        def _apply() -> None:
            service.users().messages().modify(
                userId="me", id=message_id, body={"addLabelIds": [label_id]}
            ).execute()

        await self._run(_apply)

    async def create_draft(self, draft: DraftEmail) -> CreatedDraft:
        service = self._get_service()

        def _create() -> CreatedDraft:
            raw = build_raw_draft(draft.to, draft.cc, draft.subject, draft.body_text)
            message_body: dict[str, Any] = {"raw": raw}
            if draft.thread_id:
                message_body["threadId"] = draft.thread_id
            response = (
                service.users().drafts().create(userId="me", body={"message": message_body}).execute()
            )
            return CreatedDraft(draft_id=response["id"], message=parse_message(response["message"]))

        return await self._run(_create)

    async def get_draft(self, draft_id: str) -> CreatedDraft:
        service = self._get_service()

        def _get() -> CreatedDraft:
            response = service.users().drafts().get(userId="me", id=draft_id, format="full").execute()
            return CreatedDraft(draft_id=response["id"], message=parse_message(response["message"]))

        return await self._run(_get)

    async def send_email(self, draft_id: str) -> EmailMessage:
        service = self._get_service()

        def _send() -> EmailMessage:
            sent = service.users().drafts().send(userId="me", body={"id": draft_id}).execute()
            raw = service.users().messages().get(userId="me", id=sent["id"], format="full").execute()
            return parse_message(raw)

        # Deliberately NOT retried -- see the module docstring.
        return await asyncio.to_thread(_send)
