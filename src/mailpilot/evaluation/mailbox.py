"""An in-memory mailbox for agent evaluation (Phase 5.8).

A `GmailClient` over a fixed set of messages, so evaluation scenarios run
the real agent -- graph, tools, approval gate, guardrails -- against a
known inbox, with no network and nothing that can actually leave the
machine. It records every call and can inject failures, so a scenario can
assert what the agent did (searched, drafted, never sent) and how it coped
with a Gmail error.

Search understands the common Gmail operators (`is:`, `in:`, `from:`,
`to:`, `subject:`, `label:`, `category:`, a leading `-` to negate,
top-level `OR`) plus free-text words, matched against sender, subject,
snippet, and body. Other operators (`newer_than:`, `has:`, ...) are ignored
rather than treated as a non-match, so a real model's richer queries still
find the fixture mail.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from mailpilot.gmail.client import GmailClient
from mailpilot.schemas.email import CreatedDraft, DraftEmail, EmailAddress, EmailMessage, EmailThread, Label

SYSTEM_LABELS = ("INBOX", "UNREAD", "IMPORTANT", "STARRED", "SENT", "DRAFT", "TRASH", "SPAM", "CATEGORY_PROMOTIONS")

_TOKEN = re.compile(r'-?\w+:"[^"]*"|-?\w+:\S+|-?"[^"]*"|\S+')


class MailboxError(Exception):
    """A Gmail-style API error; `status_code` drives `mailpilot.resilience.classify_error`."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code


def _haystack(message: EmailMessage) -> str:
    sender = f"{message.sender.name or ''} {message.sender.email}"
    return f"{sender} {message.subject} {message.snippet} {message.body_text or ''}".lower()


class InMemoryGmailClient(GmailClient):
    def __init__(
        self,
        messages: Iterable[EmailMessage],
        *,
        own_email: str,
        user_labels: Iterable[Label] = (),
        failures: dict[str, list[int]] | None = None,
    ) -> None:
        self.messages = {m.message_id: m.model_copy(deep=True) for m in messages}
        self.own_email = own_email
        self.labels = [Label(label_id=name, name=name, type="system") for name in SYSTEM_LABELS] + list(user_labels)
        self.drafts: dict[str, CreatedDraft] = {}
        self.sent: list[EmailMessage] = []
        self.applied_labels: list[tuple[str, str]] = []
        self.calls: list[tuple[str, tuple]] = []
        # method name -> status codes to fail with, one per call, before succeeding again
        self._failures = {name: list(codes) for name, codes in (failures or {}).items()}
        self._next_draft = 1

    def _enter(self, method: str, *args: object) -> None:
        self.calls.append((method, args))
        pending = self._failures.get(method)
        if pending:
            code = pending.pop(0)
            reason = "rateLimitExceeded" if code == 429 else "backendError" if code >= 500 else "forbidden"
            raise MailboxError(code, f"Gmail API error ({reason}) in {method}")

    def call_count(self, method: str) -> int:
        return sum(1 for name, _ in self.calls if name == method)

    # --- search ---------------------------------------------------------------

    def _label_matches(self, message: EmailMessage, value: str) -> bool:
        wanted = value.lower().replace("-", " ").replace("_", " ")
        for label in self.labels:
            if label.label_id in message.labels and wanted in (
                label.label_id.lower().replace("_", " "),
                label.name.lower().replace("-", " ").replace("_", " "),
            ):
                return True
        return False

    def _term_matches(self, message: EmailMessage, term: str) -> bool | None:
        """True/False for a term this mailbox understands; None to ignore it."""
        if ":" not in term or term.startswith('"'):
            return term.strip('"').lower() in _haystack(message)
        operator, value = term.split(":", 1)
        operator, value = operator.lower(), value.strip('"').lower()
        if operator == "is":
            flag = {"unread": "UNREAD", "starred": "STARRED", "important": "IMPORTANT"}.get(value)
            if value == "read":
                return "UNREAD" not in message.labels
            return flag in message.labels if flag else None
        if operator == "in":
            return True if value == "anywhere" else self._label_matches(message, value)
        if operator == "label":
            return self._label_matches(message, value)
        if operator == "from":
            return value in f"{message.sender.name or ''} {message.sender.email}".lower()
        if operator == "to":
            return any(value in a.email.lower() for a in [*message.to, *message.cc])
        if operator == "subject":
            return value in message.subject.lower()
        if operator == "category":
            categories = {label for label in message.labels if label.startswith("CATEGORY_")}
            return not categories if value == "primary" else f"CATEGORY_{value.upper()}" in categories
        return None  # newer_than:, has:, ... -- not modelled

    def _matches(self, message: EmailMessage, query: str) -> bool:
        for alternative in re.split(r"\s+OR\s+", query.strip()):
            matched = True
            for token in _TOKEN.findall(alternative):
                if token == "AND":
                    continue
                negate = token.startswith("-") and len(token) > 1
                result = self._term_matches(message, token[1:] if negate else token)
                if result is not None and result == negate:
                    matched = False
                    break
            if matched:
                return True
        return False

    async def search_messages(self, query: str, max_results: int = 25) -> list[EmailMessage]:
        self._enter("search_messages", query, max_results)
        found = [m for m in self.messages.values() if "DRAFT" not in m.labels and self._matches(m, query)]
        found.sort(key=lambda m: m.received_at.timestamp() if m.received_at else 0.0, reverse=True)
        return found[:max_results]

    # --- reads ----------------------------------------------------------------

    async def get_message(self, message_id: str) -> EmailMessage:
        self._enter("get_message", message_id)
        if message_id not in self.messages:
            raise MailboxError(404, f"Message '{message_id}' not found")
        return self.messages[message_id]

    async def get_thread(self, thread_id: str) -> EmailThread:
        self._enter("get_thread", thread_id)
        messages = sorted(
            (m for m in self.messages.values() if m.thread_id == thread_id),
            key=lambda m: m.received_at.timestamp() if m.received_at else 0.0,
        )
        if not messages:
            raise MailboxError(404, f"Thread '{thread_id}' not found")
        return EmailThread(thread_id=thread_id, subject=messages[0].subject, messages=messages)

    async def list_labels(self) -> list[Label]:
        self._enter("list_labels")
        return list(self.labels)

    # --- writes ---------------------------------------------------------------

    async def apply_label(self, message_id: str, label_id: str) -> None:
        self._enter("apply_label", message_id, label_id)
        if message_id not in self.messages:
            raise MailboxError(404, f"Message '{message_id}' not found")
        if label_id not in {label.label_id for label in self.labels}:
            raise MailboxError(400, f"Invalid label: {label_id}")
        if label_id not in self.messages[message_id].labels:
            self.messages[message_id].labels.append(label_id)
        self.applied_labels.append((message_id, label_id))

    async def create_draft(self, draft: DraftEmail) -> CreatedDraft:
        self._enter("create_draft", draft)
        draft_id = f"draft-{self._next_draft}"
        self._next_draft += 1
        message = EmailMessage(
            message_id=f"{draft_id}-message",
            thread_id=draft.thread_id or f"{draft_id}-thread",
            subject=draft.subject,
            sender=EmailAddress(email=self.own_email),
            to=draft.to,
            cc=draft.cc,
            body_text=draft.body_text,
            labels=["DRAFT"],
        )
        self.drafts[draft_id] = CreatedDraft(draft_id=draft_id, message=message)
        return self.drafts[draft_id]

    async def get_draft(self, draft_id: str) -> CreatedDraft:
        self._enter("get_draft", draft_id)
        if draft_id not in self.drafts:
            raise MailboxError(404, f"Draft '{draft_id}' not found")
        return self.drafts[draft_id]

    async def send_email(self, draft_id: str) -> EmailMessage:
        """Like Gmail, sending consumes the draft: a second send of it is a 404."""
        self._enter("send_email", draft_id)
        if draft_id not in self.drafts:
            raise MailboxError(404, f"Draft '{draft_id}' not found")
        message = self.drafts.pop(draft_id).message.model_copy(update={"labels": ["SENT"]})
        self.sent.append(message)
        return message
