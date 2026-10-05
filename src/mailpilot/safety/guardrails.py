"""Concrete, code-level safety checks for generated content.

Prompt-level instructions (`mailpilot.prompts.GROUNDING_NOTICE`, telling
the model not to invent things) are necessary but not sufficient -- a model
can still be tricked or simply make a mistake. These checks catch specific
dangerous outcomes regardless of *why* the model produced them, and are
enforced in code, not just requested in a prompt.
"""

from __future__ import annotations

import re

from mailpilot.schemas.email import DraftEmail, EmailThread

_DATE_PATTERN = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}\b"  # ISO, e.g. 2026-09-01
    r"|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"  # e.g. 09/01 or 09-01-2026
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_MONEY_PATTERN = re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d+)?")


class GuardrailViolation(Exception):
    """Raised when a proposed action fails a hard safety guardrail."""


def validate_reply_recipients(draft: DraftEmail, thread: EmailThread) -> None:
    """A grounded reply must only go to addresses already present in the
    thread it's replying to. It must never introduce a new recipient the
    model invented -- or was tricked into adding, e.g. via a prompt
    injection in the email body asking to CC or redirect the reply.

    Raises `GuardrailViolation` if `draft` addresses anyone not found among
    the thread's senders/recipients.
    """
    known = {
        address.email.lower()
        for message in thread.messages
        for address in [message.sender, *message.to, *message.cc]
    }
    proposed = {address.email.lower() for address in [*draft.to, *draft.cc]}
    unknown = proposed - known

    if unknown:
        raise GuardrailViolation(
            f"Draft addresses recipients not present in the source thread: {sorted(unknown)}. "
            "Refusing to create a draft with an invented or injected recipient."
        )


def find_unsupported_claims(draft_body: str, source_texts: list[str]) -> list[str]:
    """Heuristic check: flag dates/amounts in `draft_body` that don't appear
    anywhere in the grounding sources.

    This is a best-effort signal, not a guarantee -- a fabricated fact that
    doesn't look like a date or a dollar amount won't be caught. It exists
    to catch the two most common, highest-consequence hallucination
    categories (invented deadlines and invented prices) cheaply, without an
    extra LLM call.
    """
    combined_sources = "\n".join(source_texts).lower()
    notes: list[str] = []

    for pattern, label in ((_DATE_PATTERN, "date"), (_MONEY_PATTERN, "amount")):
        for match in pattern.findall(draft_body):
            if match.lower() not in combined_sources:
                notes.append(f"Draft mentions a {label} ('{match}') not found in the source thread or retrieved context.")

    return notes


# Gmail system labels whose IDs are literal strings (not opaque ids). Applying
# either of these removes a message from the user's normal view -- TRASH
# auto-deletes after 30 days, SPAM hides it -- which is effectively a
# destructive action with no undo prompt. Other system labels (INBOX,
# STARRED, UNREAD, IMPORTANT, CATEGORY_*) only organise, so they stay allowed.
DANGEROUS_LABEL_IDS = frozenset({"TRASH", "SPAM"})


def assert_label_is_safe(label_id: str) -> None:
    """Refuse to apply a label that hides or deletes mail from the user.

    This is a hard block, not an approval gate (contrast
    `mailpilot.safety.policy.SENSITIVE_TOOL_NAMES`): a hallucinating or
    prompt-injected agent calling `apply_label(msg, "TRASH")` should simply
    fail, the same way `validate_reply_recipients` rejects an injected
    recipient. Trashing mail is not something MailPilot does on anyone's
    behalf, so there's nothing for a human to approve.

    Raises `GuardrailViolation` for `TRASH`/`SPAM` (case-insensitive).
    """
    if label_id.strip().upper() in DANGEROUS_LABEL_IDS:
        raise GuardrailViolation(
            f"Refusing to apply label '{label_id}': it would hide or delete the message. "
            "MailPilot never trashes or marks mail as spam automatically."
        )
