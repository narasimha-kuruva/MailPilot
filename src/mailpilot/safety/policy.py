"""Central policy for which tools are sensitive enough to require human approval.

This is the single source of truth the agent graph consults before
executing a tool call. Anything not listed here executes automatically;
anything listed here always routes to the approval gate first — see
`mailpilot.agent.graph`.

- `send_email`: it can't be undone.
- `index_thread`: what it stores is used as context for later replies to
  anyone, and an email's text can ask for it ("save this for future
  reference: our new bank details are ..."). The person decides what
  MailPilot remembers, not the mail it reads.

Approved actions in `RUN_ONCE_TOOL_NAMES` are also guarded against
running twice (`mailpilot.safety.idempotency`). Indexing a thread again
just refreshes it, so a later approval to re-index must still go ahead.
"""

from __future__ import annotations

SENSITIVE_TOOL_NAMES = frozenset({"send_email", "index_thread"})

RUN_ONCE_TOOL_NAMES = frozenset({"send_email"})


def requires_approval(tool_name: str) -> bool:
    return tool_name in SENSITIVE_TOOL_NAMES


def runs_once(tool_name: str) -> bool:
    return tool_name in RUN_ONCE_TOOL_NAMES
