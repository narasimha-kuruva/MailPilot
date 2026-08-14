"""Central policy for which tools are sensitive enough to require human approval.

This is the single source of truth the agent graph consults before
executing a tool call. Anything not listed here executes automatically;
anything listed here always routes to the approval gate first — see
`mailpilot.agent.graph`.
"""

from __future__ import annotations

SENSITIVE_TOOL_NAMES = frozenset({"send_email"})


def requires_approval(tool_name: str) -> bool:
    return tool_name in SENSITIVE_TOOL_NAMES
