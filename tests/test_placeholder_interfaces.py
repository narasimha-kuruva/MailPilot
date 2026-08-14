"""Phase 1 sanity checks: every placeholder is an enforceable interface.

These guard against a Phase 2 implementation accidentally being partial —
if a required method is missing, instantiating the subclass will raise
`TypeError` because the base class is an ABC with abstract methods.
"""

from __future__ import annotations

import pytest

from mailpilot.agent.base import Agent
from mailpilot.audit.service import AuditService
from mailpilot.gmail.client import GmailClient
from mailpilot.mcp.base import MCPTool
from mailpilot.rag.service import RAGService
from mailpilot.safety.approval import ApprovalService

INTERFACES = [Agent, GmailClient, MCPTool, RAGService, ApprovalService, AuditService]


@pytest.mark.parametrize("interface", INTERFACES)
def test_interface_cannot_be_instantiated_directly(interface: type) -> None:
    with pytest.raises(TypeError):
        interface()
