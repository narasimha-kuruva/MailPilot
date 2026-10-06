"""The whole agent, wired as the API wires it, on a read-only request against the real inbox."""

from __future__ import annotations

import pytest

from mailpilot.api.deps import get_agent, get_audit_service
from mailpilot.schemas.agent import AgentRequest, AgentRunStatus
from mailpilot.schemas.audit import ToolCallStatus

pytestmark = pytest.mark.integration

WRITE_TOOLS = {"apply_label", "create_draft", "draft_grounded_reply", "send_email"}


@pytest.mark.asyncio
async def test_read_only_request_completes_against_the_real_inbox(gmail_client, chat_model) -> None:
    # The fixtures fail fast, with a clear message, if credentials are missing.
    agent = get_agent()

    state = await agent.run(
        AgentRequest(
            instruction="How many unread emails are in my inbox? Look at no more than 5.",
            conversation_id="integration-read-only",
        )
    )

    assert state.status is AgentRunStatus.COMPLETED, state.final_response
    assert state.final_response
    history = await get_audit_service().get_history("integration-read-only")
    assert any(r.tool_name == "search_emails" and r.status is ToolCallStatus.SUCCESS for r in history)
    assert not WRITE_TOOLS & {r.tool_name for r in history}
