"""POST /api/v1/agent/plan: a preview of the steps, with nothing executed."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from mailpilot.agent.graph import build_agent_graph
from mailpilot.agent.langgraph_agent import LangGraphAgent
from mailpilot.api.deps import get_agent, get_audit_service
from mailpilot.audit.in_memory_audit import InMemoryAuditService
from mailpilot.main import create_app
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.safety.in_memory_approval import InMemoryApprovalService
from mailpilot.schemas.agent import PlanOutline
from tests.conftest import local_client
from tests.fakes import FakeChatModel, FakeGmailClient


def test_plan_returns_steps_and_touches_nothing() -> None:
    gmail = FakeGmailClient()
    tools = build_tools(gmail)
    audit = InMemoryAuditService()
    agent = LangGraphAgent(
        agent_graph=build_agent_graph(
            FakeChatModel([PlanOutline(steps=["Find unread client emails", "Classify them", "Draft replies to urgent ones"])]),
            tools,
            audit,
            checkpointer=MemorySaver(),
        ),
        tools=tools,
        approval_service=InMemoryApprovalService(),
        audit_service=audit,
    )
    app = create_app()
    app.dependency_overrides[get_agent] = lambda: agent
    app.dependency_overrides[get_audit_service] = lambda: audit
    client = local_client(app)

    response = client.post("/api/v1/agent/plan", json={"instruction": "reply to urgent client emails", "conversation_id": "p1"})

    assert response.status_code == 200
    body = response.json()
    assert body["goal"] == "reply to urgent client emails"
    assert [step["description"] for step in body["steps"]] == [
        "Find unread client emails",
        "Classify them",
        "Draft replies to urgent ones",
    ]
    assert gmail.calls == []  # nothing ran
    assert client.get("/api/v1/agent/p1/audit").json() == []


def test_plan_needs_the_same_access_as_run() -> None:
    client = local_client(create_app(), client=("203.0.113.7", 41000))

    assert client.post("/api/v1/agent/plan", json={"instruction": "anything"}).status_code == 403
