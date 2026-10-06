"""The app's real dependency wiring with the SQLite backend, through actual HTTP requests.

Regression test: LangGraph's SQLite checkpointer binds to the running event
loop when created, and FastAPI runs sync dependencies in a worker thread --
so building it inside a sync `get_agent` failed with "no running event loop"
on the first live request, although every unit test (which built it inside
async code) passed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from mailpilot.agent.graph import build_agent_graph
from mailpilot.agent.langgraph_agent import LangGraphAgent
from mailpilot.api import deps
from mailpilot.config import Settings
from mailpilot.main import create_app
from mailpilot.mcp.tools.registry import build_tools
from mailpilot.persistence.sqlite import SqliteCompletedActions, SqlitePendingCallStore
from mailpilot.safety.idempotency import IdempotencyGuard
from tests.conftest import local_client
from tests.fakes import FakeChatModel, FakeGmailClient


@pytest.fixture
def sqlite_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The real app and wiring, with only the model and Gmail replaced."""
    settings = Settings(_env_file=None, state_backend="sqlite", state_dir=str(tmp_path / "state"))
    monkeypatch.setattr(deps, "get_settings", lambda: settings)
    gmail = FakeGmailClient()
    model = FakeChatModel(
        [AIMessage(content="", tool_calls=[{"name": "send_email", "args": {"draft_id": "draft-1"}, "id": "c1"}])]
    )

    def build(checkpointer):  # stands in for deps._build_agent, which needs a real model
        state = deps.get_persistent_state()
        tools = build_tools(gmail)
        return LangGraphAgent(
            agent_graph=build_agent_graph(model, tools, deps.get_audit_service(), checkpointer=checkpointer),
            tools=tools,
            approval_service=deps.get_approval_service(),
            audit_service=deps.get_audit_service(),
            pending_store=SqlitePendingCallStore(state.database),
            idempotency_guard=IdempotencyGuard(SqliteCompletedActions(state.database)),
        )

    monkeypatch.setattr(deps, "_build_agent", build)
    for cached in (deps.get_persistent_state, deps.get_audit_service, deps.get_approval_service):
        cached.cache_clear()
    deps.reset_agent()
    yield local_client(create_app()), tmp_path / "state"
    state = deps.get_persistent_state() if deps.get_persistent_state.cache_info().currsize else None
    if state is not None:
        state.database.close()
    for cached in (deps.get_persistent_state, deps.get_audit_service, deps.get_approval_service):
        cached.cache_clear()
    deps.reset_agent()


def test_a_request_through_the_real_wiring_persists_its_state(sqlite_app) -> None:
    client, state_dir = sqlite_app

    response = client.post("/api/v1/agent/run", json={"instruction": "send the reply", "conversation_id": "c1"})

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "awaiting_approval"
    assert (state_dir / "checkpoints.sqlite").exists()
    audit = client.get("/api/v1/agent/c1/audit").json()
    assert [(r["tool_name"], r["approval_status"]) for r in audit] == [("send_email", "pending")]
