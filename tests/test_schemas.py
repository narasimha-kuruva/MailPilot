from __future__ import annotations

from mailpilot.schemas.agent import AgentPlan, ApprovalStatus
from mailpilot.schemas.audit import AuditRecord, ToolCallStatus


def test_audit_and_plan_timestamps_are_timezone_aware() -> None:
    """`datetime.utcnow()` is deprecated and produced naive timestamps that
    compare/serialize ambiguously; both defaults must be explicit UTC."""
    record = AuditRecord(
        conversation_id="c",
        agent_request="r",
        tool_name="t",
        status=ToolCallStatus.SUCCESS,
        result_summary="",
        approval_status=ApprovalStatus.NOT_REQUIRED,
    )
    plan = AgentPlan(conversation_id="c", goal="g", steps=[])

    assert record.timestamp.tzinfo is not None
    assert record.timestamp.utcoffset().total_seconds() == 0
    assert plan.created_at.tzinfo is not None
