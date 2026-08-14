"""Agent endpoints: submit an instruction, approve/reject pending actions, view history."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from mailpilot.agent.langgraph_agent import ApprovalExpiredError
from mailpilot.api.deps import AgentDep, AuditServiceDep
from mailpilot.schemas.agent import AgentRequest, AgentRunState, ApprovalDecision
from mailpilot.schemas.audit import AuditRecord

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/run", response_model=AgentRunState)
async def run_agent(request: AgentRequest, agent: AgentDep) -> AgentRunState:
    """Submit a natural-language instruction. May return AWAITING_APPROVAL."""
    return await agent.run(request)


@router.post("/{conversation_id}/decision", response_model=AgentRunState)
async def submit_decision(
    conversation_id: str, decision: ApprovalDecision, agent: AgentDep
) -> AgentRunState:
    """Approve or reject the action a run is currently AWAITING_APPROVAL for."""
    try:
        return await agent.resume(conversation_id, decision.approved)
    except ApprovalExpiredError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{conversation_id}/audit", response_model=list[AuditRecord])
async def get_audit_trail(conversation_id: str, audit_service: AuditServiceDep) -> list[AuditRecord]:
    """Full tool-call history for a conversation: request, tool, args, result, approval."""
    return await audit_service.get_history(conversation_id)
