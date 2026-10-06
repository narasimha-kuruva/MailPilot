"""`Agent` implementation backed by the LangGraph workflow in `mailpilot.agent.graph`.

Owns one compiled graph per process (keyed internally by `conversation_id`
via the graph's checkpointer) and the small map of conversations currently
blocked on a human decision. `run()` executes up to the next approval gate,
an execution limit, or completion; `resume()` is the only place a sensitive
tool (e.g. `send_email`) is ever actually invoked, and only after an
explicit, still-valid human decision (Phase 4.6 -- approvals expire after
`approval_ttl_seconds`).
"""

from __future__ import annotations

import time
from typing import Any, Callable
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from mailpilot.agent.base import Agent
from mailpilot.agent.graph import (
    TERMINATION_AUDIT_NAMES,
    AgentGraph,
    PendingToolCall,
    fence_tool_output,
    initial_graph_state,
    serialize_result,
)
from mailpilot.audit.service import AuditService
from mailpilot.gmail.client import GmailClient
from mailpilot.logging_config import get_logger
from mailpilot.mcp.base import MCPTool
from mailpilot.prompts import AGENT_SYSTEM_PROMPT, PLANNING_SYSTEM_PROMPT
from mailpilot.resilience import describe_error, with_retries, with_timeout
from mailpilot.safety.approval import ApprovalService
from mailpilot.schemas.agent import (
    AgentPlan,
    AgentRequest,
    AgentRunState,
    AgentRunStatus,
    ApprovalStatus,
    PendingApproval,
    PlannedStep,
    PlanOutline,
)
from mailpilot.schemas.audit import AuditRecord, ToolCallStatus

logger = get_logger(__name__)


class ApprovalExpiredError(ValueError):
    """The pending approval's TTL elapsed before a decision was made.

    A `ValueError` subclass so it still satisfies the `Agent.resume()`
    contract ("raises ValueError if there is no pending approval") for
    callers that only check the base type, while letting the API layer
    distinguish "expired" (410 Gone) from "never existed" (404) if it wants to.
    """


def _message_text(message: BaseMessage) -> str:
    """Extract the human-readable text from a message.

    Newer Gemini models return `content` as a list of typed blocks (e.g.
    `{"type": "text", "text": ..., "extras": {"signature": ...}}`) rather
    than a plain string; only the text parts belong in the user-facing
    response, never the raw block structure or opaque signatures.
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text" and block.get("text"):
                parts.append(str(block["text"]))
        return "".join(parts)
    return str(content)


class LangGraphAgent(Agent):
    def __init__(
        self,
        agent_graph: AgentGraph,
        tools: dict[str, MCPTool],
        approval_service: ApprovalService,
        audit_service: AuditService,
        approval_ttl_seconds: float = 1800.0,
        tool_timeout_seconds: float = 30.0,
        gmail_client: GmailClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._agent_graph = agent_graph
        self._tools = tools
        self._approval_service = approval_service
        self._audit_service = audit_service
        self._approval_ttl_seconds = approval_ttl_seconds
        self._tool_timeout_seconds = tool_timeout_seconds
        # Used only to enrich the human-facing approval description (e.g.
        # show the real To/Subject/Body of a draft before someone approves
        # sending it) -- never to execute anything itself.
        self._gmail_client = gmail_client
        self._clock = clock
        # conversation_id -> (pending call, monotonic time it was requested).
        # In-memory and single-process, matching the other Phase 2 services.
        self._pending_calls: dict[str, tuple[PendingToolCall, float]] = {}

    async def _invoke_model(self, call: Callable[[], Any]) -> Any:
        """One model call under the graph's LLM timeout/retry policy (see `agent_node`)."""
        limits = self._agent_graph.limits
        sleep_kwargs = {"sleep": self._agent_graph.sleep} if self._agent_graph.sleep is not None else {}
        return await with_retries(
            lambda: with_timeout(call, limits.llm_timeout_seconds),
            max_retries=limits.max_llm_retries,
            base_delay_seconds=limits.llm_retry_base_delay_seconds,
            **sleep_kwargs,
        )

    async def plan(self, request: AgentRequest) -> AgentPlan:
        """Decompose the request into an ordered list of sub-goals, without executing anything.

        This is a preview for the user/API caller, not a script the executor
        is bound to: `run()` always uses the live ReAct loop in
        `mailpilot.agent.graph`, which reacts to each tool's actual result
        rather than blindly following a plan made before any information
        was gathered.
        """
        conversation_id = request.conversation_id or str(uuid4())
        tool_catalog = "\n".join(f"- {tool.name}: {tool.description}" for tool in self._tools.values())
        structured = self._agent_graph.chat_model.with_structured_output(PlanOutline)
        prompt = [
            SystemMessage(content=PLANNING_SYSTEM_PROMPT.format(tool_catalog=tool_catalog)),
            HumanMessage(content=request.instruction),
        ]
        outline = await self._invoke_model(lambda: structured.ainvoke(prompt))
        steps = [
            PlannedStep(step_id=str(index), description=text)
            for index, text in enumerate(outline.steps)
        ]
        return AgentPlan(conversation_id=conversation_id, goal=request.instruction, steps=steps)

    async def run(self, request: AgentRequest) -> AgentRunState:
        conversation_id = request.conversation_id or str(uuid4())
        config = {"configurable": {"thread_id": conversation_id}}

        existing = self._agent_graph.graph.get_state(config)
        is_new_conversation = not existing.values.get("messages")
        messages: list[Any] = [HumanMessage(request.instruction)]
        if is_new_conversation:
            messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), *messages]

        result = await self._agent_graph.graph.ainvoke(
            initial_graph_state(conversation_id, request.instruction, messages),
            config=config,
        )
        return await self._state_from_result(conversation_id, result)

    async def resume(self, conversation_id: str, approved: bool) -> AgentRunState:
        entry = self._pending_calls.pop(conversation_id, None)
        if entry is None:
            raise ValueError(f"No pending approval for conversation '{conversation_id}'.")
        pending, requested_at = entry

        config = {"configurable": {"thread_id": conversation_id}}
        snapshot = self._agent_graph.graph.get_state(config)
        instruction = snapshot.values.get("instruction", "")

        if self._clock() - requested_at > self._approval_ttl_seconds:
            await self._approval_service.mark_expired(conversation_id, pending.tool_call_id)
            await self._audit_service.record(
                AuditRecord(
                    conversation_id=conversation_id,
                    step_id=pending.tool_call_id,
                    agent_request=instruction,
                    tool_name=pending.tool_name,
                    tool_args=pending.tool_args,
                    status=ToolCallStatus.SKIPPED,
                    result_summary=f"Approval request expired after {self._approval_ttl_seconds}s without a decision.",
                    approval_status=ApprovalStatus.EXPIRED,
                )
            )
            raise ApprovalExpiredError(
                f"Approval request for conversation '{conversation_id}' expired; the action was not executed."
            )

        await self._approval_service.record_decision(conversation_id, pending.tool_call_id, approved)

        if approved:
            tool = self._tools[pending.tool_name]
            try:
                # Bounded by the same tool timeout as every other tool call,
                # but never auto-retried -- see mailpilot.gmail.google_client
                # for why a send is never blindly retried.
                result = await with_timeout(lambda: tool.run(**pending.tool_args), self._tool_timeout_seconds)
                content = serialize_result(result)
                status = ToolCallStatus.SUCCESS
            except Exception as exc:  # noqa: BLE001 - surfaced to the LLM & audit trail, not swallowed
                content = f"Error calling {pending.tool_name}: {exc}"
                status = ToolCallStatus.FAILURE
            approval_status = ApprovalStatus.APPROVED
        else:
            content = "The user did not approve this action; it was not executed."
            status = ToolCallStatus.SKIPPED
            approval_status = ApprovalStatus.REJECTED

        await self._audit_service.record(
            AuditRecord(
                conversation_id=conversation_id,
                step_id=pending.tool_call_id,
                agent_request=instruction,
                tool_name=pending.tool_name,
                tool_args=pending.tool_args,
                status=status,
                result_summary=content[:500],
                approval_status=approval_status,
            )
        )

        tool_message = ToolMessage(
            content=fence_tool_output(content, status), tool_call_id=pending.tool_call_id, name=pending.tool_name
        )
        history = [*snapshot.values["messages"], tool_message]
        try:
            final_response: AIMessage = await self._invoke_model(
                lambda: self._agent_graph.llm_with_tools.ainvoke(history)
            )
        except Exception as exc:  # noqa: BLE001 - the decision's outcome above must survive this
            # The approved action already ran (or was skipped) and is audited
            # above. A wrap-up call that fails for good must not become a
            # 500: the client would retry /decision, find no pending
            # approval, and never learn that the email was in fact sent.
            logger.error(
                "Language model call failed after approval decision",
                extra={"extra_fields": {"conversation_id": conversation_id, "error": describe_error(exc)}},
            )
            await self._audit_service.record(
                AuditRecord(
                    conversation_id=conversation_id,
                    agent_request=instruction,
                    tool_name=TERMINATION_AUDIT_NAMES["llm_error"],
                    status=ToolCallStatus.FAILURE,
                    result_summary=f"Model call after the approval decision failed: {describe_error(exc)}"[:500],
                    approval_status=ApprovalStatus.NOT_REQUIRED,
                )
            )
            self._agent_graph.graph.update_state(config, {"messages": [tool_message], "pending_approval": None})
            if not approved:
                outcome = "was not executed, as you decided"
            elif status is ToolCallStatus.SUCCESS:
                outcome = "was executed successfully"
            else:
                outcome = "was attempted but failed"
            return AgentRunState(
                conversation_id=conversation_id,
                status=AgentRunStatus.COMPLETED,
                final_response=(
                    f"The '{pending.tool_name}' action {outcome} ({content[:200]}). "
                    "The assistant could not compose a follow-up because the language model call failed: "
                    f"{describe_error(exc)}. Submit a new instruction to continue."
                ),
            )

        final_text = _message_text(final_response)
        follow_up_calls = getattr(final_response, "tool_calls", None) or []
        if follow_up_calls:
            # `resume()` only executes the one approved action -- it does not
            # re-enter the full tool-execution/approval loop. If the model's
            # next turn wants to call more tools (including possibly another
            # sensitive one), those calls are NOT executed and NOT silently
            # dropped either: they're surfaced here so the caller can see a
            # follow-up is needed, and audited so it isn't invisible.
            names = ", ".join(call["name"] for call in follow_up_calls)
            note = (
                f"\n\n[MailPilot: the assistant also proposed calling {names} next. "
                "That was not executed -- submit a new instruction to continue.]"
            )
            final_text = f"{final_text}{note}"
            await self._audit_service.record(
                AuditRecord(
                    conversation_id=conversation_id,
                    agent_request=instruction,
                    tool_name="__deferred_followup__",
                    tool_args={"proposed_tools": [call["name"] for call in follow_up_calls]},
                    status=ToolCallStatus.SKIPPED,
                    result_summary=f"Model proposed further tool calls after approval ({names}); not executed.",
                    approval_status=ApprovalStatus.NOT_REQUIRED,
                )
            )

        self._agent_graph.graph.update_state(
            config, {"messages": [tool_message, final_response], "pending_approval": None}
        )

        return AgentRunState(
            conversation_id=conversation_id,
            status=AgentRunStatus.COMPLETED,
            final_response=final_text,
        )

    async def _build_approval_description(self, pending: PendingToolCall) -> str:
        """Best-effort, human-reviewable description of the pending action.

        For `send_email` specifically, this fetches the actual draft so the
        approver sees the real recipients/subject/body -- approving based on
        `{"draft_id": "..."}` alone would be a rubber stamp, not a review.
        """
        if pending.tool_name == "send_email" and self._gmail_client is not None:
            draft_id = pending.tool_args.get("draft_id")
            if draft_id:
                try:
                    draft = await self._gmail_client.get_draft(draft_id)
                    message = draft.message
                    to = ", ".join(addr.email for addr in message.to) or "(no recipients)"
                    body_preview = (message.body_text or "")[:500]
                    return (
                        f"Send email to: {to}\n"
                        f"Subject: {message.subject}\n\n"
                        f"{body_preview}"
                    )
                except Exception as exc:  # noqa: BLE001 - fall back to the generic description below
                    logger.warning(
                        "Could not fetch draft for approval preview",
                        extra={"extra_fields": {"draft_id": draft_id, "error": str(exc)}},
                    )

        return f"Approve '{pending.tool_name}' with arguments {pending.tool_args}?"

    async def _state_from_result(self, conversation_id: str, result: dict[str, Any]) -> AgentRunState:
        pending = result.get("pending_approval")
        if pending is not None:
            self._pending_calls[conversation_id] = (pending, self._clock())
            description = await self._build_approval_description(pending)
            await self._approval_service.request_approval(conversation_id, pending.tool_call_id, description)
            return AgentRunState(
                conversation_id=conversation_id,
                status=AgentRunStatus.AWAITING_APPROVAL,
                pending_approval=PendingApproval(
                    tool_name=pending.tool_name,
                    tool_args=pending.tool_args,
                    description=description,
                ),
            )

        final_message = result["messages"][-1]
        status = AgentRunStatus.FAILED if result.get("terminated_reason") else AgentRunStatus.COMPLETED
        return AgentRunState(
            conversation_id=conversation_id,
            status=status,
            final_response=_message_text(final_message),
        )
