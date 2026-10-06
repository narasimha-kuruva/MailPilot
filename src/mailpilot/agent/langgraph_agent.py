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

import asyncio
import time
from typing import Any, Callable
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from mailpilot.agent.base import Agent
from mailpilot.agent.pending import InMemoryPendingCallStore, PendingCallStore, PendingToolCall
from mailpilot.agent.graph import (
    AgentGraph,
    fence_tool_output,
    initial_graph_state,
    serialize_result,
)
from mailpilot.audit.service import AuditService
from mailpilot.gmail.client import GmailClient
from mailpilot.logging_config import get_logger
from mailpilot.mcp.base import MCPTool
from mailpilot.observability.metrics import MetricsRegistry
from mailpilot.prompts import AGENT_SYSTEM_PROMPT, PLANNING_SYSTEM_PROMPT
from mailpilot.resilience import with_retries, with_timeout
from mailpilot.safety.approval import ApprovalService
from mailpilot.safety.idempotency import IdempotencyGuard, idempotency_key
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


def _decision_outcome(tool_name: str, approved: bool, status: ToolCallStatus) -> str:
    if not approved:
        return f"The '{tool_name}' action was not executed, as you decided."
    if status is ToolCallStatus.SUCCESS:
        return f"The approved '{tool_name}' action was executed successfully."
    if status is ToolCallStatus.SKIPPED:
        return f"The approved '{tool_name}' action was not executed again: it had already run."
    return f"The approved '{tool_name}' action was attempted but failed."


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
        clock: Callable[[], float] = time.time,
        idempotency_guard: IdempotencyGuard | None = None,
        metrics: MetricsRegistry | None = None,
        pending_store: PendingCallStore | None = None,
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
        # Wall-clock seconds: approval expiry must survive a restart when the
        # pending store is persistent.
        self._clock = clock
        # conversation_id -> (pending call, when it was requested). In memory
        # unless a persistent store is passed (see mailpilot.persistence).
        self._pending_store = pending_store or InMemoryPendingCallStore()
        # Shared by every conversation: an approved action runs at most once
        # (Phase 5.5, see mailpilot.safety.idempotency).
        self._idempotency = idempotency_guard or IdempotencyGuard()
        # Run outcomes only; tool calls and approvals reach the metrics
        # through the audit service, model usage through the model's callback.
        self._metrics = metrics

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

        existing = await self._agent_graph.graph.aget_state(config)
        is_new_conversation = not existing.values.get("messages")
        messages: list[Any] = [HumanMessage(request.instruction)]
        if is_new_conversation:
            messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), *messages]

        started = time.perf_counter()
        try:
            result = await self._agent_graph.graph.ainvoke(
                initial_graph_state(conversation_id, request.instruction, messages),
                config=config,
            )
            state = await self._state_from_result(conversation_id, result)
        except Exception:
            self._record_run(AgentRunStatus.FAILED, started)
            raise
        self._record_run(state.status, started)
        return state

    def _record_run(self, status: AgentRunStatus, started: float) -> None:
        if self._metrics is not None:
            self._metrics.record_run(status, (time.perf_counter() - started) * 1000)

    async def resume(self, conversation_id: str, approved: bool) -> AgentRunState:
        entry = await self._pending_store.take(conversation_id)
        if entry is None:
            raise ValueError(f"No pending approval for conversation '{conversation_id}'.")
        pending, requested_at = entry

        config = {"configurable": {"thread_id": conversation_id}}
        snapshot = await self._agent_graph.graph.aget_state(config)
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

        duration_ms: float | None = None
        if approved:
            content, status, duration_ms = await self._execute_approved(pending)
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
                duration_ms=duration_ms,
            )
        )

        tool_message = ToolMessage(
            content=fence_tool_output(content, status), tool_call_id=pending.tool_call_id, name=pending.tool_name
        )
        # Continue the run where the approval gate stopped it: the decision's
        # outcome goes in as if the tools node had produced it, and the graph
        # runs on from the agent node -- the same loop as run(), so follow-up
        # tool calls execute normally and another sensitive one stops at the
        # gate again (the result then comes back AWAITING_APPROVAL). The run
        # gets a fresh execution budget: the human may have taken a while.
        await self._agent_graph.graph.aupdate_state(
            config,
            {
                "messages": [tool_message],
                "pending_approval": None,
                "step_count": 0,
                "tool_call_count": 0,
                "started_at": time.monotonic(),
                "terminated_reason": None,
                "terminated_kind": None,
            },
            as_node="tools",
        )
        result = await self._agent_graph.graph.ainvoke(None, config=config)
        state = await self._state_from_result(conversation_id, result)
        if state.status is AgentRunStatus.FAILED:
            # The run stopped early after the decision, but what the decision
            # did must still be said: a client must never think an email that
            # went out didn't.
            state.final_response = f"{_decision_outcome(pending.tool_name, approved, status)} {state.final_response}"
        return state

    async def _execute_approved(self, pending: PendingToolCall) -> tuple[str, ToolCallStatus, float | None]:
        """Run an approved action once, unless that same action already ran or is running.

        Returns the result text, its status, and how long it ran (None if it didn't).
        """
        try:
            tool = self._tools[pending.tool_name]
            # Key on the validated arguments: extra fields the schema ignores
            # must not make the same send look like a different action.
            args = tool.args_schema.model_validate(pending.tool_args).model_dump()
        except Exception as exc:  # noqa: BLE001 - surfaced to the LLM & audit trail, not swallowed
            return f"Error calling {pending.tool_name}: {exc}", ToolCallStatus.FAILURE, None

        key = idempotency_key(pending.tool_name, args)
        if not self._idempotency.try_begin(key):
            if self._idempotency.completed_result(key) is None:
                reason = "the same action is still running from an earlier approval"
            else:
                reason = "this exact action already ran earlier"
            return (
                f"Not executed: {reason}. MailPilot never runs an approved '{pending.tool_name}' twice.",
                ToolCallStatus.SKIPPED,
                None,
            )

        # The action runs as its own task, and the reservation is settled when
        # that task really ends -- not when this call stops waiting for it. A
        # send that outlives the timeout is still running in its worker thread
        # (cancelling the wait can't stop it); releasing the reservation then
        # would let a second approval send the same draft concurrently.
        started = time.perf_counter()
        task = asyncio.ensure_future(tool.run(**args))
        task.add_done_callback(lambda done: self._settle(key, done))
        try:
            # Bounded by the same tool timeout as every other tool call, but
            # never auto-retried -- see mailpilot.gmail.google_client for why
            # a send is never blindly retried.
            result = await asyncio.wait_for(asyncio.shield(task), self._tool_timeout_seconds)
        except TimeoutError:
            content = (
                f"'{pending.tool_name}' did not finish within {self._tool_timeout_seconds}s and may still "
                "complete. It is not retried, and it can't be approved again until it has finished."
            )
            status = ToolCallStatus.FAILURE
        except Exception as exc:  # noqa: BLE001 - surfaced to the LLM & audit trail, not swallowed
            content = f"Error calling {pending.tool_name}: {exc}"
            status = ToolCallStatus.FAILURE
        else:
            content = serialize_result(result)
            status = ToolCallStatus.SUCCESS
        return content, status, (time.perf_counter() - started) * 1000

    def _settle(self, key: str, task: asyncio.Future) -> None:
        """Record how an approved action ended: done for good, or free to be approved again."""
        if not task.cancelled() and task.exception() is None:
            self._idempotency.complete(key, serialize_result(task.result())[:500])
        else:
            self._idempotency.abandon(key)

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
                    # Every recipient, CC included: an injected model could
                    # put a legitimate address in To and an outsider in Cc.
                    to = ", ".join(addr.email for addr in message.to) or "(no recipients)"
                    cc = ", ".join(addr.email for addr in message.cc)
                    body_preview = (message.body_text or "")[:500]
                    return (
                        f"Send email to: {to}\n"
                        + (f"Cc: {cc}\n" if cc else "")
                        + f"Subject: {message.subject}\n\n"
                        + body_preview
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
            await self._pending_store.put(conversation_id, pending, self._clock())
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
