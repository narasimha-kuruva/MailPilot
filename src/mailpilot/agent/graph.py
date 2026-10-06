"""The LangGraph workflow: a bounded ReAct-style loop over Gmail/RAG/intelligence MCP tools.

    user instruction -> agent (LLM decides) -> tool -> result -> agent -> ... -> final response

Branching and looping (Phase 4.4 -- "are there relevant emails? classify;
is it urgent? draft; ...") come from the LLM's own reasoning at each
`agent` turn, given the current tool results in `state["messages"]` and
clear tool descriptions/system prompt (see `mailpilot.prompts`) -- not from
a separate hand-built decision tree. This keeps one execution engine for
both a one-shot request ("search my inbox for X") and a multi-step goal
("find urgent client emails and draft replies"): the loop runs until the
model has no more tool calls, subject to the bounds below.

Two things stop the loop early, on purpose:

1. **The approval gate.** If the agent's next step is a sensitive tool
   (currently just `send_email`, see `mailpilot.safety.policy`), execution
   stops *before* the tool runs. The graph records the pending call in
   state and the run ends there -- nothing is sent.
   `LangGraphAgent.resume()` is the only code path that executes a
   sensitive tool, and only after an explicit human decision.
2. **Execution limits** (Phase 4.8, `AgentLimits`). A hallucinating or
   looping agent is bounded by max LLM turns, max tool calls, and a wall
   clock deadline -- LangGraph does not enforce any of this on its own; it
   is checked explicitly on every `agent` turn.

Tool failures are retried with bounded, classified backoff
(`mailpilot.resilience`, Phase 4.5) rather than either crashing the run or
retrying blindly forever. So are the model calls themselves (Phase 5): a
transient 503/429/timeout from the LLM API is retried within the run's
time budget, and a model call that fails for good ends the run with a
clear status and audit entry instead of an unhandled exception.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Awaitable, Callable, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from mailpilot.audit.service import AuditService
from mailpilot.config import Settings
from mailpilot.logging_config import get_logger
from mailpilot.mcp.base import MCPTool
from mailpilot.mcp.langchain_adapter import to_langchain_tool
from mailpilot.prompts import TOOL_RESULT_LABEL, wrap_untrusted
from mailpilot.resilience import describe_error, with_retries, with_timeout
from mailpilot.safety.policy import requires_approval
from mailpilot.schemas.agent import ApprovalStatus
from mailpilot.schemas.audit import AuditRecord, ToolCallStatus

logger = get_logger(__name__)

# Audit `tool_name` used when a run ends early, keyed by `GraphState.terminated_kind`.
TERMINATION_AUDIT_NAMES = {"limit": "__execution_limit__", "llm_error": "__llm_error__"}


class PendingToolCall(BaseModel):
    """A tool call the agent wants to make but that needs human sign-off."""

    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any]


@dataclass(frozen=True)
class AgentLimits:
    """Execution bounds (Phase 4.8). Defaults are conservative for a single
    interactive request; production values come from `Settings` via
    `from_settings()`."""

    max_steps: int = 20  # agent (LLM) turns per run
    max_tool_calls: int = 15  # total tool executions per run
    max_tool_retries: int = 2  # retries per individual tool call on transient errors
    tool_timeout_seconds: float = 30.0
    max_execution_seconds: float = 120.0
    max_output_chars: int = 12000  # cap on a single tool result / message text (a 5-message thread is ~5k)
    # Model calls (Phase 5): a transient 503/429/timeout from the LLM API is
    # retried with backoff instead of failing the run; each attempt is bounded.
    max_llm_retries: int = 3
    llm_retry_base_delay_seconds: float = 2.0
    llm_timeout_seconds: float = 60.0  # raise for slow local (CPU-only) models

    @classmethod
    def from_settings(cls, settings: Settings) -> "AgentLimits":
        """Every limit maps to the `agent_<name>` setting of the same name.

        One naming convention instead of three hand-maintained parallel
        lists (`Settings`, this class, the wiring in `api/deps.py`): a limit
        without a matching setting fails loudly at startup rather than
        silently falling back to the dataclass default.
        """
        return cls(**{item.name: getattr(settings, f"agent_{item.name}") for item in fields(cls)})


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_id: str
    instruction: str
    pending_approval: PendingToolCall | None
    step_count: int
    tool_call_count: int
    started_at: float
    terminated_reason: str | None
    terminated_kind: str | None  # a TERMINATION_AUDIT_NAMES key


def initial_graph_state(conversation_id: str, instruction: str, messages: list[AnyMessage]) -> GraphState:
    """Build the state dict passed to `graph.ainvoke()` for a new `run()` call."""
    return GraphState(
        messages=messages,
        conversation_id=conversation_id,
        instruction=instruction,
        pending_approval=None,
        step_count=0,
        tool_call_count=0,
        started_at=time.monotonic(),
        terminated_reason=None,
        terminated_kind=None,
    )


@dataclass
class AgentGraph:
    """The compiled graph, the raw chat model, and the tool-bound LLM.

    `chat_model` (unbound) is used for structured-output calls that aren't
    part of the tool-calling loop, e.g. `LangGraphAgent.plan()`.
    `llm_with_tools` drives the `agent` node and is also reused to compose
    the final reply after a human approval decision in `resume()`.
    `limits`/`sleep` are kept so `LangGraphAgent` applies the same model-call
    retry and timeout policy outside the graph (`resume()`, `plan()`).
    """

    graph: Any
    llm_with_tools: Any
    chat_model: Any
    limits: AgentLimits = field(default_factory=AgentLimits)
    sleep: Callable[[float], Awaitable[None]] | None = None


def serialize_result(result: Any) -> str:
    """Render a tool result as text the LLM can read back in a ToolMessage."""
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    if isinstance(result, list):
        return json.dumps(
            [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in result],
            default=str,
        )
    if result is None:
        return "null"
    return json.dumps(result, default=str)


def fence_tool_output(content: str, status: ToolCallStatus) -> str:
    """What the model reads back for a tool call (Phase 5.2).

    A successful result can carry email text -- a body from `read_email`, a
    thread, a subject line -- that someone other than the user wrote, so it
    is fenced as untrusted data, the same way the intelligence and drafting
    prompts fence what they insert. Error, skip, and hold messages are
    MailPilot's own text and stay as they are.
    """
    if status is ToolCallStatus.SUCCESS:
        return wrap_untrusted(TOOL_RESULT_LABEL, content)
    return content


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated to {max_chars} chars]"


def build_agent_graph(
    chat_model: BaseChatModel,
    tools: dict[str, MCPTool],
    audit_service: AuditService,
    checkpointer: BaseCheckpointSaver | None = None,
    limits: AgentLimits | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> AgentGraph:
    limits = limits or AgentLimits()
    sleep_kwargs = {"sleep": sleep} if sleep is not None else {}

    llm_with_tools = chat_model.bind_tools([to_langchain_tool(tool) for tool in tools.values()])

    async def agent_node(state: GraphState) -> dict[str, Any]:
        step_count = state["step_count"] + 1
        if step_count > limits.max_steps:
            return {
                "step_count": step_count,
                "terminated_reason": f"max_steps ({limits.max_steps}) exceeded",
                "terminated_kind": "limit",
            }

        deadline = state["started_at"] + limits.max_execution_seconds
        if time.monotonic() > deadline:
            return {
                "step_count": step_count,
                "terminated_reason": f"max_execution_seconds ({limits.max_execution_seconds}) exceeded",
                "terminated_kind": "limit",
            }

        # A transient failure from the model API (503 "high demand", a short
        # 429, a timeout) is retried with backoff instead of failing the whole
        # run -- but never past the run's wall-clock deadline. A call that
        # still fails ends the run cleanly via `terminate_node`, not as a 500.
        try:
            response: AIMessage = await with_retries(
                lambda: with_timeout(lambda: llm_with_tools.ainvoke(state["messages"]), limits.llm_timeout_seconds),
                max_retries=limits.max_llm_retries,
                base_delay_seconds=limits.llm_retry_base_delay_seconds,
                deadline=deadline,
                **sleep_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - reported through the run status and audit trail
            logger.error(
                "Language model call failed; ending run",
                extra={"extra_fields": {"conversation_id": state["conversation_id"], "error": describe_error(exc)}},
            )
            return {
                "step_count": step_count,
                "terminated_reason": f"the language model call failed: {describe_error(exc)}",
                "terminated_kind": "llm_error",
            }
        # Only plain-string content is truncated in place. Newer Gemini models
        # return a list of content blocks carrying "thought signatures" that
        # must be sent back verbatim on the next turn for tool calling to
        # work; rewriting those blocks would break the conversation. The
        # user-facing text is extracted (not mutated) by
        # `langgraph_agent._message_text`.
        if isinstance(response.content, str):
            response.content = _truncate(response.content, limits.max_output_chars)

        return {"messages": [response], "step_count": step_count}

    async def tools_node(state: GraphState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        tool_call_count = state["tool_call_count"]

        for call in last_message.tool_calls:
            tool_call_count += 1

            if tool_call_count > limits.max_tool_calls:
                status = ToolCallStatus.SKIPPED
                content = f"Tool call limit ({limits.max_tool_calls} per run) reached; '{call['name']}' was not executed."
            else:
                status = ToolCallStatus.SUCCESS
                try:
                    tool = tools[call["name"]]
                    args_model = tool.args_schema.model_validate(call["args"])

                    async def _call(_tool: MCPTool = tool, _args: dict = args_model.model_dump()) -> Any:
                        return await with_timeout(lambda: _tool.run(**_args), limits.tool_timeout_seconds)

                    result = await with_retries(_call, max_retries=limits.max_tool_retries, **sleep_kwargs)
                    content = _truncate(serialize_result(result), limits.max_output_chars)
                except Exception as exc:  # noqa: BLE001 - surfaced to the LLM & audit trail, not swallowed
                    status = ToolCallStatus.FAILURE
                    content = f"Error calling {call['name']}: {exc}"

            await audit_service.record(
                AuditRecord(
                    conversation_id=state["conversation_id"],
                    step_id=call["id"],
                    agent_request=state["instruction"],
                    tool_name=call["name"],
                    tool_args=call["args"],
                    status=status,
                    result_summary=content[:500],
                    approval_status=ApprovalStatus.NOT_REQUIRED,
                )
            )
            tool_messages.append(
                ToolMessage(content=fence_tool_output(content, status), tool_call_id=call["id"], name=call["name"])
            )

        return {"messages": tool_messages, "tool_call_count": tool_call_count}

    async def await_approval_node(state: GraphState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        calls = last_message.tool_calls
        sensitive_call = next(call for call in calls if requires_approval(call["name"]))
        pending = PendingToolCall(
            tool_call_id=sensitive_call["id"],
            tool_name=sensitive_call["name"],
            tool_args=sensitive_call["args"],
        )

        # Any other tool calls issued in the same turn are held, not executed,
        # so the sensitive action can't be worked around by batching it with
        # unrelated calls. Every tool_call_id still gets a ToolMessage so the
        # conversation stays well-formed for the next LLM call.
        held_messages: list[ToolMessage] = []
        for call in calls:
            if call["id"] == sensitive_call["id"]:
                continue
            held_messages.append(
                ToolMessage(
                    content=(
                        f"Held: not executed because '{sensitive_call['name']}' in this "
                        "turn requires human approval first."
                    ),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )
            await audit_service.record(
                AuditRecord(
                    conversation_id=state["conversation_id"],
                    step_id=call["id"],
                    agent_request=state["instruction"],
                    tool_name=call["name"],
                    tool_args=call["args"],
                    status=ToolCallStatus.SKIPPED,
                    result_summary="Held pending approval of a related action.",
                    approval_status=ApprovalStatus.NOT_REQUIRED,
                )
            )

        await audit_service.record(
            AuditRecord(
                conversation_id=state["conversation_id"],
                step_id=pending.tool_call_id,
                agent_request=state["instruction"],
                tool_name=pending.tool_name,
                tool_args=pending.tool_args,
                status=ToolCallStatus.SKIPPED,
                result_summary="Awaiting human approval before this action can run.",
                approval_status=ApprovalStatus.PENDING,
            )
        )
        return {"messages": held_messages, "pending_approval": pending}

    async def terminate_node(state: GraphState) -> dict[str, Any]:
        """End the run early -- an execution limit, or a model call that failed
        for good -- with an audit entry and a plain-language final message."""
        reason = state.get("terminated_reason") or "an execution limit was exceeded"
        kind = state.get("terminated_kind") or "limit"
        await audit_service.record(
            AuditRecord(
                conversation_id=state["conversation_id"],
                agent_request=state["instruction"],
                tool_name=TERMINATION_AUDIT_NAMES.get(kind, TERMINATION_AUDIT_NAMES["limit"]),
                status=ToolCallStatus.FAILURE,
                result_summary=reason[:500],
                approval_status=ApprovalStatus.NOT_REQUIRED,
            )
        )
        return {"messages": [AIMessage(content=f"Stopping this run: {reason}.")]}

    def route_after_agent(state: GraphState) -> str:
        if state.get("terminated_reason"):
            return "terminate"
        last_message = state["messages"][-1]
        calls = getattr(last_message, "tool_calls", None) or []
        if not calls:
            return END
        if any(requires_approval(call["name"]) for call in calls):
            return "await_approval"
        return "tools"

    graph = StateGraph(GraphState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("await_approval", await_approval_node)
    graph.add_node("terminate", terminate_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "await_approval": "await_approval", "terminate": "terminate", END: END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("await_approval", END)
    graph.add_edge("terminate", END)

    compiled = graph.compile(checkpointer=checkpointer)
    return AgentGraph(
        graph=compiled, llm_with_tools=llm_with_tools, chat_model=chat_model, limits=limits, sleep=sleep
    )
