"""Agent interface.

The concrete implementation (`mailpilot.agent.langgraph_agent.LangGraphAgent`,
Phase 2+) uses LangGraph to turn a natural-language `AgentRequest` into tool
calls, execute them step by step, and pause for human approval before any
sensitive action (e.g. sending email).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from mailpilot.schemas.agent import AgentPlan, AgentRequest, AgentRunState


class Agent(ABC):
    """Interface for the agent orchestration layer."""

    @abstractmethod
    async def plan(self, request: AgentRequest) -> AgentPlan:
        """Preview the next tool call(s) the agent would make, without executing them."""
        raise NotImplementedError

    @abstractmethod
    async def run(self, request: AgentRequest) -> AgentRunState:
        """Execute a request, pausing at `AWAITING_APPROVAL` if a sensitive
        action (see `mailpilot.safety.policy`) is next."""
        raise NotImplementedError

    @abstractmethod
    async def resume(
        self, conversation_id: str, approved: bool, *, approval_id: str | None = None
    ) -> AgentRunState:
        """Continue a run that is `AWAITING_APPROVAL` with the human's decision.

        `approval_id` (`PendingApproval.approval_id`) names the approval being
        decided. The HTTP API always passes it; leaving it out decides whatever
        is pending, which only a caller that just ran the conversation itself
        can know.

        Raises `ValueError` if that approval isn't pending for `conversation_id`.
        """
        raise NotImplementedError
