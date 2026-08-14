"""Adapts `MCPTool`s to LangChain tool objects for LLM function-calling.

Used only to describe each tool's name/description/argument schema to
`BaseChatModel.bind_tools()`. Actual execution always goes through
`MCPTool.run()` directly from `mailpilot.agent.graph`, so argument
validation and audit logging live in exactly one place regardless of which
LLM framework is driving the call.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from mailpilot.mcp.base import MCPTool


def to_langchain_tool(tool: MCPTool) -> StructuredTool:
    async def _invoke(**kwargs: Any) -> Any:
        return await tool.run(**kwargs)

    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_invoke,
    )
