"""The configured real model: tool calling, structured output, usage reporting, embeddings."""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from mailpilot.config import Settings
from mailpilot.llm.providers import LLMProviderError, build_chat_model, build_embedding_function
from mailpilot.mcp.langchain_adapter import to_langchain_tool
from mailpilot.mcp.tools.search_emails import SearchEmailsTool
from mailpilot.observability.metrics import LLMUsageCallback, MetricsRegistry
from mailpilot.schemas.intelligence import EmailClassification
from tests.fakes import FakeGmailClient

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_model_calls_a_bound_tool(chat_model: BaseChatModel) -> None:
    model = chat_model.bind_tools([to_langchain_tool(SearchEmailsTool(FakeGmailClient()))])

    response = await model.ainvoke(
        [
            SystemMessage(content="You manage the user's Gmail with the tools provided."),
            HumanMessage(content="Find my unread emails."),
        ]
    )

    assert [call["name"] for call in response.tool_calls] == ["search_emails"]
    assert response.tool_calls[0]["args"].get("query")


@pytest.mark.asyncio
async def test_model_returns_structured_output(chat_model: BaseChatModel) -> None:
    structured = chat_model.with_structured_output(EmailClassification)

    result = await structured.ainvoke(
        "Classify this email. From: alice@client.com. Subject: Contract lapses Friday. "
        "Body: Please confirm the renewal price by Friday or the contract lapses."
    )

    assert isinstance(result, EmailClassification)
    assert result.reasoning


@pytest.mark.asyncio
async def test_model_reports_token_usage(settings: Settings) -> None:
    metrics = MetricsRegistry()
    model = build_chat_model(settings, callbacks=[LLMUsageCallback(metrics)])

    await model.ainvoke("Reply with the single word: ok")

    llm = metrics.snapshot().llm
    assert llm.calls == 1
    assert llm.input_tokens > 0 and llm.output_tokens > 0


@pytest.mark.asyncio
async def test_embeddings_return_a_vector(settings: Settings) -> None:
    embeddings = build_embedding_function(settings)
    try:
        vector = await embeddings.embed_query("quarterly contract renewal")
    except LLMProviderError as exc:  # the embedding model is optional (RAG tools only)
        pytest.skip(str(exc))

    assert len(vector) > 0 and any(value != 0 for value in vector)
