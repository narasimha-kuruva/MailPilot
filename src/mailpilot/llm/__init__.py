"""LLM provider selection (Phase 5): Gemini (hosted) or Ollama (local).

Everything downstream -- the LangGraph agent, the intelligence service,
reply drafting, RAG -- already depends only on LangChain's `BaseChatModel`
/ the `EmbeddingFunction` protocol, so switching providers is purely a
construction-time decision made here from `Settings.llm_provider`.
"""
