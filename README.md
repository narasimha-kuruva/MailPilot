# MailPilot

An agentic AI email automation assistant for Gmail. MailPilot interprets
natural-language instructions ("clear my inbox and reply to urgent client
emails"), plans a sequence of Gmail operations, retrieves relevant context
from past conversations and documents, drafts replies, and only sends email
after a human explicitly approves it.

This repository is being built incrementally.

- **Phase 1 (done):** project skeleton, configuration, logging, FastAPI app
  with a health check, and abstract interfaces for every major component.
- **Phase 2 (done):** Gmail OAuth, a real `GmailClient`, the seven Gmail MCP
  tools, a first LangGraph agent workflow, a human-approval gate in front of
  `send_email`, and audit logging of every tool call.
- **Phase 3 (done):** an `IntelligenceService` for classification, priority,
  urgency, thread summarization, and task extraction, all returned as
  structured Pydantic data; a ChromaDB-backed `RAGService` (chunking,
  embedding, ingestion, similarity search, metadata filtering); and a
  grounded reply-drafting workflow that retrieves context, generates a
  reply, and checks it for unsupported dates/amounts and invented
  recipients before creating a (still unsent) Gmail draft. Four new MCP
  tools expose this: `classify_email`, `summarize_thread`, `extract_tasks`,
  `draft_grounded_reply`.
- **Phase 4 and 5:** see the full architecture write-up below -- this
  document is rewritten comprehensively at the end of Phase 5 rather than
  incrementally, so treat the sections below as authoritative for the
  final state of the repository, not just Phase 1/2/3.

## Architecture

MailPilot is organized into seven loosely-coupled layers. Each is a
separate Python subpackage under `src/mailpilot/`, and higher layers depend
on lower layers only through abstract interfaces (Python `ABC`s), so any
implementation can be swapped — a different LLM provider, a different
vector store, a mock Gmail client for tests — without touching callers.

```
src/mailpilot/
├── main.py                      # FastAPI app factory + uvicorn entry point
├── config.py                     # Settings (pydantic-settings), read from env/.env
├── logging_config.py             # Structured (JSON) logging setup
│
├── api/                          # 1. API layer
│   ├── deps.py                    #    DI aliases; lazily builds real services
│   └── routes/
│       ├── health.py               #    GET /api/v1/health
│       └── agent.py                #    POST /agent/run, /{id}/decision, GET /{id}/audit
│
├── agent/                        # 2. Agent layer
│   ├── base.py                    #    Agent interface: plan() / run() / resume()
│   ├── graph.py                    #    LangGraph workflow + the approval gate
│   └── langgraph_agent.py          #    Agent implementation wrapping the graph
│
├── mcp/                           # 3. MCP tool layer
│   ├── base.py                     #    MCPTool interface
│   ├── langchain_adapter.py         #    MCPTool -> LangChain tool (for bind_tools)
│   └── tools/                       #    search_emails, read_email, read_thread,
│                                     #    list_labels, apply_label, create_draft,
│                                     #    send_email, registry.py
│
├── gmail/                         # 4. Gmail integration layer
│   ├── client.py                   #    GmailClient interface
│   ├── auth.py                      #    OAuth consent flow, token cache/refresh
│   ├── mime_utils.py                #    Gmail JSON <-> EmailMessage / raw MIME
│   └── google_client.py             #    GmailClient impl over googleapiclient
│
├── rag/                            # 5. RAG / context layer (interface only, Phase 3)
│   └── service.py
│
├── safety/                         # 6. Safety layer
│   ├── approval.py                  #    ApprovalService interface
│   ├── in_memory_approval.py         #    In-memory ApprovalService impl
│   └── policy.py                     #    Which tools require human approval
│
├── audit/                          # 7. Audit layer
│   ├── service.py                   #    AuditService interface
│   └── in_memory_audit.py            #    In-memory impl (+ structured log emit)
│
└── schemas/                        # Shared Pydantic models (no business logic)
    ├── health.py, email.py, agent.py, audit.py, rag.py
```

### How a request flows through the system

1. A client `POST`s a natural-language instruction to `/api/v1/agent/run`.
2. `LangGraphAgent.run()` invokes the **LangGraph** workflow
   (`agent/graph.py`): an `agent` node calls Gemini (bound to the seven MCP
   tools) to decide what to do next.
3. If the model calls a non-sensitive tool (`search_emails`, `read_email`,
   `read_thread`, `list_labels`, `apply_label`, `create_draft`), a `tools`
   node validates the arguments (Pydantic `args_schema`), calls the
   corresponding `MCPTool`, which delegates to `GmailClient`, and the result
   loops back to the `agent` node — repeating until the model has a final
   answer.
4. If the model calls `send_email`, the graph routes to `await_approval`
   instead and stops. The run returns `AWAITING_APPROVAL` with the pending
   action's tool name and arguments — **`send_email` is never executed at
   this point.**
5. A client `POST`s the human's decision to `/api/v1/agent/{id}/decision`.
   `LangGraphAgent.resume()` is the *only* code path that ever calls
   `GmailClient.send_email()`, and only when `approved: true`.
6. Every tool call — executed, held, pending, approved, or rejected — is
   recorded as an `AuditRecord` (request, tool, args, result, approval
   status, timestamp) via `AuditService`, retrievable at
   `GET /api/v1/agent/{id}/audit`.

### Why the approval gate isn't a LangGraph `interrupt_before`

LangGraph supports pausing a compiled graph via `interrupt_before=[...]`
and resuming with `graph.ainvoke(None, config)`. This implementation
deliberately does *not* use that mechanism for `send_email`. Instead,
`await_approval_node` ends the graph run cleanly (state carries
`pending_approval`), and `LangGraphAgent.resume()` explicitly re-reads the
checkpointed state, executes (or skips) the tool, appends the result, and
asks the model for one more turn. This keeps the one path that can
authorize sending an email fully explicit and easy to unit-test with fakes
(see `tests/agent/test_langgraph_agent.py`) instead of depending on
`interrupt`/resume-from-checkpoint semantics.

### Why interfaces first

Every layer above the schemas is defined as an abstract base class
(`Agent`, `GmailClient`, `MCPTool`, `RAGService`, `ApprovalService`,
`AuditService`). Phase 2 added concrete implementations for everything
except `RAGService`, but callers everywhere still depend on the interface
(`api/deps.py` wires the concrete classes in one place), so a different
Gmail provider, LLM, or persistence backend can be swapped in later without
touching the agent, MCP, or API layers.

## Dependencies

Declared in `pyproject.toml`; Phase 2 is the first phase to actually
exercise most of these:

| Package | Used for |
|---|---|
| `fastapi`, `uvicorn[standard]` | API layer |
| `pydantic`, `pydantic-settings`, `python-dotenv` | Schemas + env config |
| `langchain-core`, `langchain-google-genai` | Gemini chat model + tool binding |
| `langgraph` | The agent workflow (`agent/graph.py`) |
| `google-api-python-client`, `google-auth`, `google-auth-oauthlib`, `google-auth-httplib2` | Gmail OAuth + API calls |
| `chromadb` | Vector store — still unused, reserved for Phase 3 RAG |
| `mcp` | Reserved for exposing tools over the real MCP protocol — Phase 2 implements the tools as an internal `MCPTool` ABC + a LangChain adapter, not yet as an MCP server |

Dev-only: `pytest`, `pytest-asyncio`, `httpx` (FastAPI's `TestClient`).

## Running locally

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -e ".[dev]"
copy .env.example .env        # Windows; cp on macOS/Linux
```

Then fill in `.env`:

- `GOOGLE_API_KEY` — a Gemini API key from Google AI Studio. Required for
  `/api/v1/agent/*`; `/health` works without it.
- `GOOGLE_OAUTH_CLIENT_SECRETS_FILE` — path to an OAuth **Desktop app**
  client secrets JSON downloaded from Google Cloud Console (Gmail API must
  be enabled on that project). Required the first time any `/agent/*`
  route actually calls Gmail.
- `GOOGLE_OAUTH_TOKEN_FILE` — where the authorized user's token is cached
  after the one-time interactive consent flow (a browser window opens on
  first use; subsequent runs silently refresh the cached token).

Run the API:

```bash
python -m mailpilot.main
# or: uvicorn mailpilot.main:app --reload
```

- Health check: `GET http://localhost:8000/api/v1/health`
- Interactive docs: `http://localhost:8000/docs`
- Submit an instruction: `POST http://localhost:8000/api/v1/agent/run`
  `{"instruction": "search my inbox for unread emails from Alice"}`
- Approve/reject a pending action:
  `POST http://localhost:8000/api/v1/agent/{conversation_id}/decision`
  `{"approved": true}`
- Audit trail: `GET http://localhost:8000/api/v1/agent/{conversation_id}/audit`

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

28 tests, all runnable with no Gmail/Gemini credentials — Gmail is stubbed
with `FakeGmailClient` and the LLM with `FakeChatModel`
(`tests/fakes.py`), so agent/tool/graph logic is fully covered without
network access:

- `tests/test_health.py`, `tests/test_config.py` — Phase 1 checks, including
  that `require_approval_before_send` defaults to `True`.
- `tests/test_placeholder_interfaces.py` — every layer's ABC still can't be
  instantiated without a full implementation.
- `tests/gmail/test_mime_utils.py` — Gmail JSON parsing and raw-MIME
  building against fixture payloads.
- `tests/mcp/test_tools.py` — each MCP tool validates its args and
  delegates to `GmailClient` correctly.
- `tests/safety/test_policy.py` — only `send_email` requires approval.
- `tests/agent/test_langgraph_agent.py` — the important ones: a
  non-sensitive tool call runs end-to-end; a `send_email` call stops at
  `AWAITING_APPROVAL` **without sending**; `resume(approved=True)` sends
  and completes; `resume(approved=False)` **never** sends; every step is
  audited with the right status/approval fields.

## Assumptions made in Phase 2

- The approval gate is enforced entirely in-process (in-memory
  `ApprovalService` + a `conversation_id -> pending call` map inside
  `LangGraphAgent`). It does not survive a process restart. A production
  deployment would back this with a database, but the `ApprovalService`
  interface is unchanged either way.
- Gmail OAuth scope is `gmail.modify` (read/write/labels/send, excludes
  permanent delete), requested via the "Desktop app" OAuth client flow with
  `run_local_server` — appropriate for a local/dev deployment, not a
  server-side multi-user deployment (that would need a different OAuth
  flow, e.g. a web application client with a stored refresh token per
  user).
- MCP tools are implemented as an internal `MCPTool` ABC and adapted to
  LangChain tool objects for Gemini function-calling. They are not yet
  exposed over the actual Model Context Protocol (an MCP server); the `mcp`
  package dependency is reserved for that in a later phase.
- Multiple simultaneous tool calls in one LLM turn are supported: if any of
  them is `send_email`, the *entire* turn is held (not partially executed)
  until a decision is made, and every tool call still receives a
  `ToolMessage` so the conversation stays well-formed.
- No live Gmail account or Gemini API key was available in this
  environment, so `GoogleGmailClient` and the real `ChatGoogleGenerativeAI`
  integration are exercised by code review and by the fake-backed test
  suite, not by an end-to-end run against real Google services. Please
  verify against your own account before relying on it.

## Phase 3 (proposed, not yet implemented)

- RAG / context layer: implement `RAGService` against ChromaDB — chunking,
  embedding, ingestion of threads/documents, cited similarity retrieval —
  and give the agent a `retrieve_context` tool.
- Multi-step planning beyond a single tool call per turn for compound
  instructions ("clear my inbox and reply to urgent client emails"),
  likely by letting the graph loop with an explicit `AgentPlan` the agent
  checks off, rather than only ever reacting one tool call at a time.
- Persistent `ApprovalService` / `AuditService` (database-backed) so
  pending approvals and history survive a restart.
- Expose the MCP tools over an actual MCP server (`mcp` package) in
  addition to the current LangChain adapter, so other MCP-compatible
  clients can use them.
- Priority/triage classification and thread summarization as first-class
  agent capabilities (currently the agent can only do what a single Gemini
  turn's tool call accomplishes).
