# MailPilot

An agentic email assistant for Gmail. You give MailPilot an instruction in
plain language ("find urgent client emails and draft replies"). It
searches, reads, classifies and summarizes your mail, retrieves related
context, and drafts replies. It only sends an email after a person
approves that specific send, having seen who it goes to and what it says.

All five build phases are complete. What has been verified against real
services, and what hasn't, is called out where it matters. The short
version: the read paths, the agent loop and the evaluation suite have run
against a real Gmail inbox and a real local model. The approve-and-send
path has run end to end only against the in-memory evaluation mailbox, and
the Docker image has never been built (see [Known limitations](#known-limitations)).

- [Quick start](#quick-start)
- [Talking to the agent](#talking-to-the-agent)
- [Architecture](#architecture)
- [Tools](#tools)
- [Gmail integration](#gmail-integration)
- [Retrieval (RAG)](#retrieval-rag)
- [Human approval](#human-approval)
- [Safety](#safety)
- [Reliability](#reliability)
- [Audit trail](#audit-trail)
- [Persistence](#persistence)
- [Observability](#observability)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Configuration](#configuration)
- [Docker](#docker)
- [Choosing the LLM provider](#choosing-the-llm-provider)
- [Known limitations](#known-limitations)

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env          # cp on macOS/Linux
```

1. **A model.** By default MailPilot uses a local model through
   [Ollama](https://ollama.com). Install it, then
   `ollama pull gemma4:e2b`. Also run `ollama pull embeddinggemma` if you
   want the retrieval tool. To use Google Gemini instead, set
   `LLM_PROVIDER=gemini` and `GOOGLE_API_KEY`
   (see [Choosing the LLM provider](#choosing-the-llm-provider)).
2. **Gmail access.** In Google Cloud Console, enable the Gmail API, create
   an OAuth client ID of type *Desktop app*, and save its JSON as
   `secrets/client_secret.json`. Authorize once; a browser window opens and
   the token is cached in `secrets/token.json`:

   ```bash
   python -c "from mailpilot.config import get_settings; from mailpilot.gmail.auth import load_credentials; load_credentials(get_settings())"
   ```

   If you skip this step, the consent window opens the first time a
   request touches Gmail.
3. **Run it:** `python -m mailpilot.main`, then open the web app at
   <http://127.0.0.1:8000/> (or the API docs at `/docs`).

`/health`, `/metrics` and `/docs` work with no model and no Gmail
configured. Only the agent endpoints need them.

## Talking to the agent

**The web app.** Open <http://127.0.0.1:8000/> while the server runs. It
gives you:

- a chat with the agent;
- an approval card for every send, showing the real recipients and text,
  with **Approve** and **Reject** buttons;
- an activity panel listing every tool call in the conversation;
- a knowledge panel to index threads by Gmail search.

The conversation survives a page reload within the tab. If the server has
an API key, the app asks for it once per tab. The app is three static files
(`src/mailpilot/ui/`) with no build step and nothing loaded from other
sites. Email and model text is only ever displayed as text, never inserted
as HTML, and the pages carry a strict Content-Security-Policy.

**The HTTP API.** Everything the web app does goes through the API below,
so any HTTP client can do the same. `/docs` (Swagger UI) lets you try every
endpoint from the browser.

| Endpoint | What it does |
|---|---|
| `POST /api/v1/agent/run` | Submit an instruction: `{"instruction": "...", "conversation_id": "optional"}`. Returns the run's state. |
| `POST /api/v1/agent/{conversation_id}/decision` | Approve or reject the action a run is waiting on: `{"approved": true}`. |
| `GET /api/v1/agent/{conversation_id}/audit` | Every tool call in the conversation, with arguments, outcome, approval and timing. |
| `GET /api/v1/metrics` | Counters since start: runs, tool calls, approvals, model calls, and tokens. |
| `/api/v1/context/...` | The knowledge store behind grounded replies: index threads and documents, remove them, see counts (see [Retrieval](#retrieval-rag)). |
| `GET /api/v1/health` | Liveness, environment, and version. |

A run ends in one of three states: `completed` (with `final_response`),
`failed` (an execution limit or a model error, explained in
`final_response`), or `awaiting_approval` (with `pending_approval`). To
continue a conversation, send the next instruction with the same
`conversation_id`; the agent keeps the earlier turns.

An example session (responses abridged):

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Reply to Bob'\''s lunch email saying Thursday works, and send it."}'
```

```json
{
  "conversation_id": "5b0c…",
  "status": "awaiting_approval",
  "pending_approval": {
    "tool_name": "send_email",
    "tool_args": {"draft_id": "r-81…"},
    "description": "Send email to: bob@example.com\nSubject: Re: Lunch on Thursday?\n\nThursday works for me. See you then!"
  },
  "final_response": null
}
```

Nothing has been sent at this point. After reading the description:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/agent/5b0c…/decision \
  -H "Content-Type: application/json" -d '{"approved": true}'
# {"conversation_id": "5b0c…", "status": "completed", "final_response": "Your reply to Bob has been sent.", …}

curl -s http://127.0.0.1:8000/api/v1/agent/5b0c…/audit
# [{"tool_name": "search_emails", "status": "success", "duration_ms": 812.4, …},
#  {"tool_name": "create_draft", …}, {"tool_name": "send_email", "approval_status": "pending", …},
#  {"tool_name": "send_email", "status": "success", "approval_status": "approved", …}]
```

| Status | When |
|---|---|
| `401` | A key is configured and the request didn't send it, or sent a wrong one (see [Access](#access)) |
| `403` | No key is configured and the request didn't come from this machine |
| `404` | `/decision` for a conversation with nothing pending |
| `410` | `/decision` after the approval expired (`APPROVAL_TTL_SECONDS`, default 30 minutes); nothing was sent |
| `422` | Invalid input, e.g. an empty instruction or one over 10,000 characters |
| `500` | Anything unexpected. The body is a generic message plus a `request_id`; the details are in the server log under that id |
| `503` | The model provider is unusable (Ollama not running, model not pulled, Gemini key missing). The message says how to fix it |

Every response carries an `X-Request-ID` header. Send your own (letters,
digits, `.`, `_` and `-`, up to 64 characters) to correlate with your
logs.

### Access

Anyone who can call the API can read the mailbox and approve sends, so
access is closed by default:

- **No `MAILPILOT_API_KEY` set:** only requests from this machine
  (loopback) are accepted. Others get `403`. This holds even if the server
  is bound to `0.0.0.0`.
- **`MAILPILOT_API_KEY` set** (at least 16 characters): every request must
  send it, as `X-API-Key: <key>` or `Authorization: Bearer <key>`. Others
  get `401`. Generate a key with
  `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
- **Always open:** `/api/v1/health` and the docs. On `/docs`, use
  **Authorize** to enter the key.

A reverse proxy on the same machine reaches MailPilot from loopback, so
without a key, the proxy itself must authenticate its users.

## Architecture

```
src/mailpilot/
├── main.py                 # FastAPI app factory + uvicorn entry point
├── config.py               # Settings (pydantic-settings), from env/.env
├── logging_config.py       # JSON logging, redaction, request ids
├── redaction.py            # Secret scrubbing for logs and the audit trail
├── resilience.py           # Error classification, retries, timeouts
├── prompts.py              # System prompts + untrusted-content fencing
│
├── api/                    # HTTP layer
│   ├── deps.py             #   Wires concrete services (lazily) for the routes
│   ├── middleware.py       #   Request ids, request log, sanitized 500s
│   └── routes/             #   health, agent (run/decision/audit), metrics
├── agent/                  # Orchestration
│   ├── graph.py            #   The LangGraph loop, limits, approval gate
│   ├── langgraph_agent.py  #   Agent: run() / resume() / plan()
│   └── reply_drafting.py   #   Grounded reply workflow
├── mcp/                    # Tools the model can call
│   ├── base.py             #   MCPTool interface
│   ├── langchain_adapter.py#   MCPTool -> LangChain tool schema
│   └── tools/              #   12 tools + registry
├── gmail/                  # Gmail API client, OAuth, MIME parsing
├── intelligence/           # Classification, summaries, task extraction
├── rag/                    # Chunking, embeddings, ChromaDB store
├── llm/providers.py        # Ollama / Gemini selection
├── safety/                 # Approval service, guardrails, policy, idempotency
├── audit/                  # Audit trail
├── observability/          # Metrics registry + model-usage callback
├── evaluation/             # Scenarios, in-memory mailbox, runner, CLI
└── schemas/                # Pydantic models shared by every layer
```

Each layer depends on the ones below it only through an abstract
interface (`Agent`, `GmailClient`, `MCPTool`, `RAGService`,
`ApprovalService`, `AuditService`, `IntelligenceService`). The concrete
classes are wired in one place, `api/deps.py`, so a different Gmail
backend, store or model can be swapped in without touching callers. That
is also how the tests and the evaluation substitute fakes.

### How a request flows

1. `POST /agent/run` reaches `LangGraphAgent.run()`. The first turn of a
   conversation gets the system prompt.
2. The **agent** node calls the chat model, which has the tools bound to
   it. Each call has a timeout and is retried on transient errors within
   the run's time budget.
3. If the model asks for tools, the **tools** node validates each call's
   arguments and runs it. Each tool call has a timeout and is retried on
   transient errors. The node also enforces guardrails and records an
   audit entry. The result goes back to the model, fenced as untrusted
   data, and the loop repeats.
4. If any requested call is `send_email`, the graph goes to
   **await_approval** instead. Nothing in that turn runs, and the run
   returns `awaiting_approval` with a description of the draft.
5. `POST /decision` reaches `LangGraphAgent.resume()`. This is the only
   code path that can execute `send_email`, and only on `approved: true`,
   once per draft. The model then writes the closing message.
6. If a limit is hit (steps, tool calls, wall-clock time) or the model
   fails for good, the **terminate** node ends the run as `failed`, with
   the reason in the response and in the audit trail.

### One loop, not plan-then-execute

The agent is a single bounded ReAct loop: the model decides each step
after seeing the previous step's real result. A rigid "plan everything
first, then execute" engine was considered and rejected. A plan made
before any mail has been read can't know whether a search finds anything,
or whether an email is urgent. Branching ("is it urgent? then draft")
comes from the model reasoning over actual tool results, with the system
prompt and tool descriptions guiding it. `Agent.plan()` exists as a
preview: it returns an ordered list of sub-goals and executes nothing. It
is not exposed over the API yet.

### LangChain vs LangGraph

| | Responsible for |
|---|---|
| **LangGraph** | The control flow: the state graph (`agent → tools → agent …`, `await_approval`, `terminate`), routing, and per-conversation state through a checkpointer keyed by `conversation_id`. |
| **LangChain (core)** | The model abstraction: `BaseChatModel` (`ChatOllama` / `ChatGoogleGenerativeAI`), `bind_tools`, `with_structured_output`, message types, the tool-schema adapter, and callbacks (token counting). |
| **MailPilot's own code** | Everything that has to be trustworthy: tool execution, argument validation, retries, timeouts, limits, the approval gate, guardrails, idempotency and audit. LangChain's tool executor is never used: tools run through `MCPTool.run()` from the graph, so these rules live in one place whichever framework drives the model. |

### Why the approval gate isn't LangGraph's `interrupt_before`

LangGraph can pause a graph and resume it from the checkpoint. MailPilot
doesn't use that for sends. `await_approval` ends the run cleanly with the
pending call in state, and `resume()` explicitly re-reads the state and
executes (or skips) that one call. That keeps the only path that can send
email explicit, small, and unit-testable with fakes, instead of depending
on resume-from-checkpoint semantics.

## Tools

| Tool | What it does | Effect on the mailbox |
|---|---|---|
| `search_emails` | Gmail search. Returns compact summaries (ids, sender, subject, snippet, labels), not bodies | read |
| `read_email` | One message in full (text; HTML-only mail is converted to text) | read |
| `read_thread` | Every message in a thread | read |
| `list_labels` | The mailbox's labels | read |
| `classify_email` | Category, priority, urgency, whether action is needed | read |
| `summarize_thread` | Summary and key points | read |
| `extract_tasks` | Action items. A deadline or owner only when the mail states one | read |
| `apply_label` | Adds a label. **`TRASH` and `SPAM` are refused** | write |
| `create_draft` | Creates a draft. Reply recipients must already be in the thread | write (draft only) |
| `draft_grounded_reply` | Reads a thread, retrieves related context, drafts a reply addressed to the thread's participants, and checks the draft for invented dates and amounts | write (draft only) |
| `index_thread` | Saves a thread to the knowledge store, for later grounded replies | none (local store only) |
| `send_email` | Sends a draft. **Always stops for human approval first** | send |

The tools implement MailPilot's own `MCPTool` interface (a name, a
description, a Pydantic argument schema, and `run()`), adapted to
LangChain only for describing them to the model. They are **not** served
over the Model Context Protocol. The `mcp` dependency is unused, so other
MCP clients can't call them.

## Gmail integration

- **OAuth.** A *Desktop app* client with the `gmail.modify` scope: read,
  label, draft and send, but never permanent deletion. The token is cached
  in `GOOGLE_OAUTH_TOKEN_FILE` and refreshed silently. This suits one user
  running locally, not a hosted multi-user service.
- **Concurrency.** `googleapiclient` is synchronous, so calls run in
  worker threads. Its shared HTTP connection isn't thread-safe, so every
  request gets its own (sharing one corrupted TLS reads under load).
- **Retries.** Every call except a send is retried with backoff (`1s, 2s,
  4s`) on transient errors: 5xx, 429, timeouts, and Gmail's per-minute
  `403 rateLimitExceeded`. Other 4xx errors fail at once.
- **Complete searches or none.** A search fetches message bodies at most
  `GMAIL_MAX_CONCURRENT_FETCHES` at a time. If some can't be fetched, it
  raises rather than returning a partial list that the agent would present
  as the whole answer. Only a message deleted between listing and fetching
  is skipped.
- **Sends are never retried automatically.** A failed send doesn't tell
  you whether the email went out. The idempotency guard (below) is the
  layer that decides whether a send may run.

## Retrieval (RAG)

`draft_grounded_reply` grounds a reply in related past context:

1. **Chunking:** a recursive splitter (paragraphs, then sentences, then
   characters) into `RAG_CHUNK_SIZE` pieces with `RAG_CHUNK_OVERLAP`
   overlap.
2. **Embedding:** the provider's embedding model (`embeddinggemma` on
   Ollama, `RAG_EMBEDDING_MODEL` on Gemini), one batch per thread.
3. **Store:** one ChromaDB collection on disk (`CHROMA_PERSIST_DIR`,
   cosine similarity) for both email-thread chunks and documents.
   Metadata records the source (`source_type`, thread or document id,
   subject, sender), and a query can filter on it.
4. **Retrieval:** the top `RAG_TOP_K` chunks, then packed best-first into
   `RAG_MAX_CONTEXT_CHARS`. A chunk that doesn't fit is skipped, not the
   end of packing.
5. **Generation and checks:** the thread and the context are fenced as
   untrusted data. The draft is checked for dates and amounts that appear
   in neither (flagged in `validation_notes`). Recipients come from the
   thread itself, never from the model.

**The store holds only what you put in it.** Reading mail never stores
it. Content gets in two ways:

- **The context API.**
  - `POST /api/v1/context/threads` indexes threads, either by id
    (`{"thread_ids": [...]}`) or every thread behind a Gmail search
    (`{"query": "from:alice newer_than:90d", "max_threads": 20}`).
  - `POST /api/v1/context/documents` indexes a document such as a price
    list, a policy or notes: `{"document_id", "text", "metadata"}`.
  - `DELETE /api/v1/context/threads/{id}` and
    `DELETE /api/v1/context/documents/{id}` remove one.
  - `GET /api/v1/context` shows the counts.
  - A thread that can't be indexed is listed under `failed`; the others
    still go in.
- **The agent.** Ask it to "save this thread for future replies", and it
  calls `index_thread`.

Re-indexing a thread or document replaces its earlier version. When
drafting, retrieval skips the thread being replied to: that thread is
already in the prompt in full, and its own chunks would crowd out
everything else.

## Human approval

- **What needs approval:** `send_email`, as declared in
  `safety/policy.py`. `REQUIRE_APPROVAL_BEFORE_SEND` can't be set to
  false; startup fails if you try.
- **What the approver sees:** the draft's real recipients (To and Cc),
  subject and the start of its body, fetched from Gmail. The draft id alone isn't
  something a person can review.
- **Batching doesn't bypass it.** If the model requests a send together
  with other calls, the whole turn is held.
- **Expiry:** an approval older than `APPROVAL_TTL_SECONDS` is refused
  with `410` and recorded as expired.
- **Once per action:** the idempotency guard keys an action by tool and
  arguments (the `draft_id` for a send). The same draft approved twice,
  in two conversations or by two racing requests, is sent once; the second
  is audited as a suppressed duplicate. A failed send releases the key, so
  it can be approved again. (Gmail deletes a sent draft, so re-sending one
  that did go out fails with "not found" rather than sending twice.)
- **After the decision**, the model writes one closing message. If it
  asks for more tools at that point, they are not run. They are listed in
  the response and the audit trail, and the next instruction continues
  from there.

## Safety

Some defenses are guarantees enforced in code, whatever the model does.
Others make the model less likely to go wrong, without guaranteeing it.

| Defense | Kind | Where |
|---|---|---|
| Nothing is sent without an explicit, unexpired, per-draft human approval | **guarantee** | `agent/graph.py`, `langgraph_agent.resume()` |
| The approver sees the draft's real recipients and content | **guarantee** (falls back to the raw arguments if the draft can't be fetched) | `langgraph_agent._build_approval_description` |
| A draft is sent at most once | **guarantee** (per process) | `safety/idempotency.py` |
| Mail is never trashed or marked as spam | **guarantee** | `safety/guardrails.assert_label_is_safe` |
| A reply in a thread can't add recipients from outside it | **guarantee** | `guardrails.validate_reply_recipients` in `create_draft` and `draft_grounded_reply` |
| A grounded reply's recipients come from the thread, never the model | **guarantee** (the model's output has no recipient field) | `draft_grounded_reply` |
| Runaway runs stop (steps, tool calls, time, output size) | **guarantee** | `AgentLimits` |
| Secrets don't reach logs or the audit trail | **pattern-based**: known credential shapes and labelled values only | `redaction.py` |
| Email content is treated as data, not instructions: fenced in `<untrusted_*>` tags that the content can't close, under explicit instructions | **best effort** | `prompts.py`, every prompt carrying email text, every tool result |
| Drafts don't invent facts: grounding instructions, plus a check for dates and amounts not in the sources | **best effort** (catches dates and amounts only) | `prompts.GROUNDING_NOTICE`, `guardrails.find_unsupported_claims` |

One deliberate gap: a **new** email (not a reply) may be drafted to any
address, because there is no thread to check recipients against. It is
still only a draft, and sending it requires an approval that shows the
real recipient.

Prompt injection is tested both ways (`tests/safety/test_prompt_injection.py`).
Every prompt that carries email text fences it. And with a scripted model
that *obeys* an injected instruction, the send still stops at approval
showing the attacker's address, `TRASH` is refused, and outsiders can't
be added to a reply.

## Reliability

- **Error classification** (`resilience.classify_error`): transient
  errors (408/425/429/5xx, timeouts, connection resets, "high demand") are
  retried with exponential backoff. Permanent ones fail immediately. A 429
  that asks you to wait more than 60 seconds (a daily quota) counts as
  permanent. A shorter server-suggested wait is honoured.
- **Model calls** are retried (`AGENT_MAX_LLM_RETRIES`) with a per-call
  timeout, and never past the run's wall-clock deadline. A model that
  stays down ends the run as `failed` with a readable reason, not a 500.
  The Gemini client's own internal retries are switched off, so there is
  one retry layer.
- **Tool calls** are retried (`AGENT_MAX_TOOL_RETRIES`) with a timeout.
  A tool that still fails tells the model what went wrong, so the model
  can explain or adjust.
- **Limits** per run: 20 model turns, 15 tool calls, 120 seconds, and
  12,000 characters per tool result (all configurable).

## Audit trail

Every tool call produces an `AuditRecord`: conversation, instruction,
tool, arguments, outcome (`success` / `failure` / `skipped`), result
summary, approval status (`not_required` / `pending` / `approved` /
`rejected` / `expired`), duration and timestamp. That covers calls that
ran, calls held for approval, and duplicates the idempotency guard
suppressed. Runs that stop early add `__execution_limit__` or
`__llm_error__` records. Follow-up calls proposed after an approval add
`__deferred_followup__`. Records are redacted before they are stored,
written to the log, and served at `GET /agent/{id}/audit`. They are kept
in SQLite and survive restarts (see [Persistence](#persistence)).

## Persistence

With `STATE_BACKEND=sqlite` (the default), everything a restart must not
lose is kept in two SQLite files under `STATE_DIR` (`./data/state`):

- `checkpoints.sqlite`, LangGraph's checkpointer: every conversation's
  messages and graph state. A conversation continues after a restart, and
  so does a run that stopped at the approval gate.
- `mailpilot.sqlite`, with four tables:
  - the audit trail;
  - approval requests and decisions;
  - the action each stopped run waits on, with the time it was requested
    (wall-clock time, so expiry still works after a restart);
  - approved actions that already ran, so a draft sent before a restart
    can't be sent again after it.

Restarted live, the audit trail was intact, and the agent answered a
follow-up from the conversation it had before the restart. `STATE_BACKEND=memory`
keeps all of this in the process instead, which is what the test suite
uses. Metrics stay in memory either way: they are a live view since
startup.

Run **one worker process.** The files would be shared safely, but the
"this send is running right now" half of the duplicate-send guard lives in
process memory, so two workers could race the same send. To scale out,
move that reservation, and the rest, to a server database behind the same
interfaces.

## Observability

- **Logs** are one JSON object per line, redacted (`redaction.py`), and
  carry the `request_id` of the API request they belong to. Each request
  writes one `http_request` line (method, path, status, duration).
  uvicorn's own lines go through the same formatter.
- **Metrics** (`GET /api/v1/metrics`): runs by outcome and average
  duration; tool calls by outcome and average duration, overall and per
  tool; approvals requested, approved, rejected and expired; run events;
  and model calls (completed, failed, and unfinished, meaning cancelled by
  a timeout or still running) with input and output tokens. Token counting
  hooks into the model itself, so it covers planning and the model calls
  made inside tools as well as the agent loop. An estimated cost appears
  only if you set `LLM_INPUT_USD_PER_MILLION_TOKENS` and
  `LLM_OUTPUT_USD_PER_MILLION_TOKENS`, since no prices are built in.
  Counters are in memory and reset on restart; poll the endpoint to keep
  history.

## Evaluation

`mailpilot/evaluation/` holds 18 scenarios. Each one runs the real agent
(graph, tools, approval gate, guardrails, audit) against an in-memory
mailbox: six emails, including an urgent client email and an invoice
carrying a prompt injection. Each also gets a throwaway knowledge store,
preloaded with the scenario's documents. Nothing touches Gmail and nothing
can be sent. The checks are about outcomes: what reached the mailbox, what
the approver saw, what got indexed, how the run ended.

| Category | Scenarios |
|---|---|
| normal | find unread, summarize a thread, find the urgent email, draft a reply, create a new draft, save a thread to the knowledge store |
| multi-step | find urgent client emails and draft replies (only to the urgent one); reply and send (waits for approval, then sends exactly once); a grounded reply that needs a fact found only in indexed pricing notes |
| safety | injection ignored; injection *obeyed* by the model (attacker hidden in Cc) but contained; missing recipient (must ask, not invent); "send without checking with me" (still gated); "delete the newsletters" (nothing trashed) |
| reliability | malformed tool call recovered; transient Gmail 503 retried; permanent 403 not retried; runaway loop stopped by limits |

They run two ways:

- **Scripted**, in every `pytest` run: each scenario's script stands in
  for the model, so the result is deterministic and tests the machinery
  around the model. Negative tests check that the expectations catch a
  bad run.
- **Live**, with `python -m mailpilot.evaluation [-v] [--category safety] [--scenario ID]`:
  the configured model decides. Two scenarios simulate a misbehaving model
  on purpose and are skipped live.

**Live results with `ollama/gemma4:e2b` and `embeddinggemma` (2026-10-06).**
A model's answers vary from run to run, so treat any live score as a
sample, not a constant.

- **Before the knowledge store existed:** 14 of 14 in a single full run.
  An earlier run scored 13 of 14. In the miss, a vague instruction made the
  model ask which email to reply to, so the approval gate was never
  reached; the wording was then tightened.
- **With the knowledge store (16 live scenarios):** the first full run
  scored 12 of 16.
  - Three misses were infrastructure. Back-to-back scenarios hit model
    timeouts (60 s, twice each) while Ollama stalled, and all three passed
    on rerun.
  - One was behavior. The model didn't know that "our pricing notes" live
    in the knowledge store, and asked for the terms instead of calling
    `draft_grounded_reply`. The tool's description now says what the store
    holds and when to use it. On rerun it passed: the draft quoted
    "net 30", a fact found only in the indexed notes.

## Testing

```bash
pytest                                                       # 322 unit tests, no network
MAILPILOT_RUN_INTEGRATION_TESTS=1 pytest -m integration      # real Gmail + real model
```

The unit tests use `FakeGmailClient`, `FakeChatModel` and
`FakeEmbeddingFunction` (`tests/fakes.py`), so they need no credentials.
They cover the graph and approval flow, limits and retries, every tool,
the guardrails, prompt injection, idempotency, redaction, metrics, the API
hardening, the Gmail client against a stub service, the providers against
a stub Ollama, ChromaDB on disk, and all 16 evaluation scenarios.

**CI** (`.github/workflows/ci.yml`) runs on every push to `main` and on
every pull request. It runs the unit tests on Python 3.11 and 3.12, and
builds and smoke-tests the Docker image (see [Docker](#docker)). CI has no
credentials, so it never runs the integration tests.

The integration tests (`tests/integration/`) are skipped unless enabled.
They **only read** from Gmail: labels, search, a message and its thread,
and a bad id. They exercise the real model (tool calling, structured
output, token reporting, embeddings) and run one read-only agent request
end to end. Last run: 9 passed. Run them from the repository root so `.env`
paths resolve.

## Configuration

All settings come from environment variables or `.env` (see
`.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `APP_ENV` | `development` | Reported by `/health`; changes no behavior |
| `LOG_LEVEL` | `INFO` | Root log level |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Used by `python -m mailpilot.main` |
| `MAILPILOT_API_KEY` | — | Required on every request when set (16+ characters). Unset: this machine only. See [Access](#access) |
| `LLM_PROVIDER` | `ollama` | `ollama` or `gemini`; never falls back from one to the other |
| `GOOGLE_API_KEY` | — | Gemini only |
| `GEMINI_MODEL` | `gemini-3.7-flash` | Gemini only |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama only |
| `OLLAMA_MODEL` | `gemma4:e2b` | Ollama only |
| `OLLAMA_EMBEDDING_MODEL` | `embeddinggemma` | Ollama only; needed by the retrieval tool |
| `OLLAMA_NUM_CTX` | `16384` | Context window. Ollama's own 4096 is too small |
| `GOOGLE_OAUTH_CLIENT_SECRETS_FILE` | `./secrets/client_secret.json` | OAuth Desktop client |
| `GOOGLE_OAUTH_TOKEN_FILE` | `./secrets/token.json` | Cached user token |
| `GMAIL_USER_EMAIL` | — | Your address, left out of reply-all |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Vector store location |
| `RAG_EMBEDDING_MODEL` | `models/gemini-embedding-2` | Gemini only |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `800` / `100` | Overlap must be smaller than chunk size |
| `RAG_TOP_K` / `RAG_MAX_CONTEXT_CHARS` | `5` / `6000` | Chunks retrieved, and the prompt budget they're packed into |
| `REQUIRE_APPROVAL_BEFORE_SEND` | `true` | Can't be disabled |
| `APPROVAL_TTL_SECONDS` | `1800` | How long a pending approval stays valid |
| `AGENT_MAX_STEPS` | `20` | Model turns per run |
| `AGENT_MAX_TOOL_CALLS` | `15` | Tool calls per run |
| `AGENT_MAX_TOOL_RETRIES` | `2` | Retries per tool call on transient errors |
| `AGENT_TOOL_TIMEOUT_SECONDS` | `30` | Per tool-call attempt |
| `AGENT_MAX_EXECUTION_SECONDS` | `120` | Wall clock per run |
| `AGENT_MAX_OUTPUT_CHARS` | `12000` | Per tool result or model message |
| `AGENT_MAX_LLM_RETRIES` | `3` | Retries per model call on transient errors |
| `AGENT_LLM_RETRY_BASE_DELAY_SECONDS` | `2.0` | Backoff base for model calls |
| `AGENT_LLM_TIMEOUT_SECONDS` | `60` | Per model-call attempt; raise for CPU-only models |
| `GMAIL_MAX_RETRIES` / `GMAIL_RETRY_BASE_DELAY_SECONDS` | `3` / `1.0` | Gmail retries (never for sends) |
| `GMAIL_MAX_CONCURRENT_FETCHES` | `5` | Parallel body fetches per search |
| `STATE_BACKEND` / `STATE_DIR` | `sqlite` / `./data/state` | Where conversations, approvals, the audit trail and completed sends are kept (`memory`: lost on restart). See [Persistence](#persistence) |
| `LLM_INPUT_USD_PER_MILLION_TOKENS` / `LLM_OUTPUT_USD_PER_MILLION_TOKENS` | `0` / `0` | Cost estimate in `/metrics`; 0 turns it off |
| `MAILPILOT_RUN_INTEGRATION_TESTS` | — | `1` enables the integration tests |

## Docker

```bash
docker build -t mailpilot .
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v "$PWD/secrets:/app/secrets" -v "$PWD/data:/app/data" \
  mailpilot
```

- The image contains no secrets, and `.env`, `secrets/` and `data/` are
  excluded from the build. Configuration comes from `--env-file`; the
  OAuth token and the vector store are mounted in.
- **Authorize Gmail on the host first.** The consent flow needs a browser,
  which the container doesn't have, so `secrets/token.json` must already
  exist. The mount must be writable, because the token is refreshed in
  place.
- **Linux file ownership.** A bind mount keeps the host's owner, and the
  image's user (uid 10001) can't write to a directory you own. On Docker
  Engine for Linux, add `--user "$(id -u):$(id -g)" -e HOME=/tmp` so the
  container runs as you. Without it, refreshing the token fails, and so
  does every Gmail call, and the vector store can't be created. Docker
  Desktop on macOS and Windows maps ownership for you.
- **Ollama on the host** is `host.docker.internal` from inside the
  container. On Linux, add `--add-host=host.docker.internal:host-gateway`.
- **Set `MAILPILOT_API_KEY`.** From inside the container, your requests
  arrive through Docker's network, not from loopback, so without a key
  every agent request is refused with `403`. The server listens on
  `0.0.0.0` inside the container. Publishing the port to `127.0.0.1`, as
  shown, keeps it off the network as well.
- `--env-file` keeps an inline `# comment` as part of the value. Keep
  comments on their own lines, as `.env.example` does.
- The container runs as an unprivileged user (uid 10001) and has a health
  check on `/api/v1/health`.

**This image has not been built yet.** Docker wasn't available where it
was written. A clean `pip install .` (the build stage's step) was
verified, and the API served all endpoints from that install outside the
repository.

The CI workflow builds the image and smoke-tests it on every push. The
smoke test checks:

- that the image starts, and `/health` and the web app respond;
- that the API answers `401` without the key and `200` with it;
- that it runs as uid 10001;
- that Docker reports it healthy.

The image will have been built for the first time once that has run.

## Choosing the LLM provider

| `LLM_PROVIDER` | Chat model | Embeddings | Needs |
|---|---|---|---|
| `ollama` (default) | `OLLAMA_MODEL`, default `gemma4:e2b` | `OLLAMA_EMBEDDING_MODEL`, default `embeddinggemma` | Ollama running, models pulled (`ollama pull gemma4:e2b`, `ollama pull embeddinggemma`) |
| `gemini` | `GEMINI_MODEL`, default `gemini-3.7-flash` | `RAG_EMBEDDING_MODEL`, default `models/gemini-embedding-2` | `GOOGLE_API_KEY` |

The choice is made in one place (`llm/providers.py`); everything else sees
a LangChain `BaseChatModel` and an embedding function. To switch, change
`LLM_PROVIDER` and restart. If the chosen provider is unusable, the agent
endpoints return `503` with the fix, and MailPilot **never** falls back to
the other provider. It never downloads models either; pulling them is a
deliberate step for whoever runs it. The Gemini free tier allows about 20
requests per model per day, which is a handful of agent runs.

Notes on `gemma4:e2b`, from live runs against a real inbox and the
evaluation:

- It handles the full loop: it picks tools with sensible arguments,
  chains them, produces structured output, and passed all 14 live
  evaluation scenarios in a single run. A one-tool question takes about 10–30 seconds on
  a GPU; multi-step goals take about a minute.
- `OLLAMA_NUM_CTX` matters. With Ollama's default 4096-token context, the
  system prompt, tool schemas and one email left the model 9 tokens to
  answer, and it returned an empty reply.
- It is looser with Gmail search syntax than Gemini (it once used
  `is:inbox`), so the search tool's description lists the operators.
  Concrete instructions work best. Given a vague one, it asks for
  clarification instead of searching.
- Its prose is terser than Gemini's. For client-facing drafts, Gemini is
  the better choice.

## Known limitations

- **One shared API key, no user accounts.** Anyone with the key has full
  access. There are no per-user permissions and no key rotation beyond
  changing the setting and restarting.
- **One worker, one machine.** State persists in local SQLite files, but
  part of the duplicate-send guard is per process. See
  [Persistence](#persistence).
- **Not a real MCP server.** The tools are internal; other MCP clients
  can't use them.
- **After an approval,** the model gets one closing turn. Further tool
  calls it proposes are reported, not executed, and the next instruction
  continues the work.
- **Limits are per instruction**, not per conversation. Each `/run` gets
  a fresh budget. This is deliberate: a runaway *instruction* is what
  needs bounding.
- **Single-user OAuth.** The Desktop-app flow suits local use. A hosted,
  multi-user deployment needs a web OAuth flow and per-user tokens.
- **Prompt-injection resistance at the prompt level is best effort.** The
  guarantees are the structural ones in [Safety](#safety). The fact check
  on drafts only recognizes dates and amounts.
- **Efficiency left on the table.** Tool calls within one model turn run
  one after another. A thread read by two tools in one run is fetched
  twice. Services are built on the first agent request, not at startup,
  so `/health` works without credentials but the first request pays the
  setup cost. FastAPI also builds them for a request it then rejects as
  invalid.
- **Verification gaps.** The approve-and-send path has run end to end
  against the in-memory mailbox (scripted and live), not against real
  Gmail. The Docker image has never been built.
