# MailPilot — Phase 3/4/5 Continuation Notes

**Purpose of this file:** if this session runs out of budget mid-Phase-5, a new
session can read this file plus the code itself and resume exactly where work
stopped, without re-deriving context. Delete this file once Phase 5 is fully
verified and the final report has been delivered.

**Last known state (2026-10-06):** all tests passing. Sections A, B, B2, B3,
C, C2, C3 are done and committed; section D is being worked through in order,
one commit per item (the user asked for "commit after each item"). See each
D item's DONE note for what landed.

---

## Status by phase

### Phase 1 — Foundation: DONE, verified
### Phase 2 — Gmail OAuth, MCP tools, LangGraph, approval gate, audit: DONE, verified
### Phase 3 — Intelligence & RAG: DONE, verified

- `schemas/intelligence.py`: `EmailClassification`, `Priority`, `EmailCategory`,
  `ThreadSummary`, `ExtractedTask`, `TaskExtractionResult`, `GroundedDraft`,
  `DraftGroundedReplyResult`.
- `prompts.py`: `UNTRUSTED_CONTENT_NOTICE`, `GROUNDING_NOTICE`, `wrap_untrusted()`,
  (later extended with `AGENT_SYSTEM_PROMPT`, `PLANNING_SYSTEM_PROMPT` in Phase 4).
- `intelligence/service.py` (ABC) + `intelligence/gemini_service.py` (impl via
  `chat_model.with_structured_output(...)`).
- `rag/chunking.py` (hand-rolled recursive splitter), `rag/embeddings.py`
  (`EmbeddingFunction` protocol + `GeminiEmbeddingFunction`), `rag/chroma_service.py`
  (`ChromaRAGService`, one Chroma collection for both threads and documents,
  `source_type` metadata field distinguishes them, cosine similarity).
- `safety/guardrails.py`: `validate_reply_recipients()` (reject any draft
  recipient not already in the source thread), `find_unsupported_claims()`
  (heuristic regex check for dates/amounts in a draft not present in the
  grounding sources).
- `agent/reply_drafting.py`: `draft_grounded_reply()` — read thread → RAG
  query → pack context by relevance/budget (`_pack_context`) → LLM structured
  output → validate → return `GroundedDraft`.
- New MCP tools: `classify_email`, `summarize_thread`, `extract_tasks`,
  `draft_grounded_reply` (registered only when `intelligence_service`/
  `rag_service`/`chat_model` are passed to `build_tools()` — keeps old
  callers that only pass `gmail_client` working unchanged).
- `config.py`: added `rag_*` settings.
- `api/deps.py`: added `get_embedding_function`, `get_rag_service`,
  `get_intelligence_service`, wired into `get_agent()`.
- Tests: `tests/rag/`, `tests/intelligence/`, `tests/safety/test_guardrails.py`,
  `tests/agent/test_reply_drafting.py`, `tests/mcp/test_intelligence_tools.py`.

### Phase 4 — Autonomous agent workflows: DONE, verified

Key design decision (documented in code comments, should also go in the final
README): **kept the existing ReAct loop (agent ⇄ tools) as the single
execution engine** rather than building a separate rigid plan-and-execute
state machine. Branching/looping (Phase 4.4) comes from the LLM's own
reasoning at each turn given real tool results, not a hand-built decision
tree — this is simpler, more flexible, and reuses 100% of the tested Phase 2
mechanism. `Agent.plan()` was upgraded to do genuine multi-step natural-
language decomposition via structured output (`PlanOutline`), but it's a
**preview only** — `run()` does not execute a fixed script, it reacts step by
step.

- `resilience.py` (new): `classify_error()` (transient vs permanent, checks
  HTTP status codes then exception type/message heuristics), `with_retries()`
  (bounded exponential backoff, injectable `sleep` for tests), `with_timeout()`
  (wraps `asyncio.wait_for`, converts timeout into `TimeoutClassifiedError`
  which classifies as transient).
- `agent/graph.py` heavily extended:
  - `AgentLimits` frozen dataclass: `max_steps`, `max_tool_calls`,
    `max_tool_retries`, `tool_timeout_seconds`, `max_execution_seconds`,
    `max_output_chars`.
  - `GraphState` gained `step_count`, `tool_call_count`, `started_at`,
    `terminated_reason`.
  - `agent_node` checks step/time limits before calling the LLM.
  - `tools_node` checks the tool-call limit per call, wraps each tool
    execution in `with_timeout` + `with_retries`.
  - New `limit_exceeded` terminal node + routing.
  - `AgentGraph` dataclass gained a `chat_model` field (raw, unbound) used by
    `plan()`.
  - `initial_graph_state()` helper.
- `schemas/agent.py`: `ApprovalStatus` gained `EXPIRED`, `CANCELLED`.
  Added `PlanOutline` schema.
- `safety/approval.py` / `in_memory_approval.py`: added `mark_expired()`.
- `agent/langgraph_agent.py`: `plan()` rewritten to use `PlanOutline`
  structured output; `run()` injects `AGENT_SYSTEM_PROMPT` only on the first
  turn of a new conversation (checked via `graph.get_state(config)` being
  empty); `resume()` checks approval TTL and raises `ApprovalExpiredError`
  (a `ValueError` subclass) if expired.
- `api/routes/agent.py`: catches `ApprovalExpiredError` → 410 Gone.
- `api/deps.py`: wires `AgentLimits` from `Settings`.
- Tests: `tests/test_resilience.py`, `tests/agent/test_limits_and_retries.py`,
  approval-expiry test in `tests/agent/test_langgraph_agent.py`, `plan()` test
  rewritten for `PlanOutline`.

### Phase 5 — Production hardening: IN PROGRESS

**What's done so far** (this sub-phase started by running the `code-review`
skill, which unexpectedly launched ~6 parallel background review agents
covering different angles — see findings section below):

1. ✅ `rag/chunking.py`: fixed crash when `overlap >= chunk_size` (was
   `range(0, n, chunk_size - overlap)` → `ValueError` or silently-dropped
   content when the step was 0/negative). Now clamps
   `overlap = max(0, min(overlap, chunk_size - 1))`.
2. ✅ `config.py`: added two `Settings` validators:
   - `require_approval_before_send` now **rejects** being set to `False`
     (raises `ValidationError`) — it was previously a decorative field that
     nothing read; the actual gate is `safety/policy.SENSITIVE_TOOL_NAMES`.
     Making it a hard-reject invariant means a misconfigured `False` fails
     loudly at startup instead of silently doing nothing.
   - `rag_chunk_overlap` must be `< rag_chunk_size` (model validator),
     so the chunking crash above can't be reached via real configuration
     even before the code-level clamp.
3. ✅ `gmail/client.py`: added abstract `get_draft(draft_id) -> CreatedDraft`
   to the `GmailClient` interface.
4. ✅ `gmail/google_client.py`: rewritten —
   - Added `get_draft()` implementation (`drafts().get(...)`).
   - Added a private `_run()` helper: `asyncio.to_thread(fn)` wrapped in
     `resilience.with_retries()` using `settings.gmail_max_retries` /
     `gmail_retry_base_delay_seconds` (these settings existed since the
     config pass but were previously never read by anything — now wired).
   - Every method except `send_email` now goes through `_run()`.
     `send_email` deliberately stays un-retried (documented why: a network
     error on send doesn't tell you whether it actually went out).
   - `search_messages`: per-message fetch now uses
     `asyncio.gather(..., return_exceptions=True)` and drops/logs individual
     failed fetches instead of failing the whole search if one message
     4 04s between the list and get calls.
5. ✅ `tests/fakes.py`: `FakeGmailClient` now tracks created drafts in
   `self._drafts` (keyed by auto-incrementing `draft-N` id) and implements
   `get_draft()` (raises `KeyError` for unknown ids, matching what a real
   404 would surface as).
6. ✅ `agent/langgraph_agent.py`: rewritten —
   - New `_build_approval_description()`: for `send_email` specifically,
     calls `gmail_client.get_draft()` (if a `gmail_client` was passed to the
     constructor) and builds a description showing the **real** To/Subject/
     Body preview, falling back to the old generic
     `"Approve 'x' with arguments {...}"` string if `gmail_client` is `None`
     or the lookup fails (logs a warning, never crashes). **This was the
     highest-severity finding**: previously a human "approving" a send only
     ever saw `{"draft_id": "abc123"}`, which is not reviewable — a
     prompt-injected recipient in `create_draft` could sail through a
     rubber-stamp approval undetected.
   - `resume()`'s approved-tool execution now wraps `tool.run(...)` in
     `with_timeout()` (it previously had no timeout at all, unlike every
     other tool call in `tools_node`). Still deliberately not wrapped in
     `with_retries` (same non-idempotent-send reasoning).
   - `resume()` now inspects `final_response.tool_calls` after the
     post-approval LLM call. If the model wants to call more tools
     (previously **silently discarded** — a real correctness bug: e.g. if
     it decided to also send a follow-up email or apply a label, that
     action vanished with no trace), it's now surfaced: appended as a note
     to `final_response` text, and recorded as a
     `__deferred_followup__` audit entry with the proposed tool names. Full
     re-entry into the graph's tool-execution loop from inside `resume()`
     was considered but rejected as too large/risky a change for this pass
     — documented as a known limitation (see below).
   - Constructor gained `tool_timeout_seconds` and `gmail_client` (optional)
     params.
7. ✅ `api/deps.py`: `get_agent()` now passes `gmail_client=get_gmail_client()`
   and `tool_timeout_seconds=settings.agent_tool_timeout_seconds` into
   `LangGraphAgent`.
8. ✅ Tests added/updated:
   - `tests/test_config.py`: `test_require_approval_before_send_cannot_be_disabled`,
     `test_rag_chunk_overlap_must_be_smaller_than_chunk_size`.
   - `tests/rag/test_chunking.py`: `test_overlap_equal_to_chunk_size_does_not_crash`,
     `test_overlap_greater_than_chunk_size_does_not_crash`.
   - `tests/agent/test_langgraph_agent.py`: `_build_agent()` helper now
     passes `gmail_client=gmail_client` to `LangGraphAgent`;
     added `test_approval_description_shows_real_draft_content_for_send_email`,
     `test_approval_description_falls_back_gracefully_when_draft_lookup_fails`.

   **Last full test run at this point: 74 passed** (before the two new
   approval-description tests were added — those were run standalone
   afterward, 9/9 passed in `tests/agent/test_langgraph_agent.py`; a full
   suite run has NOT yet been done since adding them — **do that first** in
   the next session).

---

## What's left in Phase 5 (in priority order)

### A. Safety fixes still pending (from code review findings, not yet done)

1. ✅ **DONE** (already implemented in code before this session resumed -- see
   `mcp/tools/create_draft.py`; the note below is stale.)
   ~~**`create_draft` has no recipient validation.**~~ Only
   `draft_grounded_reply` calls `validate_reply_recipients()` — the
   standalone `create_draft` MCP tool (`mcp/tools/create_draft.py`) lets the
   agent draft to *any* address with zero checking. Fix: in
   `CreateDraftTool.run()`, if `args.thread_id` is provided (i.e. this is a
   reply), fetch the thread via `self._gmail_client.get_thread(thread_id)`
   and call `validate_reply_recipients(draft, thread)` before calling
   `create_draft`. If `thread_id` is `None` (composing a brand-new email,
   not a reply), there's no known-safe recipient set to validate against —
   document this as an accepted, lower-risk path (it's still just a draft,
   never sent without separate approval, and the approval description fix
   above now means a human reviewing the send will see the real recipient).
   Let `GuardrailViolation` propagate uncaught (same pattern as
   `draft_grounded_reply.py`) so it surfaces through the normal tool-error
   handling in `tools_node`.
   Add a test: `create_draft` with `thread_id` set to a thread whose
   participants don't include the requested recipient → raises/fails with
   a guardrail message; a matching recipient succeeds.

2. ✅ **DONE (2026-10-05)** -- `apply_label` dangerous-label guardrail.
   `safety/guardrails.py` gained `DANGEROUS_LABEL_IDS = {"TRASH", "SPAM"}` and
   `assert_label_is_safe(label_id)` (case-insensitive, raises
   `GuardrailViolation`). `mcp/tools/apply_label.py` calls it before touching
   Gmail; its args now also have `min_length=1`. Hard block, not an approval
   gate, as planned. `classify_error()` treats the violation as PERMANENT so
   `tools_node` won't retry it. Tests: `tests/safety/test_guardrails.py`
   (reject/allow parametrized) and `tests/mcp/test_tools.py`
   (`FakeGmailClient` never receives the call). Suite: 88 passed.

### B. Correctness bugs found by review, not yet fixed

3. ✅ **DONE (2026-10-05)** -- `_pack_context` now `continue`s past a chunk that
   doesn't fit instead of `break`ing, so smaller lower-scored chunks still get
   packed; the "first chunk always included" behaviour is preserved. Tests in
   `tests/agent/test_reply_drafting.py`.

4. ✅ **DONE (2026-10-05)** -- reply-all. New
   `gmail/mime_utils.reply_all_recipients(message, own_email)` builds To =
   sender + original To, Cc = original Cc, deduped case-insensitively, minus
   the user's own address, never repeating a To in Cc, and falling back to
   the sender if To would be empty. `DraftGroundedReplyTool` uses it and takes
   an `own_email` ctor param, threaded through `build_tools(own_email=...)`
   from `Settings.gmail_user_email` in `api/deps.py` (if `GMAIL_USER_EMAIL` is
   unset the user may be included in To -- harmless, documented).
   `FakeGmailClient.set_thread_messages()` added so tests can use a
   multi-recipient thread. Suite: 96 passed.

### B2. Live-credential findings (2026-10-05, real Gemini key now available)

- `gemini-1.5-pro` and `text-embedding-004` are retired. Defaults changed to
  `gemini-3.7-flash` / `models/gemini-embedding-2` (3072 dims) in `config.py`,
  `.env.example`, `.env`. Verified live: plain, `with_structured_output`,
  `bind_tools`, and a full `LangGraphAgent.run()` with a real tool call.
  `gemini-3.5/3.6-flash` also work. Pro models return 429 "check plan and
  billing" (not on free tier). `gemini-3.8-flash` was 503 "high demand".
- **Free tier = 20 requests/day/model.** Burned through it while probing; the
  approval-gate + `resume()` round trip was NOT reached live (unit-tested only).
  Re-run `RUN 2` from this session's e2e script once quota resets or billing
  is enabled.
- Gemini 3.x returns `content` as a list of blocks with thought signatures.
  Fixed `_message_text` to extract only text (was `str(list)` -> signature
  blob leaked into `final_response`). `agent_node` deliberately does not
  truncate/mutate list content (signatures must round-trip). Test added.
- **New reliability gap found:** `agent_node`'s LLM call is NOT wrapped in
  `with_retries` (only tool calls are), so a single transient 503 from Gemini
  kills the whole run with a 500. Add to section C: wrap the
  `llm_with_tools.ainvoke` in `with_retries` (classify_error already treats
  503/429 as transient -- but a 429 *daily quota* 429 should probably NOT be
  retried; check the message for "quota" / RetryInfo before retrying).
- `GMAIL_USER_EMAIL` is set; `secrets/client_secret.json` still missing.

- ✅ **Live Gmail bug found & fixed (2026-10-05):** `googleapiclient`'s service
  object shares one `httplib2.Http`, which is not thread-safe. `search_messages`'
  concurrent per-message fetch (`asyncio.gather` over `to_thread`) collided on it
  -> sporadic `SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC`, silently dropped as
  "failed to fetch" (lost ~50% of results; `is:unread` returned 0 of 3).
  Fix: `GoogleGmailClient._execute(request)` runs every request with a fresh
  `AuthorizedHttp(creds, httplib2.Http())`; all 10 call sites go through it.
  Verified live: 35/35 fetched across three searches. Test:
  `tests/gmail/test_google_client.py` (stub service asserts distinct http per
  request). OAuth consent flow + token refresh also verified live.

- ✅ **Live agent bug found & fixed (2026-10-05):** first real-inbox run looped
  `search_emails` 9x until `max_execution_seconds` (limits worked as designed).
  Cause: `search_emails` returned full `EmailMessage`s incl. `body_html`; 5 real
  messages = ~29k chars vs the 4k `max_output_chars` cap -> model saw truncated,
  unparseable JSON and re-searched. Fixes: new `schemas.email.EmailSummary`
  (ids/sender/subject/snippet/labels/is_unread, no bodies) returned by
  `search_emails` (description tells the model to use read_* for content);
  `read_email`/`read_thread` drop `body_html`; `mime_utils.html_to_text()` +
  `parse_message` derives `body_text` from HTML for HTML-only mail (2 of 5 real
  inbox mails had no text part); `max_output_chars` default 4000 -> 12000.
  Verified live (gemini-3.6-flash): same instruction completes in 12s with one
  tool call, correct answer. Search result now ~2.9k chars.

### B3. Local LLM provider (Ollama) -- DONE, live-verified (2026-10-05)

- `LLM_PROVIDER=gemini|ollama` (`Settings.llm_provider`), `OLLAMA_BASE_URL`,
  `OLLAMA_MODEL` (default `gemma4:e2b`), `OLLAMA_EMBEDDING_MODEL` (default
  `embeddinggemma`, only needed by the RAG tools). New `mailpilot/llm/providers.py`:
  `build_chat_model()` / `build_embedding_function()` dispatch on the provider;
  `ensure_ollama_ready()` checks `GET /api/tags` and raises `LLMProviderError`
  ("install/start Ollama" or "run `ollama pull <model>`") -- MailPilot never
  downloads models. `api/deps.py` delegates to the factory and maps
  `LLMProviderError` -> 503 (the old missing-GOOGLE_API_KEY 503 is unchanged).
  Ollama embeddings check readiness on first use, so the agent starts without
  the embedding model. `langchain-ollama` added to pyproject. Tests in
  `tests/llm/test_providers.py` stub the Ollama endpoint (no Ollama needed).
- Everything downstream only sees `BaseChatModel` (`bind_tools` +
  `with_structured_output`), so no agent/tool/RAG code changed.
- **Ollama is now the DEFAULT provider** (`Settings.llm_provider = "ollama"`,
  `.env.example`, `.env`); Gemini stays fully supported via
  `LLM_PROVIDER=gemini`. No fallback in either direction.
- Live-verified with Ollama 0.35.1 + `gemma4:e2b` (4.6B, Q4_K_M, GPU):
  tool call with sensible args, continuation after a tool result, a
  search->read chain, structured output, and the read-only "list my 5 most
  recent inbox emails" run against the real inbox (16s, correct). Found and
  fixed: Ollama's default `num_ctx` is 4096 -> the read-and-summarise run
  ended with `done_reason=length` after 9 output tokens (prompt was 4085).
  New `OLLAMA_NUM_CTX` (default 16384) passed as `num_ctx` to `ChatOllama`.
  Observed limitation: it once used `is:inbox` instead of `in:inbox`
  (Gmail tolerated it) -> `search_emails` description now lists operators.
  Answers are terser than Gemini's. The approval/send path has NOT been run
  with the local model (read-only only, by instruction).

### C. Quality/efficiency fixes (lower priority, do if time remains)

5. DONE (2026-10-05) -- `rag/chroma_service.py`: `count`/`query`/`upsert` now run via
   `asyncio.to_thread`.
6. DONE (2026-10-05) -- `ingest_thread` collects every message's chunks first and
   makes ONE `embed_documents()` call; shared `_embed_and_upsert()` helper for
   thread + document ingest. Test asserts a single embedding batch for a
   3-message thread.
7. DONE (2026-10-05) -- `AgentLimits.from_settings(settings)` maps every field by
   the `agent_<field>` naming convention (`fields(cls)` + `getattr`), so a limit
   without a setting fails loudly; `api/deps.py` uses it. Test guards the
   convention.
8. DONE (2026-10-05) -- `min_length=1` on every required id/query/intent field
   across the tool arg schemas; parametrized test over all 8 tools.
9. DONE (2026-10-05) -- `datetime.now(timezone.utc)` in `schemas/agent.py` and
   `schemas/audit.py`; `tests/test_schemas.py`.

### C2. Model-call resilience (2026-10-05) -- DONE

Found live: a single transient 503 from Gemini killed the whole run with a 500.
- `resilience.py`: `status_code()` walks the `__cause__` chain (LangChain's
  Gemini wrapper raises its own error types *from* `google.genai` errors that
  carry `.code`; `ollama.ResponseError.status_code`); `retry_after_seconds()`
  parses Google's "Please retry in 11h27m32s" / `retryDelay`; a 429 whose
  suggested wait exceeds `MAX_RETRY_AFTER_SECONDS` (60s, i.e. a daily quota) is
  PERMANENT, a short one is TRANSIENT and the wait is honoured;
  `is_retryable` (langchain_core ModelError) honoured; httpx error names and
  "high demand"/"try again later" tokens added; `with_retries(deadline=...)`
  never sleeps past the run's wall-clock budget; `describe_error()`.
- `agent/graph.py`: `agent_node` wraps the model call in
  `with_retries(with_timeout(...))` using new `AgentLimits.max_llm_retries` /
  `llm_retry_base_delay_seconds` / `llm_timeout_seconds` (settings
  `AGENT_MAX_LLM_RETRIES`, `AGENT_LLM_RETRY_BASE_DELAY_SECONDS`,
  `AGENT_LLM_TIMEOUT_SECONDS`). A model call that still fails ends the run via
  the (renamed) `terminate` node -> `AgentRunStatus.FAILED`, final_response
  "Stopping this run: the language model call failed: ...", audit tool_name
  `__llm_error__` (`TERMINATION_AUDIT_NAMES`; limits still `__execution_limit__`).
  `AgentGraph` now carries `limits` + `sleep`.
- `agent/langgraph_agent.py`: `_invoke_model()` applies the same policy to
  `plan()` and to `resume()`'s wrap-up call. If that wrap-up call fails AFTER an
  approved send, the result is still COMPLETED with text stating the action was
  executed, `__llm_error__` audited, and the graph state advanced (no 500, no
  phantom "no pending approval" on a client retry). Test covers it.
- Verified live: real daily-quota 429 classifies PERMANENT (no retry sleep) and
  the run ends FAILED in ~1s with a readable reason; a real read-only run on
  gemini-3.6-flash completes through the wrapper.
- Still open (D16): a global exception handler for anything else unhandled.

### C3. Two more found while verifying C2 live (2026-10-05) -- DONE

- **Gemini client's own retries.** `ChatGoogleGenerativeAI.max_retries` defaults
  to 6 -> google-genai `HttpRetryOptions(attempts=6)` retrying 429/503 with
  ~31s of exponential backoff *inside* the client, before our layer sees
  anything. A daily-quota 429 therefore took 35.9s to surface. Fix:
  `llm/providers.py` builds the Gemini model with `max_retries=1` (the SDK
  treats 0 as "use defaults"; the LangChain docstring itself says set 1 and do
  custom retries). `mailpilot.resilience` is now the only retry layer;
  `agent_max_llm_retries` default raised 2 -> 3 to compensate (2s, 4s, 8s).
- **Gmail per-minute query-cost quota.** Bursts of concurrent `messages.get`
  calls get `403 Forbidden` reason `rateLimitExceeded` ("Quota exceeded for
  quota metric 'Total Query Cost' and limit 'Units per minute per user'").
  `classify_error` treated every 403 as PERMANENT, and `search_messages`
  silently dropped the failed fetches: a 50-message search returned 6, and
  the agent would have presented that as the whole inbox. Fixes:
  `resilience.is_rate_limited()` (429, or 403 with rate-limit wording) ->
  TRANSIENT; "Retry after <ISO timestamp>" parsed into `retry_after_seconds`;
  `search_messages` throttles body fetches with a semaphore
  (`GMAIL_MAX_CONCURRENT_FETCHES=5`), drops a message ONLY on 404 (deleted
  between list and get), and otherwise raises `GmailSearchIncompleteError`
  (cause chained) instead of returning a partial list. Gmail retry defaults
  `GMAIL_MAX_RETRIES=3`, base delay 1.0s. `search_emails` tool default
  `max_results` 25 -> 10, max 100 -> 50 (every result costs quota).
  Tests in `tests/gmail/test_google_client.py`, `tests/test_resilience.py`.
  Live re-check: 4x25 complete (~5s each, throttled); a 50-search under a
  saturated quota raised the incomplete error instead of returning 37/50; the
  next 50-search (window moved on) fetched 50/50 in 9.5s. Refinement after
  that run (the failing search took 43s because each of 13 failing fetches
  spent its own backoff): once one fetch confirms a rate limit the queued
  fetches fail immediately, and `GmailSearchIncompleteError.is_retryable =
  False` so `tools_node` doesn't re-run the whole search against the same
  quota -- the agent sees the error at once.

### D. Not yet started at all (full Phase 5 checklist items)

10. ✅ **DONE (2026-10-06)** -- Phase 5.1 redaction + safe logging. New
    `mailpilot/redaction.py`: `redact_text()` (by shape: `AIza` API keys,
    `ya29.` access tokens, `1//` refresh tokens, `GOCSPX-` client secrets,
    JWTs, `Bearer` headers; by label: `access_token=`, `"client_secret":`,
    `password:` ...), `redact_value()` (recursive; values under
    `SENSITIVE_KEYS` replaced outright unless empty). Deliberately no generic
    "long random string" rule (would mangle Gmail ids / thought signatures).
    `StructuredFormatter.format()` redacts the whole payload (message,
    exception, extra_fields) and anything rendered via `default=str`.
    `InMemoryAuditService.record()` stores the record redacted (so
    `GET /agent/{id}/audit` never serves a secret back); the `AuditService`
    docstring makes that a requirement for future implementations. The LLM
    still sees unredacted tool results (only logs/audit are scrubbed).
    Tests: `tests/test_redaction.py` (24).

11. ✅ **DONE (2026-10-06)** -- Phase 5.2 prompt injection. Writing the tests
    found a real gap: tool results in the agent loop (e.g. a `read_email`
    body) reached the model UNFENCED, although the system prompt told it to
    distrust `<untrusted_*>` content. Now `graph.fence_tool_output()` wraps
    every successful tool result in `<untrusted_tool_result>` (in
    `tools_node` and `resume()`); errors/holds stay plain (our own text);
    the audit summary stays unwrapped. `AGENT_SYSTEM_PROMPT` says so.
    `wrap_untrusted()` now neutralizes ANY closing `untrusted_*` tag
    (case-insensitive, spacing-tolerant), not just the exact own tag.
    Tests: `tests/safety/test_prompt_injection.py` (16) -- fence can't be
    closed from inside; every prompt that carries email text fences it
    (agent loop, classify/summarize/extract, reply drafting incl. retrieved
    context); `GroundedDraft` has no recipient field; and with a scripted
    model that OBEYS the injection: send stops at approval showing the
    attacker's address and nothing is sent, TRASH is blocked, an outsider
    can't be added to a thread reply. `FakeGmailClient.set_message()` added.
    Live-checked with Ollama/gemma4:e2b: reads fenced results fine.

12. ✅ **DONE (2026-10-06)** -- Phase 5.5 idempotency guard.
    `safety/idempotency.py`: `idempotency_key(tool, args)` (sorted-key JSON)
    and `IdempotencyGuard` with `try_begin` / `complete` / `abandon` /
    `completed_result`. Differs from the sketch below on purpose: it also
    tracks IN-FLIGHT actions, so two approvals racing for the same draft
    can't both send (the sketch only marked completion after the await).
    `abandon()` on failure or cancellation (in a `finally`), so a failed
    send can be approved again (Gmail deletes a sent draft, so a send that
    did go out fails "not found" rather than sending twice).
    `LangGraphAgent(idempotency_guard=...)`, default one per agent (shared
    across conversations); `resume()` now calls `_execute_approved()`; a
    suppressed duplicate is audited SKIPPED + APPROVED with the reason, and
    the model is told it was not run again. Tests:
    `tests/safety/test_idempotency.py` (7) incl. two conversations
    approving the same draft (sequential and concurrent) -> one send.

13. **Phase 5.7 Observability.** Add `mailpilot/observability/metrics.py`
    with a dependency-free `MetricsRegistry` (counters: agent runs total/
    succeeded/failed, tool calls total/failed, approvals requested/approved/
    rejected/expired; plus average tool-call duration). Add
    `duration_ms: float | None = None` to `AuditRecord`
    (`schemas/audit.py`) and measure it around tool execution in
    `agent/graph.py`'s `tools_node` (and in `resume()`'s approved-call path).
    Wire an optional `metrics: MetricsRegistry | None` into
    `InMemoryAuditService` so every `record()` call updates it — this
    reuses the existing single choke point rather than scattering metrics
    calls through `graph.py`. Add `GET /api/v1/metrics` returning a snapshot.
    For LLM token usage: check `getattr(response, "usage_metadata", None)`
    on `AIMessage` responses in `agent_node` (real `ChatGoogleGenerativeAI`
    populates this; `FakeChatModel` won't, so tests degrade gracefully to
    "no usage tracked" rather than erroring) and feed into the registry if
    present. For cost estimation, add `gemini_input_price_per_1k_tokens` /
    `gemini_output_price_per_1k_tokens` settings **defaulting to 0.0**
    (cost estimation off unless the user fills in real current pricing —
    do NOT hardcode a specific $ figure as fact, pricing changes and can't
    be verified from within this environment).

14. **Phase 5.8 Evaluation framework.** Add
    `mailpilot/evaluation/scenarios.py`: a data-only list of scenario
    definitions (id, category: normal/multi-step/safety/reliability,
    instruction, fake setup, expected-behavior assertions) covering at
    least: find unread emails, summarize a thread, find urgent emails,
    draft a reply, create a draft; find urgent client emails and draft
    replies (multi-step); prompt injection inside an email; missing
    recipient; send without approval (must be impossible — assert the
    architecture prevents it, don't just assert a test passes); malformed
    tool output; Gmail API failure (transient → retried; permanent → not).
    Then `tests/evaluation/test_scenarios.py` parametrizes pytest over
    those scenario definitions and runs them against the fake-backed agent.
    This satisfies "create an agent evaluation suite" concretely and stays
    runnable via plain `pytest` (no separate framework/dependency needed).

15. **Phase 5.9 Testing — separate integration tests.** Add a pytest marker
    `integration` (register in `pyproject.toml`'s `[tool.pytest.ini_options]`
    via `markers = ["integration: requires real Gmail/Gemini credentials"]`).
    Add `tests/integration/test_real_gmail.py` with tests marked
    `@pytest.mark.integration`, skipped by default unless an env var like
    `MAILPILOT_RUN_INTEGRATION_TESTS=1` is set (use a module-level
    `pytestmark = pytest.mark.skipif(...)`). Document in the README how to
    run them (`pytest -m integration`) and that they need real
    `GOOGLE_API_KEY` + completed OAuth. These should NOT run in the normal
    `pytest` invocation used throughout this project.

16. **Phase 5.10 API hardening.** In `main.py`, add a global exception
    handler (`@app.exception_handler(Exception)`) that catches anything
    unhandled, logs it server-side with full detail, and returns a
    sanitized JSON response (generic message + a request id, no stack
    trace, no secrets) — currently an unhandled exception would leak a
    default FastAPI/Starlette traceback in debug scenarios. Add a small
    request-ID middleware (generate a UUID per request, stash in
    `request.state`, include as `X-Request-ID` response header and in log
    records via the existing `extra_fields` mechanism in
    `logging_config.py`). Double check all `HTTPException` usages return
    sensible status codes (already mostly done: 503 missing API key, 404
    missing pending approval, 410 expired approval — add 400 for
    `GuardrailViolation` if/when it can bubble up to a route, 422 comes
    automatically from Pydantic validation).

17. **Phase 5.11 Docker.** Add `Dockerfile` (multi-stage: builder installs
    deps with `pip install .`, final stage copies venv/site-packages, runs
    as a non-root user, `CMD ["uvicorn", "mailpilot.main:app", "--host",
    "0.0.0.0", "--port", "8000"]`, no secrets baked in — `.env` mounted or
    env vars passed at `docker run` time). Add `.dockerignore` (`.venv/`,
    `.git/`, `data/`, `secrets/`, `.env`, `__pycache__/`, `.pytest_cache/`,
    this file itself, test files if you want a slimmer image).

18. **Phase 5.12 Full README rewrite.** Do this LAST, after everything above
    is implemented and verified, so it documents what's actually true. Must
    cover: complete architecture (update the layer diagram — RAG is no
    longer "interface only"), agent execution flow (the ReAct-loop-not-
    plan-and-execute decision and why), LangChain vs LangGraph responsibility
    split, MCP tool architecture (11 tools now, still an internal `MCPTool`
    ABC + LangChain adapter, not a real MCP server — note as a limitation),
    Gmail integration (OAuth scope, retry behavior), RAG architecture
    (chunking/embedding/Chroma/metadata/filtering), human-in-the-loop flow
    (including the blind-approval fix, TTL/expiry, idempotency guard),
    safety architecture (guardrails, prompt injection defenses — structural
    AND prompt-level, be honest about what's a guarantee vs best-effort),
    audit logging, observability/metrics, evaluation strategy, testing
    strategy (unit vs integration split), environment configuration (all
    `.env.example` vars), Docker usage, and an explicit "Known limitations"
    section. An ASCII architecture diagram is fine (matches the existing
    README style) — don't reach for an image.

### E. Final steps (after all of the above)

19. Run the **complete** test suite, confirm green.
20. Boot the app (`uvicorn mailpilot.main:app`), hit `/health`, `/docs`,
    `/openapi.json`, `/api/v1/metrics`, and `/api/v1/agent/run` without an
    API key (expect 503) — same smoke-test pattern used after every prior
    phase in this session (see any `curl ... /api/v1/health` command earlier
    in the conversation for the exact incantation, including how the
    background `uvicorn` process was started/stopped on Windows via
    `netstat -ano | grep LISTENING` + `powershell Stop-Process`).
21. Do one more `code-review` pass (or manual read-through) focused only on
    the *new* Phase 5 code, since the six-agent review this session covered
    Phase 3/4 code plus early Phase 5 — the later Phase 5 additions
    (idempotency, metrics, evaluation, API hardening, Docker) won't have
    been reviewed yet.
22. Write the FINAL REPORT the original task requested — 17 numbered
    sections (final project structure, phase summaries, architecture,
    execution flow, MCP architecture, LangChain vs LangGraph split, RAG
    architecture, human-in-the-loop flow, security mechanisms, reliability
    mechanisms, evaluation scenarios, test results, Docker/deployment
    instructions, remaining limitations, key technical decisions + why).
    Do not claim anything works that wasn't actually verified — this
    project has no real Gmail/Gemini credentials available in this
    environment, so say so explicitly wherever it's relevant (Gmail API
    live calls, real embedding calls, actual token/cost tracking).

---

## Code review findings — full disposition list

Six background agents ran (triggered by invoking the `code-review` skill
mid-Phase-5). Below is every finding and what happened to it, so nothing
gets silently re-litigated or silently forgotten in a future session.

**Fixed (see "What's done so far" above):** blind send_email approval
description; resume() missing timeout; require_approval_before_send
decorative; gmail_max_retries/gmail_retry_base_delay_seconds unused;
chunk_text crash on overlap>=chunk_size; search_messages one-failure-kills-
the-whole-search.

**Confirmed real, not yet fixed** (see section A/B above): create_draft no
recipient validation; apply_label no dangerous-label guardrail;
_pack_context breaks too early; draft_grounded_reply drops non-sender
recipients (no reply-all); resume() silently dropped follow-up tool_calls
(this one IS fixed already — see item 6 in "What's done so far").

**Confirmed real, fix planned but not done** (see section C): ChromaDB sync
calls not in `asyncio.to_thread`; `ingest_thread` embeds per-message instead
of batched; `AgentLimits`/`Settings`/`deps.py` triple-mapping;
`datetime.utcnow()` deprecated.

**Deliberately deferred — documented as known limitations, do NOT attempt
without a good reason, each was a considered trade-off:**

- **Per-`run()` (not per-conversation-lifetime) execution limits.** One
  reviewer noted `AgentLimits` resets every `/agent/run` call even for the
  same `conversation_id`, so a long-lived conversation could rack up
  unbounded *cumulative* tool calls across many turns even though each
  individual turn is bounded. This is judged an intentional design choice
  (each user instruction gets its own execution budget, matching how most
  production agent systems work — a "runaway agent" within one instruction
  is what needs bounding, not a legitimately long back-and-forth
  conversation), NOT a bug. Just make sure the README documents this
  explicitly rather than leaving it implicit.
- **`resume()` doesn't fully re-enter the graph's tool-execution loop.**
  Partially addressed (follow-up tool_calls are now surfaced, not silently
  dropped — item 6 above), but a full fix (routing back into the real
  `tools`/`await_approval` graph nodes from inside `resume()`) was judged
  too large/risky a change for this pass. If picked up later: the cleanest
  approach is probably to make `resume()` call `graph.update_state(...)`
  with the tool message, then re-invoke a *second* internal graph
  (re-entrant from the `agent` node) rather than calling
  `llm_with_tools.ainvoke()` directly the way it does now — needs careful
  thought about the checkpointer state transitions, don't rush it.
- **MCPTool boilerplate across 10 files** (`args =
  self.args_schema.model_validate(kwargs)` repeated identically). A base
  template method would remove duplication but touches every tool file —
  skipped as stylistic-only, no bug, real risk of breaking something across
  10 files under time pressure for marginal benefit.
- **Double validation**: `tools_node` in `graph.py` validates tool args via
  Pydantic, then each tool's own `run()` re-validates the same args again.
  Judged a deliberate defense-in-depth choice (a tool's `run()` is a public
  method that can be called directly — e.g. from `resume()`, or from tests —
  not only from `tools_node`, so it should self-validate rather than trust
  an already-validated dict blindly). Do not "fix" this.
- **`get_thread` re-fetched redundantly** when an agent chains e.g.
  `summarize_thread` then `extract_tasks` on the same thread in one run (2-3
  full Gmail round-trips for the same data). A request-scoped cache would
  help but adds complexity (cache invalidation, scoping — per-run? per-
  conversation?) — skipped, documented as a future optimization.
  `search_messages`' internal concurrent gather is already fine as-is.
- **Concurrent tool-call execution** (`asyncio.gather` over multiple tool
  calls in one LLM turn inside `tools_node`) was suggested for efficiency,
  but interacts with the sequential `tool_call_count` limit-enforcement
  logic (which currently increments and checks one call at a time,
  deterministically). Parallelizing would need up-front counting before
  dispatch and was judged too risky to rush — skipped, documented as a
  future optimization.
- **`gmail/auth.py`'s `creds.refresh()` not wrapped in `asyncio.to_thread`**:
  confirmed NOT currently a bug (only ever called from within
  `_get_service()`, which itself only runs inside already-`to_thread`-
  wrapped methods today) — no fix needed unless a new async call site
  bypasses that path later.
- **`api/deps.py` lazy service construction costs latency on the first
  real request** (Chroma DB opens on disk, Gmail OAuth token loads,
  synchronously on the request path) rather than at FastAPI startup. This
  is the deliberate Phase 2 design (so `/health` works with zero
  credentials configured) — not changing it, just documented as a known
  trade-off.
- **Duplicated thread-rendering logic** between
  `intelligence/gemini_service.py` (`_render_message`/`_render_thread`) and
  `agent/reply_drafting.py` (`_render_thread`) — they already differ
  slightly (one includes a `Received:` line). Real duplication, low value
  to fix, skipped.
- **`AuditRecord(...)` construction repeated ~6 times** across
  `graph.py`/`langgraph_agent.py` with the same field shape. A small
  `_record(...)` helper would help; judged lower priority than the safety
  fixes and correctness bugs — pick up if time remains, otherwise leave for
  a future cleanup pass.
- **Default-argument closure trick** in `graph.py`'s `tools_node`
  (`async def _call(_tool: MCPTool = tool, _args: dict = ...)`) — one
  reviewer flagged it as an unnecessary idiom, another confirmed it's
  correct as-is (the closure is invoked immediately within the same loop
  iteration, no late-binding bug exists). Leave as-is; not worth the churn
  given conflicting reviewer opinions and zero actual bug.

---

## How to resume

1. `cd D:\MailPilot`
2. `source .venv/Scripts/activate` (the venv already has all deps installed)
3. `python -m pytest -q` -- confirm still green before making any change.
4. Work through section **A**, then **B**, then **C**, then **D** (10-18) in
   order, running the full suite after each numbered item or small group of
   related items — this project's convention throughout has been "verify
   before moving on," not "implement everything then test once."
5. Read the "Code review findings" section above before re-investigating
   anything — most of the obvious-looking issues have already been triaged.
6. Delete this file once Phase 5 is complete, verified, and the final
   report has been delivered to the user.
