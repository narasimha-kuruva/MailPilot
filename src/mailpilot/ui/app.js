// MailPilot web UI: a thin client over the HTTP API (see README, "Talking to the agent").
//
// Everything shown here -- the model's answers, approval descriptions, audit
// summaries -- can carry text from emails written by strangers. It is
// therefore only ever put into the page as text (textContent / DOM nodes),
// never as HTML.

"use strict";

const API = "/api/v1";
const STORAGE_KEY = "mailpilot.chat";
const API_KEY_STORAGE = "mailpilot.apiKey";

const $ = (id) => document.getElementById(id);

// --- State, kept per browser tab so a reload doesn't lose the conversation -----

let chat = loadChat();

function loadChat() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "null");
    if (saved && Array.isArray(saved.messages)) return saved;
  } catch (_) {
    // unreadable or unavailable storage: start fresh
  }
  return { conversationId: null, messages: [] };
}

function saveChat() {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(chat));
  } catch (_) {
    // storage full or blocked: the page still works, it just won't survive a reload
  }
}

function getApiKey() {
  try {
    return sessionStorage.getItem(API_KEY_STORAGE);
  } catch (_) {
    return null;
  }
}

function setApiKey(key) {
  try {
    sessionStorage.setItem(API_KEY_STORAGE, key);
  } catch (_) {
    // not remembered; the dialog will ask again
  }
}

// --- API -------------------------------------------------------------------------

class ApiError extends Error {
  constructor(status, detail, requestId) {
    super(detail);
    this.status = status;
    this.requestId = requestId;
  }
}

function describeDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  return JSON.stringify(detail);
}

async function api(path, { method = "GET", body } = {}, retried = false) {
  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const key = getApiKey();
  if (key) headers["X-API-Key"] = key;

  const response = await fetch(API + path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.status === 401 && !retried) {
    if (await askForApiKey()) return api(path, { method, body }, true);
  }
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    // non-JSON body
  }
  if (!response.ok) {
    const detail = payload && payload.detail !== undefined ? describeDetail(payload.detail) : response.statusText;
    throw new ApiError(response.status, detail, response.headers.get("X-Request-ID"));
  }
  return payload;
}

function askForApiKey() {
  const dialog = $("key-dialog");
  const input = $("key-input");
  input.value = "";
  return new Promise((resolve) => {
    const finish = (ok) => {
      dialog.removeEventListener("close", onClose);
      $("key-cancel").onclick = null;
      resolve(ok);
    };
    const onClose = () => {
      if (dialog.returnValue === "ok" && input.value.trim()) {
        setApiKey(input.value.trim());
        finish(true);
      } else {
        finish(false);
      }
    };
    $("key-cancel").onclick = () => dialog.close("cancel");
    dialog.addEventListener("close", onClose);
    dialog.returnValue = "";
    dialog.showModal();
  });
}

// --- Rendering --------------------------------------------------------------------

// A small, safe subset of Markdown (paragraphs, bullet/numbered lists,
// headings, **bold**, `code`), built from DOM nodes -- nothing is parsed as HTML.
function renderRichText(text) {
  const root = document.createElement("div");
  root.className = "rich";
  let list = null;
  for (const line of String(text || "").split("\n")) {
    const item = line.match(/^\s*(?:[-*•]|(\d+)[.)])\s+(.*)$/);
    if (item) {
      const ordered = item[1] !== undefined;
      if (!list || list.ordered !== ordered) {
        list = { ordered, el: document.createElement(ordered ? "ol" : "ul") };
        root.append(list.el);
      }
      const li = document.createElement("li");
      appendInline(li, item[2]);
      list.el.append(li);
      continue;
    }
    list = null;
    if (!line.trim()) continue;
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    const block = document.createElement(heading ? "h4" : "p");
    appendInline(block, heading ? heading[1] : line);
    root.append(block);
  }
  return root;
}

function appendInline(element, text) {
  for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)) {
    if (!part) continue;
    if (part.length > 4 && part.startsWith("**") && part.endsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = part.slice(2, -2);
      element.append(strong);
    } else if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
      const code = document.createElement("code");
      code.textContent = part.slice(1, -1);
      element.append(code);
    } else {
      element.append(document.createTextNode(part));
    }
  }
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderMessage(message, index) {
  if (message.kind === "user") return el("div", "message user", message.text);

  if (message.kind === "agent") {
    const node = el("div", `message agent${message.failed ? " failed" : ""}`);
    node.append(renderRichText(message.text));
    if (message.failed) node.append(el("div", "meta", "This run stopped early."));
    return node;
  }

  if (message.kind === "error") {
    const node = el("div", "message error", message.text);
    if (message.requestId) node.append(el("div", "meta", `Request id: ${message.requestId}`));
    return node;
  }

  if (message.kind === "approval") {
    const node = el("div", "message approval");
    node.append(el("h3", null, message.toolName === "send_email" ? "Approve sending this email?" : `Approve ${message.toolName}?`));
    node.append(el("pre", null, message.description));
    if (message.outcome) {
      const labels = {
        approved: "Approved",
        rejected: "Rejected — not sent",
        expired: "Expired — not sent",
        withdrawn: "Withdrawn by your next instruction — not sent",
        gone: "No longer waiting for a decision — see Activity for what happened",
      };
      node.append(el("div", `outcome ${message.outcome}`, labels[message.outcome] || message.outcome));
    } else {
      const actions = el("div", "actions");
      const approve = el("button", null, "Approve");
      const reject = el("button", "danger", "Reject");
      approve.type = reject.type = "button";
      approve.onclick = () => decide(index, true);
      reject.onclick = () => decide(index, false);
      actions.append(approve, reject);
      node.append(actions);
    }
    return node;
  }
  return el("div", "message", String(message.text || ""));
}

function render() {
  const container = $("messages");
  const empty = $("empty");
  container.replaceChildren(empty);
  empty.hidden = chat.messages.length > 0;
  chat.messages.forEach((message, index) => container.append(renderMessage(message, index)));
  container.scrollTop = container.scrollHeight;
}

function addMessage(message) {
  chat.messages.push(message);
  saveChat();
  render();
}

// --- Busy indicator -----------------------------------------------------------------

let busyTimer = null;

function setBusy(busy) {
  $("send").disabled = busy;
  $("instruction").disabled = busy;
  // A reply that arrives after the chat was replaced would land in the new one.
  $("new-conversation").disabled = busy;
  document.querySelectorAll(".approval button, .suggestion").forEach((b) => (b.disabled = busy));
  $("working").hidden = !busy;
  clearInterval(busyTimer);
  if (busy) {
    const started = Date.now();
    $("elapsed").textContent = "0s";
    busyTimer = setInterval(() => {
      $("elapsed").textContent = `${Math.round((Date.now() - started) / 1000)}s`;
    }, 1000);
  }
}

function reportError(error) {
  addMessage({
    kind: "error",
    text: error instanceof ApiError ? `${error.status}: ${error.message}` : `Couldn't reach MailPilot: ${error.message}`,
    requestId: error instanceof ApiError ? error.requestId : null,
  });
}

// --- Actions --------------------------------------------------------------------------

function handleRunState(state) {
  chat.conversationId = state.conversation_id;
  if (state.status === "awaiting_approval" && state.pending_approval) {
    addMessage({
      kind: "approval",
      approvalId: state.pending_approval.approval_id,
      toolName: state.pending_approval.tool_name,
      description: state.pending_approval.description,
      outcome: null,
    });
  } else {
    addMessage({ kind: "agent", text: state.final_response || "(no response)", failed: state.status === "failed" });
  }
}

async function send(instruction) {
  addMessage({ kind: "user", text: instruction });
  setBusy(true);
  try {
    const body = { instruction };
    if (chat.conversationId) body.conversation_id = chat.conversationId;
    const state = await api("/agent/run", { method: "POST", body });
    // The server withdraws an action still waiting for a decision when a new
    // instruction arrives; its card can't be acted on any more.
    for (const message of chat.messages) {
      if (message.kind === "approval" && !message.outcome) message.outcome = "withdrawn";
    }
    handleRunState(state);
  } catch (error) {
    reportError(error);
  } finally {
    setBusy(false);
    refreshActivity();
  }
}

async function decide(index, approved) {
  const message = chat.messages[index];
  if (!message || message.outcome || !chat.conversationId) return;
  setBusy(true);
  try {
    const state = await api(`/agent/${encodeURIComponent(chat.conversationId)}/decision`, {
      method: "POST",
      body: { approval_id: message.approvalId, approved },
    });
    message.outcome = approved ? "approved" : "rejected";
    saveChat();
    handleRunState(state);
  } catch (error) {
    if (error instanceof ApiError && (error.status === 410 || error.status === 404)) {
      message.outcome = error.status === 410 ? "expired" : "gone";
      saveChat();
    }
    reportError(error);
  } finally {
    setBusy(false);
    refreshActivity();
  }
}

async function refreshActivity() {
  const list = $("activity");
  if (!chat.conversationId) {
    list.replaceChildren(el("li", "panel-hint", "Nothing yet."));
    return;
  }
  try {
    const records = await api(`/agent/${encodeURIComponent(chat.conversationId)}/audit`);
    if (!records.length) {
      list.replaceChildren(el("li", "panel-hint", "No tool calls yet."));
      return;
    }
    list.replaceChildren(
      ...records.map((record) => {
        const item = el("li");
        const head = el("div");
        head.append(el("span", "tool", record.tool_name), el("span", `badge ${record.status}`, record.status));
        if (record.approval_status && record.approval_status !== "not_required") {
          head.append(el("span", "badge", record.approval_status));
        }
        if (record.duration_ms != null) head.append(el("span", "badge", `${Math.round(record.duration_ms)} ms`));
        item.append(head);
        if (record.result_summary) {
          const summary = el("div", "summary", record.result_summary);
          summary.title = record.result_summary;
          item.append(summary);
        }
        return item;
      })
    );
  } catch (_) {
    list.replaceChildren(el("li", "panel-hint", "Couldn't load the activity log."));
  }
}

async function refreshKnowledge() {
  try {
    const stats = await api("/context");
    $("knowledge-stats").textContent =
      `${stats.threads} thread(s) and ${stats.documents} document(s) indexed (${stats.chunks} chunks).`;
  } catch (error) {
    $("knowledge-stats").textContent =
      error instanceof ApiError ? `Unavailable: ${error.message}` : "Couldn't reach MailPilot.";
  }
}

async function indexThreads(event) {
  event.preventDefault();
  const query = $("index-query").value.trim();
  const maxThreads = Number($("index-max").value) || 10;
  if (!query) return;
  const result = $("index-result");
  result.textContent = "Indexing…";
  try {
    const response = await api("/context/threads", { method: "POST", body: { query, max_threads: maxThreads } });
    const failed = response.failed.length ? `, ${response.failed.length} failed` : "";
    result.textContent = `Indexed ${response.indexed.length} thread(s)${failed}.`;
  } catch (error) {
    result.textContent = error instanceof ApiError ? `${error.status}: ${error.message}` : error.message;
  }
  refreshKnowledge();
}

async function checkHealth() {
  const status = $("status");
  try {
    const health = await api("/health");
    status.textContent = `Connected · v${health.version} · ${health.app_env}`;
    status.classList.remove("bad");
  } catch (_) {
    status.textContent = "Can't reach the MailPilot server.";
    status.classList.add("bad");
  }
}

// --- Wiring ---------------------------------------------------------------------------

function submitInstruction() {
  const box = $("instruction");
  const text = box.value.trim();
  if (!text || $("send").disabled) return;
  box.value = "";
  send(text);
}

document.addEventListener("DOMContentLoaded", () => {
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    submitInstruction();
  });
  $("instruction").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submitInstruction();
    }
  });
  document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
      $("instruction").value = button.textContent;
      submitInstruction();
    });
  });
  $("new-conversation").addEventListener("click", () => {
    chat = { conversationId: null, messages: [] };
    saveChat();
    render();
    refreshActivity();
    $("instruction").focus();
  });
  $("index-form").addEventListener("submit", indexThreads);
  $("knowledge-panel").addEventListener("toggle", (event) => {
    if (event.target.open) refreshKnowledge();
  });

  render();
  checkHealth();
  refreshActivity();
  $("instruction").focus();
});
