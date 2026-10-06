"""SQLite persistence: everything a restart must not lose (`STATE_BACKEND=sqlite`, the default).

Two files under `STATE_DIR` (default `./data/state`):

- `checkpoints.sqlite` -- LangGraph's own checkpointer (`AsyncSqliteSaver`):
  every conversation's messages and graph state, so a conversation -- and a
  run stopped at the approval gate -- picks up where it left off.
- `mailpilot.sqlite` -- MailPilot's tables, through one `sqlite3` connection
  guarded by a lock and used off the event loop:
  - `audit_records`: the audit trail (`SqliteAuditService`);
  - `approvals`: approval requests and decisions (`SqliteApprovalService`);
  - `pending_calls`: the action each stopped run waits on (`SqlitePendingCallStore`);
  - `completed_actions`: approved actions that already ran, so a draft sent
    before a restart can't be sent again after it (`SqliteCompletedActions`).

Both use WAL mode, which keeps readers from blocking the writer. This is
storage for one machine and one worker process. Several workers would share
the files safely, but the in-flight half of the duplicate-send guard
(`IdempotencyGuard`) lives in each process's memory, so two workers could
race the same send. More than one worker, or more than one machine, needs
that reservation moved into a server database behind the same interfaces.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from mailpilot.agent.pending import PendingCallStore, PendingToolCall
from mailpilot.audit.service import AuditService
from mailpilot.observability.metrics import MetricsRegistry
from mailpilot.safety.approval import ApprovalService
from mailpilot.schemas.agent import ApprovalStatus
from mailpilot.schemas.audit import AuditRecord

T = TypeVar("T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_records_by_conversation ON audit_records (conversation_id, id);

CREATE TABLE IF NOT EXISTS approvals (
    conversation_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    status TEXT NOT NULL,
    description TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (conversation_id, step_id)
);

CREATE TABLE IF NOT EXISTS pending_calls (
    conversation_id TEXT PRIMARY KEY,
    call TEXT NOT NULL,
    requested_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS completed_actions (
    key TEXT PRIMARY KEY,
    result_summary TEXT NOT NULL,
    completed_at REAL NOT NULL
);
"""


class SqliteDatabase:
    """MailPilot's own tables. Every statement runs under one lock, so each call is atomic."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Autocommit; the lock makes each `execute()` callable one atomic unit.
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def execute(self, work: Callable[[sqlite3.Connection], T]) -> T:
        with self._lock:
            return work(self._conn)

    async def run(self, work: Callable[[sqlite3.Connection], T]) -> T:
        """`execute()` off the event loop."""
        return await asyncio.to_thread(self.execute, work)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class SqliteAuditService(AuditService):
    def __init__(self, database: SqliteDatabase, metrics: MetricsRegistry | None = None) -> None:
        super().__init__(metrics)
        self._db = database

    async def _append(self, record: AuditRecord) -> None:
        await self._db.run(
            lambda conn: conn.execute(
                "INSERT INTO audit_records (conversation_id, record) VALUES (?, ?)",
                (record.conversation_id, record.model_dump_json()),
            )
        )

    async def get_history(self, conversation_id: str) -> list[AuditRecord]:
        rows = await self._db.run(
            lambda conn: conn.execute(
                "SELECT record FROM audit_records WHERE conversation_id = ? ORDER BY id", (conversation_id,)
            ).fetchall()
        )
        return [AuditRecord.model_validate_json(row[0]) for row in rows]


class SqliteApprovalService(ApprovalService):
    def __init__(self, database: SqliteDatabase) -> None:
        self._db = database

    async def _set(self, conversation_id: str, step_id: str, status: ApprovalStatus, description: str | None = None) -> ApprovalStatus:
        await self._db.run(
            lambda conn: conn.execute(
                "INSERT INTO approvals (conversation_id, step_id, status, description, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (conversation_id, step_id) DO UPDATE SET status = excluded.status, "
                "description = COALESCE(excluded.description, approvals.description), updated_at = excluded.updated_at",
                (conversation_id, step_id, str(status), description, time.time()),
            )
        )
        return status

    async def request_approval(self, conversation_id: str, step_id: str, description: str) -> ApprovalStatus:
        return await self._set(conversation_id, step_id, ApprovalStatus.PENDING, description)

    async def record_decision(self, conversation_id: str, step_id: str, approved: bool) -> ApprovalStatus:
        return await self._set(conversation_id, step_id, ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED)

    async def get_status(self, conversation_id: str, step_id: str) -> ApprovalStatus:
        row = await self._db.run(
            lambda conn: conn.execute(
                "SELECT status FROM approvals WHERE conversation_id = ? AND step_id = ?", (conversation_id, step_id)
            ).fetchone()
        )
        return ApprovalStatus(row[0]) if row else ApprovalStatus.NOT_REQUIRED

    async def mark_expired(self, conversation_id: str, step_id: str) -> ApprovalStatus:
        return await self._set(conversation_id, step_id, ApprovalStatus.EXPIRED)


class SqlitePendingCallStore(PendingCallStore):
    def __init__(self, database: SqliteDatabase) -> None:
        self._db = database

    async def put(self, conversation_id: str, call: PendingToolCall, requested_at: float) -> None:
        await self._db.run(
            lambda conn: conn.execute(
                "INSERT OR REPLACE INTO pending_calls (conversation_id, call, requested_at) VALUES (?, ?, ?)",
                (conversation_id, call.model_dump_json(), requested_at),
            )
        )

    async def take(self, conversation_id: str) -> tuple[PendingToolCall, float] | None:
        # One statement under the database lock: two decisions racing for the
        # same conversation get the row once between them.
        row = await self._db.run(
            lambda conn: conn.execute(
                "DELETE FROM pending_calls WHERE conversation_id = ? RETURNING call, requested_at", (conversation_id,)
            ).fetchone()
        )
        return (PendingToolCall.model_validate_json(row[0]), row[1]) if row else None


class SqliteCompletedActions:
    """`CompletedActionStore` for `IdempotencyGuard` (synchronous by design, see there)."""

    def __init__(self, database: SqliteDatabase) -> None:
        self._db = database

    def get(self, key: str) -> str | None:
        row = self._db.execute(
            lambda conn: conn.execute("SELECT result_summary FROM completed_actions WHERE key = ?", (key,)).fetchone()
        )
        return row[0] if row else None

    def put(self, key: str, result_summary: str) -> None:
        self._db.execute(
            lambda conn: conn.execute(
                "INSERT OR REPLACE INTO completed_actions (key, result_summary, completed_at) VALUES (?, ?, ?)",
                (key, result_summary, time.time()),
            )
        )


# Types stored inside graph state that LangGraph's serializer must be told
# are safe to rebuild (beyond its own and LangChain's message types).
_CHECKPOINT_TYPES = [("mailpilot.agent.pending", "PendingToolCall")]


def build_checkpointer(path: str | Path) -> AsyncSqliteSaver:
    """LangGraph's SQLite checkpointer. It connects lazily, on first use."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serde = JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES)
    return AsyncSqliteSaver(aiosqlite.connect(str(path)), serde=serde)


@dataclass
class PersistentState:
    """Everything `api/deps.py` needs to make the agent survive a restart."""

    database: SqliteDatabase
    checkpoint_path: Path
    _checkpointer: AsyncSqliteSaver | None = None

    @classmethod
    def open(cls, state_dir: str | Path) -> "PersistentState":
        """Opens MailPilot's tables. Safe from any thread; the checkpointer comes later."""
        state_dir = Path(state_dir)
        return cls(database=SqliteDatabase(state_dir / "mailpilot.sqlite"), checkpoint_path=state_dir / "checkpoints.sqlite")

    def checkpointer(self) -> AsyncSqliteSaver:
        """LangGraph's checkpointer, created on first use.

        Call it from the event loop: `AsyncSqliteSaver` binds to the running
        loop when it is created, so building it in a worker thread (where
        FastAPI runs sync dependencies) fails.
        """
        if self._checkpointer is None:
            self._checkpointer = build_checkpointer(self.checkpoint_path)
        return self._checkpointer

    async def close(self) -> None:
        if self._checkpointer is not None:
            await self._checkpointer.conn.close()
        self.database.close()
