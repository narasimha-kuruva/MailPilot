"""Idempotency guard for approved, irreversible actions (Phase 5.5).

`LangGraphAgent.resume()` already pops a pending approval before acting on
it, so one approval can't be replayed. That says nothing about the *action*
behind it: the same real-world send (same `draft_id`) can reach the
approval gate twice -- through two conversations, or two runs of one
conversation where the model proposes the send again -- and a human
approving both would send the email twice. A future retry layer above
`resume()` (a client retrying a timed-out `/decision` call, a job queue)
would do the same. This guard keys on the action itself -- tool name plus
arguments -- and lets it run once:

- `try_begin()` reserves the action. It fails if the action already
  completed, or if it is running right now (two approvals racing).
- `complete()` records the outcome; every later attempt is suppressed and
  pointed at it.
- `abandon()` releases the reservation after a failure, so a later,
  explicit approval may try again. A failed send is not proof the email
  wasn't sent -- but Gmail deletes a draft once it is sent, so retrying a
  send that did go out fails with "not found" rather than sending twice.

The caller settles the reservation when the action *actually* ends, not
when it stops waiting: a send that outlives its timeout keeps running in a
worker thread, and releasing it early would let a second approval send the
same draft concurrently (see `LangGraphAgent._execute_approved`). Keys are
built from the arguments after schema validation, so fields the tool
ignores can't make one send look like two.

Completed actions live in a `CompletedActionStore`: in memory by default,
or SQLite (`mailpilot.persistence.sqlite`) so a send that went out is still
known after a restart. In-flight reservations are always in memory: they
describe work this process is doing right now.
"""

from __future__ import annotations

import json
from typing import Any, Protocol


def idempotency_key(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Identify an action by what it does, not by which conversation asked for it."""
    return f"{tool_name}:{json.dumps(tool_args, sort_keys=True, default=str)}"


class CompletedActionStore(Protocol):
    """Synchronous on purpose -- see `IdempotencyGuard.try_begin`. A local
    SQLite lookup takes microseconds, so blocking the event loop for it is fine."""

    def get(self, key: str) -> str | None: ...

    def put(self, key: str, result_summary: str) -> None: ...


class InMemoryCompletedActions:
    def __init__(self) -> None:
        self._results: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._results.get(key)

    def put(self, key: str, result_summary: str) -> None:
        self._results[key] = result_summary


class IdempotencyGuard:
    def __init__(self, completed: CompletedActionStore | None = None) -> None:
        self._completed = completed if completed is not None else InMemoryCompletedActions()
        self._in_flight: set[str] = set()

    def try_begin(self, key: str) -> bool:
        """Reserve `key`; False if that action already completed or is running now.

        Synchronous on purpose: with no `await` between the check and the
        reservation, two coroutines can't both pass it.
        """
        if key in self._in_flight or self._completed.get(key) is not None:
            return False
        self._in_flight.add(key)
        return True

    def complete(self, key: str, result_summary: str) -> None:
        self._completed.put(key, result_summary)
        self._in_flight.discard(key)

    def abandon(self, key: str) -> None:
        """The action failed: release the reservation without marking it done."""
        self._in_flight.discard(key)

    def completed_result(self, key: str) -> str | None:
        return self._completed.get(key)
