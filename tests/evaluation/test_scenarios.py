"""Phase 5.8: every evaluation scenario, run deterministically with its scripted model.

The same scenarios run against a real model with `python -m mailpilot.evaluation`.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from mailpilot.evaluation.runner import run_scenario
from mailpilot.evaluation.scenarios import SCENARIOS, Category, Scenario
from tests.fakes import FakeChatModel


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: f"{s.category}-{s.id}")
async def test_scenario_passes_with_its_scripted_model(scenario: Scenario) -> None:
    script = scenario.script()
    chat_model = FakeChatModel(script)

    result = await run_scenario(scenario, chat_model, sleep=_no_sleep)

    assert result.passed, "\n".join(result.failures)
    # Every scripted response was used: the script describes the run exactly.
    assert len(chat_model.invocations) == len(script)


def test_scenarios_cover_every_category_with_unique_ids() -> None:
    assert {s.category for s in SCENARIOS} == set(Category)
    assert len({s.id for s in SCENARIOS}) == len(SCENARIOS)


def _scenario(scenario_id: str) -> Scenario:
    return next(s for s in SCENARIOS if s.id == scenario_id)


@pytest.mark.asyncio
async def test_expectations_catch_a_model_that_invents_a_recipient() -> None:
    """The safety expectations are not vacuous: a bad run fails them."""
    scenario = _scenario("missing_recipient")
    invented = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "create_draft",
                    "args": {"to": ["team@example.com"], "subject": "Tomorrow's meeting", "body_text": "See you."},
                    "id": "t-0",
                }
            ],
        ),
        AIMessage(content="Drafted it."),
    ]

    result = await run_scenario(scenario, FakeChatModel(invented), sleep=_no_sleep)

    assert not result.passed
    assert any("expected no draft" in failure for failure in result.failures)


@pytest.mark.asyncio
async def test_expectations_catch_an_incomplete_answer() -> None:
    scenario = _scenario("find_unread")
    script = scenario.script()
    script[-1] = AIMessage(content="You have an unread email from Alice about the invoice.")

    result = await run_scenario(scenario, FakeChatModel(script), sleep=_no_sleep)

    assert result.failures == [
        "response mentions none of ['bob']: 'you have an unread email from alice about the invoice.'"
    ]


@pytest.mark.asyncio
async def test_a_scenario_that_raises_is_a_failed_result_not_a_crash() -> None:
    not_a_message = object()  # the graph can't handle this, so the run itself raises

    result = await run_scenario(_scenario("find_unread"), FakeChatModel([not_a_message]), sleep=_no_sleep)

    assert not result.passed
    assert result.failures[0].startswith("run raised")
