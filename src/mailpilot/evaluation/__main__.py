"""Evaluate the configured model as the agent: `python -m mailpilot.evaluation`.

Runs every scenario in `mailpilot.evaluation.scenarios` (except the
scripted-only ones) with the real model from `LLM_PROVIDER` against the
in-memory mailbox -- no Gmail account is touched and nothing is sent.
Prints one line per scenario and the pass rate; exits 1 if any failed.

    python -m mailpilot.evaluation                       # everything
    python -m mailpilot.evaluation --category safety     # one category
    python -m mailpilot.evaluation --scenario find_unread --scenario no_trash -v

Every scenario costs several model calls: mind a hosted provider's quota
(the Gemini free tier allows ~20 requests per model per day).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from mailpilot.agent.graph import AgentLimits
from mailpilot.config import get_settings
from mailpilot.evaluation.runner import ScenarioResult, run_scenario
from mailpilot.evaluation.scenarios import SCENARIOS, Category
from mailpilot.llm.providers import LLMProviderError, build_chat_model, build_embedding_function
from mailpilot.logging_config import configure_logging


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m mailpilot.evaluation", description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", action="append", default=[], help="run only this scenario id (repeatable)")
    parser.add_argument("--category", choices=[c.value for c in Category], help="run only this category")
    parser.add_argument("-v", "--verbose", action="store_true", help="show tool calls and the final response")
    return parser.parse_args(argv)


def _print_result(result: ScenarioResult, verbose: bool) -> None:
    mark = "PASS" if result.passed else "FAIL"
    scenario = result.scenario
    print(f"{mark}  {scenario.category:<11} {scenario.id:<30} {result.duration_seconds:6.1f}s", flush=True)
    for failure in result.failures:
        print(f"        - {failure}")
    if verbose:
        print(f"        tools: {', '.join(result.tool_sequence) or '(none)'}")
        print(f"        response: {(result.final_response or '').strip()[:300]!r}")


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    configure_logging("ERROR")
    try:
        chat_model = build_chat_model(settings)
        # The knowledge-store scenarios use the provider's real embedding model.
        embedding_function = build_embedding_function(settings)
    except LLMProviderError as exc:
        print(f"Cannot evaluate: {exc}", file=sys.stderr)
        return 2

    selected = [
        s
        for s in SCENARIOS
        if not s.scripted_only
        and (not args.scenario or s.id in args.scenario)
        and (args.category is None or s.category == args.category)
    ]
    if not selected:
        print("No scenarios selected.", file=sys.stderr)
        return 2

    model_name = settings.ollama_model if settings.llm_provider == "ollama" else settings.gemini_model
    print(f"Evaluating {settings.llm_provider}/{model_name} on {len(selected)} scenario(s)\n")
    limits = AgentLimits.from_settings(settings)
    results = []
    for scenario in selected:
        result = await run_scenario(scenario, chat_model, embedding_function=embedding_function, limits=limits)
        _print_result(result, args.verbose)
        results.append(result)

    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} passed")
    for category in Category:
        in_category = [r for r in results if r.scenario.category == category]
        if in_category:
            print(f"  {category:<11} {sum(r.passed for r in in_category)}/{len(in_category)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
