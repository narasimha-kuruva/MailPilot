"""Serve MailPilot's tools to an MCP client over stdio: `python -m mailpilot.mcp`.

Configuration is the same `.env` the API uses, read from `--project-dir`
(default: the current directory), so relative paths in it -- the OAuth
token, the knowledge store, the audit database -- resolve as they do for
the API. A Claude Desktop entry looks like:

    "mailpilot": {
      "command": "D:\\\\MailPilot\\\\.venv\\\\Scripts\\\\python.exe",
      "args": ["-m", "mailpilot.mcp", "--project-dir", "D:\\\\MailPilot"]
    }

Over stdio, stdout belongs to the protocol: logs go to stderr, and Gmail must
already be authorized (the consent flow would print to stdout), so the
server refuses to start without a usable token. The model-backed tools
(classify, summarize, extract, grounded drafting) are offered only when the
configured LLM provider is usable; the Gmail tools work without it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from pathlib import Path

from fastapi import HTTPException
from mcp.server.stdio import stdio_server

from mailpilot.logging_config import configure_logging, get_logger

logger = get_logger("mailpilot.mcp")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m mailpilot.mcp", description="MailPilot's tools over MCP (stdio).")
    parser.add_argument("--project-dir", help="directory holding MailPilot's .env (default: the current directory)")
    return parser.parse_args(argv)


def _optional(build, what: str):
    """Build an optional service, or None (with a log line) if its provider is unusable."""
    try:
        return build()
    except HTTPException as exc:  # deps report provider problems as 503s
        logger.warning(f"{what} unavailable, its tools are not offered: {exc.detail}")
    except Exception as exc:  # noqa: BLE001 - the Gmail tools are still worth serving
        logger.warning(f"{what} unavailable, its tools are not offered: {exc}")
    return None


async def _serve() -> int:
    # Imported after chdir: settings are read from the project's .env.
    from mailpilot.api import deps
    from mailpilot.gmail.auth import load_credentials
    from mailpilot.mcp.server import NOT_EXPOSED, build_server
    from mailpilot.mcp.tools.registry import build_tools

    settings = deps.get_settings()
    configure_logging(settings.log_level, stream=sys.stderr)

    if not Path(settings.google_oauth_token_file).exists():
        print(
            f"MailPilot: no Gmail authorization at {settings.google_oauth_token_file}. Authorize once first: "
            'python -c "from mailpilot.config import get_settings; from mailpilot.gmail.auth import '
            'load_credentials; load_credentials(get_settings())"',
            file=sys.stderr,
        )
        return 2
    # Refresh an expired token now, with anything it prints kept off the protocol stream.
    with contextlib.redirect_stdout(sys.stderr):
        load_credentials(settings)

    chat_model = _optional(deps.get_chat_model, "The language model")
    intelligence = _optional(deps.get_intelligence_service, "Email intelligence") if chat_model else None
    rag_service = _optional(deps.get_rag_service, "The knowledge store")
    tools = build_tools(
        deps.get_gmail_client(),
        intelligence_service=intelligence,
        rag_service=rag_service,
        chat_model=chat_model,
        max_context_chars=settings.rag_max_context_chars,
        top_k=settings.rag_top_k,
        own_email=settings.gmail_user_email,
    )
    server = build_server(
        tools,
        deps.get_audit_service(),
        timeout_seconds=settings.agent_tool_timeout_seconds,
        max_retries=settings.agent_max_tool_retries,
        max_output_chars=settings.agent_max_output_chars,
    )
    offered = sorted(name for name in tools if name not in NOT_EXPOSED)
    logger.info(f"MailPilot MCP server ready with {len(offered)} tools: {', '.join(offered)}")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.project_dir:
        os.chdir(args.project_dir)
    return asyncio.run(_serve())


if __name__ == "__main__":
    sys.exit(main())
