"""Application entry point: FastAPI app factory and uvicorn launcher."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from mailpilot.api.deps import get_persistent_state, reset_agent
from mailpilot.api.middleware import request_context
from mailpilot.api.routes.agent import router as agent_router
from mailpilot.api.routes.context import router as context_router
from mailpilot.api.routes.health import router as health_router
from mailpilot.api.routes.metrics import router as metrics_router
from mailpilot.api.security import require_api_access
from mailpilot.config import get_settings
from mailpilot.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

UI_DIR = Path(__file__).parent / "ui"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info(
        "MailPilot starting up",
        extra={"extra_fields": {"app_env": settings.app_env}},
    )
    yield
    logger.info("MailPilot shutting down")
    # Close the SQLite state if a request opened it (flushes the WAL cleanly).
    reset_agent()
    if get_persistent_state.cache_info().currsize:
        state = get_persistent_state()
        if state is not None:
            await state.close()
        get_persistent_state.cache_clear()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="MailPilot",
        description="Agentic AI email automation assistant for Gmail.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Request ids, one log line per request, and a sanitized 500 for anything
    # unhandled -- see mailpilot.api.middleware.
    app.middleware("http")(request_context)

    # Everything except /health needs the API key, or a loopback client when
    # no key is configured -- see mailpilot.api.security.
    app.include_router(health_router, prefix="/api/v1")
    protected = [Depends(require_api_access)]
    app.include_router(agent_router, prefix="/api/v1", dependencies=protected)
    app.include_router(metrics_router, prefix="/api/v1", dependencies=protected)
    app.include_router(context_router, prefix="/api/v1", dependencies=protected)

    # The web UI: static files that call the API above (which does the access
    # checks). Served with a strict CSP -- see mailpilot.api.middleware.
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "mailpilot.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
