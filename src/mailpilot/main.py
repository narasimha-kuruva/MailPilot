"""Application entry point: FastAPI app factory and uvicorn launcher."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from mailpilot.api.middleware import request_context
from mailpilot.api.routes.agent import router as agent_router
from mailpilot.api.routes.context import router as context_router
from mailpilot.api.routes.health import router as health_router
from mailpilot.api.routes.metrics import router as metrics_router
from mailpilot.config import get_settings
from mailpilot.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info(
        "MailPilot starting up",
        extra={"extra_fields": {"app_env": settings.app_env}},
    )
    yield
    logger.info("MailPilot shutting down")


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

    app.include_router(health_router, prefix="/api/v1")
    app.include_router(agent_router, prefix="/api/v1")
    app.include_router(metrics_router, prefix="/api/v1")
    app.include_router(context_router, prefix="/api/v1")

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
