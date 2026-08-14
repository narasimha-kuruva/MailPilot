"""Health-check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from mailpilot import __version__
from mailpilot.api.deps import SettingsDep
from mailpilot.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(status="ok", app_env=settings.app_env, version=__version__)
