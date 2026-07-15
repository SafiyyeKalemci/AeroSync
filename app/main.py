from __future__ import annotations

from fastapi import FastAPI

from app.api.endpoints import router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.services.registry import ServiceRegistry, build_services


def create_app(
    settings: Settings | None = None,
    services: ServiceRegistry | None = None,
) -> FastAPI:
    configure_logging()
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title="AeroSync Integrated",
        description="TEKNOFEST görevleri için güvenli ve genişletilebilir entegrasyon servisi",
        version="0.1.0",
    )
    app.state.settings = resolved_settings
    app.state.services = services or build_services(resolved_settings)
    app.include_router(router)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"message": "AeroSync Integrated aktif"}

    return app


app = create_app()
