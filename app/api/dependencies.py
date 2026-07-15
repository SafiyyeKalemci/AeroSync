from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    supplied_key: str | None = Security(api_key_header),
) -> str:
    configured_key = request.app.state.settings.api_key
    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AEROSYNC_SECRET_KEY yapılandırılmadı.",
        )
    if supplied_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Geçersiz veya eksik API anahtarı.",
        )
    return supplied_key
