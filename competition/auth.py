def backend_headers(api_key: str) -> dict[str, str]:
    if not api_key:
        raise RuntimeError("AEROSYNC_SECRET_KEY yapılandırılmadı.")
    return {"X-API-Key": api_key}
