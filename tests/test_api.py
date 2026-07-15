from dataclasses import replace

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import create_app


@pytest.mark.asyncio
async def test_process_frame_returns_safe_empty_results():
    settings = replace(
        get_settings(),
        api_key="test-secret",
        team_user_url="http://example/users/1/",
        detection_enabled=False,
        localization_enabled=False,
        matching_enabled=True,
    )
    app = create_app(settings=settings)
    payload = {
        "url": "frame-1",
        "image_url": "not-read-because-there-are-no-references.jpg",
        "video_name": "video",
        "session": "session-1",
        "translation_x": None,
        "translation_y": None,
        "translation_z": None,
        "gps_health_status": 0,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/process_frame",
            json=payload,
            headers={"X-API-Key": "test-secret"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["detected_objects"] == []
    assert body["detected_translations"] == []
    assert body["detected_undefined_objects"] == []


@pytest.mark.asyncio
async def test_protected_endpoint_rejects_missing_key():
    settings = replace(get_settings(), api_key="test-secret", matching_enabled=False)
    app = create_app(settings=settings)
    payload = {
        "url": "frame-1",
        "image_url": "unused",
        "video_name": "video",
        "session": "session-1",
        "gps_health_status": 1,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/process_frame", json=payload)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_health_is_available_without_model_loading():
    settings = replace(
        get_settings(),
        detection_enabled=False,
        localization_enabled=False,
        matching_enabled=True,
    )
    app = create_app(settings=settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["task1_detection"] == "disabled"
    assert response.json()["task2_localization"] == "disabled"
