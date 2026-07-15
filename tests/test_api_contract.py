import json
from dataclasses import replace

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import create_app
from app.schemas import DetectedObject, ImageModality, LandingStatus, MotionStatus, ObjectClass
from app.services.localization.disabled_service import DisabledLocalizationService
from app.services.localization.session_store import LocalizationSessionStore
from app.services.matching.interface import ReferenceStateInfo
from app.services.registry import ServiceRegistry


class TwoDetectionService:
    async def process_frame(self, frame):
        return [
            DetectedObject(
                cls=ObjectClass.TASIT,
                landing_status=LandingStatus.NOT_APPLICABLE,
                motion_status=MotionStatus.STATIONARY,
                top_left_x=1,
                top_left_y=2,
                bottom_right_x=10,
                bottom_right_y=20,
                confidence=0.8,
            ),
            DetectedObject(
                cls=ObjectClass.INSAN,
                landing_status=LandingStatus.NOT_APPLICABLE,
                motion_status=MotionStatus.MOVING,
                top_left_x=30,
                top_left_y=40,
                bottom_right_x=50,
                bottom_right_y=60,
                confidence=0.7,
            ),
        ]


class EmptyMatchingService:
    async def set_references(self, session_id, references, frame_modality=None):
        return 0

    async def process_frame(self, frame):
        return []

    async def clear_session(self, session_id):
        return None

    async def list_references(self, session_id):
        return None, []

    async def remove_reference(self, session_id, object_id):
        return False


class StatefulMatchingService(EmptyMatchingService):
    def __init__(self):
        self.frame_modality = None
        self.references = {}

    async def set_references(self, session_id, references, frame_modality=None):
        self.frame_modality = frame_modality
        for item in references:
            self.references[item.object_id] = ReferenceStateInfo(
                object_id=item.object_id,
                active_from_frame=item.active_from_frame,
                active_until_frame=item.active_until_frame,
                modality=item.modality or ImageModality.UNKNOWN,
            )
        return len(references)

    async def list_references(self, session_id):
        return self.frame_modality, list(self.references.values())

    async def remove_reference(self, session_id, object_id):
        return self.references.pop(object_id, None) is not None


def payload(gps_health_status=0):
    return {
        "url": "frame-1",
        "image_url": "frame.jpg",
        "video_name": "video",
        "session": "session-1",
        "translation_x": None,
        "translation_y": None,
        "translation_z": None,
        "gps_health_status": gps_health_status,
    }


@pytest.mark.asyncio
async def test_endpoint_preserves_multiple_detections_and_none_translation():
    settings = replace(get_settings(), api_key="secret", matching_enabled=False)
    services = ServiceRegistry(
        detection=TwoDetectionService(),
        localization=DisabledLocalizationService(),
        localization_sessions=LocalizationSessionStore(),
        matching=EmptyMatchingService(),
    )
    app = create_app(settings=settings, services=services)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/process_frame",
            json=payload(),
            headers={"X-API-Key": "secret"},
        )
    assert response.status_code == 200
    body = response.json()
    assert len(body["detected_objects"]) == 2
    assert body["detected_translations"] == []
    assert body["detected_undefined_objects"] == []


@pytest.mark.asyncio
async def test_invalid_gps_status_returns_validation_error():
    settings = replace(get_settings(), api_key="secret", matching_enabled=False)
    app = create_app(settings=settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/process_frame",
            json=payload(gps_health_status=2),
            headers={"X-API-Key": "secret"},
        )
    assert response.status_code == 422


def test_openapi_schema_is_json_serializable_and_contains_process_frame():
    settings = replace(get_settings(), matching_enabled=False)
    schema = create_app(settings=settings).openapi()
    serialized = json.dumps(schema, allow_nan=False)
    assert serialized
    assert "/process_frame" in schema["paths"]
    operation = schema["paths"]["/process_frame"]["post"]
    assert "200" in operation["responses"]


@pytest.mark.asyncio
async def test_local_reference_lifecycle_api_preserves_metadata(tmp_path):
    image_path = tmp_path / "reference.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xe0reference")
    matching = StatefulMatchingService()
    settings = replace(get_settings(), api_key="secret", matching_enabled=True)
    services = ServiceRegistry(
        detection=TwoDetectionService(),
        localization=DisabledLocalizationService(),
        localization_sessions=LocalizationSessionStore(),
        matching=matching,
    )
    app = create_app(settings=settings, services=services)
    headers = {"X-API-Key": "secret"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        loaded = await client.post(
            "/sessions/session-1/references",
            headers=headers,
            json={
                "frame_modality": "rgb",
                "references": [
                    {
                        "object_id": 42,
                        "image_url": str(image_path),
                        "active_from_frame": 10,
                        "active_until_frame": 20,
                        "modality": "thermal",
                    }
                ],
            },
        )
        listed = await client.get("/sessions/session-1/references", headers=headers)
        removed = await client.delete(
            "/sessions/session-1/references/42", headers=headers
        )
        missing = await client.delete(
            "/sessions/session-1/references/42", headers=headers
        )

    assert loaded.status_code == 200
    assert loaded.json() == {"session": "session-1", "loaded": 1}
    assert listed.status_code == 200
    assert listed.json() == {
        "session": "session-1",
        "frame_modality": "rgb",
        "references": [
            {
                "object_id": 42,
                "active_from_frame": 10,
                "active_until_frame": 20,
                "modality": "thermal",
            }
        ],
    }
    assert removed.status_code == 204
    assert missing.status_code == 404
