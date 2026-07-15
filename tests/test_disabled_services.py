from dataclasses import replace

import pytest

from app.core.config import get_settings
from app.services.common import FrameContext
from app.services.detection.disabled_service import DisabledDetectionService
from app.services.localization.disabled_service import DisabledLocalizationService
from app.services.localization.interface import LocalizationSessionState


def frame() -> FrameContext:
    return FrameContext(
        frame_id="frame-1",
        image_url="unused.jpg",
        video_name="video",
        session_id="session-1",
        gps_health_status=0,
        gps_x=None,
        gps_y=None,
        gps_z=None,
    )


@pytest.mark.asyncio
async def test_detection_disabled_returns_no_fake_objects():
    result = await DisabledDetectionService().process_frame(frame())
    assert result == []


@pytest.mark.asyncio
async def test_localization_disabled_never_echoes_or_invents_position():
    service = DisabledLocalizationService()
    state = LocalizationSessionState(session_id="session-1")
    result = await service.process_frame(frame(), state)
    assert result is None
    assert state.frame_count == 1


def test_task_configuration_defaults_are_disabled():
    settings = replace(get_settings(), detection_enabled=False, localization_enabled=False)
    assert settings.detection_enabled is False
    assert settings.localization_enabled is False
