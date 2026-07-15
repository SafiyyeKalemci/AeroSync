from types import SimpleNamespace

import pytest

from app.schemas import DetectedTranslation, MatchedReferenceObject
from app.services.common import FrameContext
from app.services.frame_processor import FrameProcessor
from app.services.localization.interface import LocalizationSessionState


class FailingDetectionService:
    async def process_frame(self, frame):
        raise RuntimeError("detection failed")


class WorkingLocalizationService:
    async def process_frame(self, frame, state):
        return DetectedTranslation(translation_x=1, translation_y=2, translation_z=3)


class WorkingMatchingService:
    async def process_frame(self, frame):
        return [
            MatchedReferenceObject(
                object_id=7,
                top_left_x=10,
                top_left_y=20,
                bottom_right_x=30,
                bottom_right_y=40,
                confidence=0.9,
            )
        ]


def frame() -> FrameContext:
    return FrameContext(
        frame_id="frame-1",
        image_url="frame.jpg",
        video_name="video",
        session_id="session-1",
        gps_health_status=0,
        gps_x=None,
        gps_y=None,
        gps_z=None,
    )


@pytest.mark.asyncio
async def test_one_task_failure_does_not_discard_other_task_results(caplog):
    services = SimpleNamespace(
        detection=FailingDetectionService(),
        localization=WorkingLocalizationService(),
        matching=WorkingMatchingService(),
    )
    result = await FrameProcessor(services).process(
        frame(),
        LocalizationSessionState(session_id="session-1"),
    )
    assert result.detected_objects == []
    assert result.detected_translation == DetectedTranslation(
        translation_x=1,
        translation_y=2,
        translation_z=3,
    )
    assert [item.object_id for item in result.matched_reference_objects] == [7]
    record = next(record for record in caplog.records if record.message == "frame_task_failed")
    assert record.event == "frame_task_failed"
    assert record.task == "detection"
    assert record.session_id == "session-1"
