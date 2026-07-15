"""Termal referans → RGB frame XoFTR çapraz-modal servis yolu testleri."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import ImageModality
from app.services.common import FrameContext
from app.services.matching.interface import ReferenceImage
from app.services.matching.service import DinoReferenceMatchingService
from app.services.matching.xoftr_crossmodal import XoFTRUnavailable


def png_bytes(gray: bool = False) -> bytes:
    if gray:
        image = np.full((64, 64, 3), 120, dtype=np.uint8)
    else:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :, 2] = 200
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def rgb_frame_context(index: int = 5) -> FrameContext:
    return FrameContext(
        frame_id=f"frame-{index}",
        image_url="in-memory.png",
        video_name="video",
        session_id="session-1",
        gps_health_status=1,
        gps_x=0.0,
        gps_y=0.0,
        gps_z=0.0,
        frame_index=index,
        image_modality=ImageModality.RGB,
    )


async def fake_image_reader(url: str, timeout: float) -> bytes:
    return png_bytes()


class FakeXoFTRMatcher:
    def __init__(self, box=None, error: Exception | None = None):
        self.box = box
        self.error = error
        self.calls: list[tuple] = []

    def bbox(self, reference_image, frame_image):
        self.calls.append((reference_image.shape, frame_image.shape))
        if self.error is not None:
            raise self.error
        return self.box


def good_box() -> dict[str, float]:
    return {
        "top_left_x": 4.0,
        "top_left_y": 4.0,
        "bottom_right_x": 40.0,
        "bottom_right_y": 40.0,
        "confidence": 0.9,
        "inliers": 20,
        "angle": 0,
    }


def make_service(matcher, *, xoftr_enabled: bool = True) -> DinoReferenceMatchingService:
    settings = replace(
        get_settings(),
        matching_enabled=True,
        matching_dinov2_enabled=False,
        matching_xoftr_enabled=xoftr_enabled,
    )
    return DinoReferenceMatchingService(
        settings,
        image_reader=fake_image_reader,
        xoftr_factory=lambda _settings: matcher,
    )


async def load_thermal_reference(service: DinoReferenceMatchingService, object_id: int = 3) -> None:
    loaded = await service.set_references(
        "session-1",
        [
            ReferenceImage(
                object_id=object_id,
                content=png_bytes(gray=True),
                active_from_frame=1,
                active_until_frame=10,
                modality=ImageModality.THERMAL,
            )
        ],
    )
    assert loaded == 1


@pytest.mark.asyncio
async def test_thermal_reference_matched_via_xoftr():
    matcher = FakeXoFTRMatcher(box=good_box())
    service = make_service(matcher)
    await load_thermal_reference(service)
    results = await service.process_frame(rgb_frame_context())
    assert len(results) == 1
    matched = results[0]
    assert matched.object_id == 3
    assert matched.top_left_x == pytest.approx(4.0)
    assert matched.bottom_right_y == pytest.approx(40.0)
    assert matched.confidence == pytest.approx(0.9)
    assert len(matcher.calls) == 1


@pytest.mark.asyncio
async def test_thermal_reference_skipped_when_xoftr_disabled():
    matcher = FakeXoFTRMatcher(box=good_box())
    service = make_service(matcher, xoftr_enabled=False)
    await load_thermal_reference(service)
    assert await service.process_frame(rgb_frame_context()) == []
    assert matcher.calls == []


@pytest.mark.asyncio
async def test_xoftr_error_yields_empty_result_without_crash():
    matcher = FakeXoFTRMatcher(error=RuntimeError("model çöktü"))
    service = make_service(matcher)
    await load_thermal_reference(service)
    assert await service.process_frame(rgb_frame_context()) == []


@pytest.mark.asyncio
async def test_xoftr_unavailable_yields_empty_result():
    def failing_factory(_settings):
        raise XoFTRUnavailable("artefakt yok")

    settings = replace(
        get_settings(),
        matching_enabled=True,
        matching_dinov2_enabled=False,
        matching_xoftr_enabled=True,
    )
    service = DinoReferenceMatchingService(
        settings,
        image_reader=fake_image_reader,
        xoftr_factory=failing_factory,
    )
    await load_thermal_reference(service)
    assert await service.process_frame(rgb_frame_context()) == []


@pytest.mark.asyncio
async def test_low_confidence_box_rejected_by_validator():
    low_confidence = {**good_box(), "confidence": 0.05}
    matcher = FakeXoFTRMatcher(box=low_confidence)
    service = make_service(matcher)
    await load_thermal_reference(service)
    assert await service.process_frame(rgb_frame_context()) == []
    assert len(matcher.calls) == 1


@pytest.mark.asyncio
async def test_no_match_returns_empty():
    matcher = FakeXoFTRMatcher(box=None)
    service = make_service(matcher)
    await load_thermal_reference(service)
    assert await service.process_frame(rgb_frame_context()) == []


@pytest.mark.asyncio
async def test_thermal_frame_still_returns_empty():
    matcher = FakeXoFTRMatcher(box=good_box())
    service = make_service(matcher)
    await load_thermal_reference(service)
    context = replace_frame_modality(rgb_frame_context(), ImageModality.THERMAL)
    assert await service.process_frame(context) == []
    assert matcher.calls == []


def replace_frame_modality(context: FrameContext, modality: ImageModality) -> FrameContext:
    from dataclasses import replace as dc_replace

    return dc_replace(context, image_modality=modality)
