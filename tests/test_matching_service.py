from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import ImageModality
from app.services.common import FrameContext
from app.services.matching.interface import ReferenceImage
from app.services.matching.service import DinoReferenceMatchingService


def png_bytes() -> bytes:
    ok, encoded = cv2.imencode(".png", np.zeros((8, 12, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def frame(session: str = "session-1", index: int = 5) -> FrameContext:
    return FrameContext(
        frame_id=f"frame-{index}",
        image_url="never-read-by-stage-one.jpg",
        video_name="video",
        session_id=session,
        gps_health_status=1,
        gps_x=1.0,
        gps_y=2.0,
        gps_z=3.0,
        frame_index=index,
    )


@pytest.mark.asyncio
async def test_no_references_returns_empty():
    service = DinoReferenceMatchingService(replace(get_settings(), matching_enabled=True, matching_preload_models=False, matching_warmup_enabled=False))
    assert await service.process_frame(frame()) == []


@pytest.mark.asyncio
async def test_loaded_reference_is_indexed_but_model_result_stays_empty():
    service = DinoReferenceMatchingService(replace(get_settings(), matching_enabled=True, matching_preload_models=False, matching_warmup_enabled=False))
    loaded = await service.set_references(
        "session-1",
        [
            ReferenceImage(
                object_id=7,
                content=png_bytes(),
                active_from_frame=5,
                active_until_frame=10,
                modality=ImageModality.RGB,
            )
        ],
    )
    assert loaded == 1
    assert await service.active_reference_ids("session-1", 5) == (7,)
    assert await service.process_frame(frame()) == []


@pytest.mark.asyncio
async def test_disabled_matching_returns_empty_and_stores_nothing():
    service = DinoReferenceMatchingService(replace(get_settings(), matching_enabled=False))
    loaded = await service.set_references(
        "session-1", [ReferenceImage(object_id=1, content=png_bytes())]
    )
    assert loaded == 0
    assert await service.list_references("session-1") == (None, [])
    assert await service.process_frame(frame()) == []
