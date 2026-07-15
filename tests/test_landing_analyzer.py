from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import LandingStatus, MotionStatus, ObjectClass
from app.services.common import FrameContext
from app.services.detection.landing_analyzer import LandingAnalyzer, LandingPolicy
from app.services.detection.service import YoloDetectionService


def policy(**changes) -> LandingPolicy:
    values = dict(
        edge_margin_px=2.0,
        edge_margin_ratio=0.0,
        min_intersection_pixels=16.0,
        occupancy_ratio=0.05,
        use_center_check=True,
        use_bottom_center_check=True,
        min_area_pixels=64.0,
    )
    values.update(changes)
    return LandingPolicy(**values)


def analyze(landing=(10, 10, 50, 50), obstacles=None, **changes):
    return LandingAnalyzer(policy(**changes)).analyze(
        raw_bbox=landing,
        clipped_bbox=(max(0, landing[0]), max(0, landing[1]), min(100, landing[2]), min(100, landing[3])),
        frame_width=100,
        frame_height=100,
        obstacles=obstacles or [],
    )


@pytest.mark.parametrize("object_name", ["UAP", "UAI"])
def test_empty_uap_and_uai_areas_are_suitable(object_name):
    assert analyze() is LandingStatus.SUITABLE


@pytest.mark.parametrize("obstacle", [(20, 20, 30, 40), (25, 25, 45, 45)])
def test_person_or_vehicle_in_center_is_unsuitable(obstacle):
    assert analyze(obstacles=[obstacle]) is LandingStatus.UNSUITABLE


def test_multiple_obstacles_and_nearby_vehicle():
    assert analyze(obstacles=[(60, 10, 80, 30), (70, 70, 90, 90)]) is LandingStatus.SUITABLE
    assert analyze(obstacles=[(60, 10, 80, 30), (20, 20, 30, 30)]) is LandingStatus.UNSUITABLE


def test_one_or_two_pixel_contact_stays_below_absolute_threshold():
    assert analyze(obstacles=[(49, 20, 51, 22)]) is LandingStatus.SUITABLE


def test_minimum_intersection_threshold_can_trigger_occupancy():
    assert analyze(obstacles=[(46, 20, 54, 24)], occupancy_ratio=0.5) is LandingStatus.UNSUITABLE


def test_center_and_bottom_center_checks_are_configurable():
    obstacle = (20, 0, 30, 30)
    assert analyze(obstacles=[obstacle], occupancy_ratio=0.9, use_center_check=True, use_bottom_center_check=False) is LandingStatus.UNSUITABLE
    obstacle = (20, 0, 30, 12)
    assert analyze(obstacles=[obstacle], occupancy_ratio=0.9, use_center_check=False, use_bottom_center_check=True) is LandingStatus.UNSUITABLE


def test_out_of_frame_and_edge_tolerance():
    assert analyze(landing=(-5, 10, 40, 50)) is LandingStatus.UNSUITABLE
    assert analyze(landing=(-1, 10, 40, 50)) is LandingStatus.SUITABLE


@pytest.mark.parametrize("landing", [(10, 10, 14, 14), (10, 10, 10, 20), (float("nan"), 0, 10, 10), (0, 0, float("inf"), 10)])
def test_small_or_invalid_geometry_never_invents_suitability(landing):
    clipped = landing if all(isinstance(v, (int, float)) for v in landing) else (0, 0, 1, 1)
    analyzer = LandingAnalyzer(policy())
    assert analyzer.analyze(raw_bbox=landing, clipped_bbox=clipped, frame_width=100, frame_height=100, obstacles=[]) is LandingStatus.NOT_APPLICABLE


class Runtime:
    confidence = 0.25

    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, _image):
        return [SimpleNamespace(boxes=self.boxes)]


def box(cls, coords):
    return SimpleNamespace(cls=[cls], conf=[0.9], xyxy=[coords])


def service(boxes, **changes):
    values = {
        "detection_enabled": True,
        "detection_motion_enabled": False,
        "detection_landing_enabled": True,
        "matching_enabled": False,
    }
    values.update(changes)
    settings = replace(get_settings(), **values)

    async def reader(_source, _timeout):
        return b"image"

    return YoloDetectionService(
        settings,
        runtime=Runtime(boxes),
        image_reader=reader,
        image_decoder=lambda _content: np.zeros((100, 100, 3), np.uint8),
    )


def frame():
    return FrameContext(frame_id="f1", image_url="image", video_name="v", session_id="s", gps_health_status=None, gps_x=None, gps_y=None, gps_z=None, frame_index=1)


@pytest.mark.asyncio
async def test_service_mixed_classes_multiple_areas_and_motion_unchanged():
    detection = service([box(0, [20, 20, 30, 30]), box(1, [70, 70, 80, 80]), box(2, [10, 10, 40, 40]), box(3, [50, 50, 90, 90])])
    result = await detection.process_frame(frame())
    assert [item.landing_status for item in result] == [LandingStatus.NOT_APPLICABLE, LandingStatus.NOT_APPLICABLE, LandingStatus.UNSUITABLE, LandingStatus.UNSUITABLE]
    assert all(item.motion_status is MotionStatus.UNKNOWN for item in result)
    assert [item.cls for item in result] == [ObjectClass.TASIT, ObjectClass.INSAN, ObjectClass.UAP, ObjectClass.UAI]


@pytest.mark.asyncio
async def test_service_can_produce_one_clear_and_one_occupied_area():
    result = await service([box(1, [15, 15, 25, 25]), box(2, [10, 10, 40, 40]), box(3, [60, 60, 95, 95])]).process_frame(frame())
    assert result[1].landing_status is LandingStatus.UNSUITABLE
    assert result[2].landing_status is LandingStatus.SUITABLE


@pytest.mark.asyncio
async def test_landing_disabled_keeps_not_applicable():
    result = await service([box(2, [10, 10, 40, 40])], detection_landing_enabled=False).process_frame(frame())
    assert result[0].landing_status is LandingStatus.NOT_APPLICABLE


@pytest.mark.asyncio
async def test_landing_failure_is_isolated(caplog):
    class BrokenAnalyzer:
        def analyze(self, **_kwargs):
            raise RuntimeError("geometry failed")

    detection = service([box(0, [60, 60, 80, 80]), box(2, [10, 10, 40, 40])])
    detection._landing_analyzer = BrokenAnalyzer()
    result = await detection.process_frame(frame())
    assert len(result) == 2
    assert result[1].landing_status is LandingStatus.NOT_APPLICABLE
    assert any(record.message == "landing_analysis_failed" for record in caplog.records)


@pytest.mark.parametrize("change", [
    {"detection_landing_edge_margin_px": -1},
    {"detection_landing_edge_margin_ratio": 0.5},
    {"detection_landing_min_intersection_pixels": 0},
    {"detection_landing_occupancy_ratio": 1.1},
    {"detection_landing_min_area_pixels": 0},
])
def test_landing_config_validation(change):
    with pytest.raises(ValueError):
        replace(get_settings(), **change).validate_detection_landing()
