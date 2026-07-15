from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import DetectedObject, LandingStatus, MotionStatus, ObjectClass
from app.services.detection.homography_local_motion import HomographyLocalMotionAnalyzer
from app.services.detection.homography_motion import (
    HomographyComputation,
    HomographyDiagnostics,
    HomographyMotionField,
)
from app.services.detection.motion_analyzer import MotionAnalyzer
from app.services.detection.service import YoloDetectionService


def _detection(bbox, cls=ObjectClass.TASIT):
    return DetectedObject(
        cls=cls,
        top_left_x=bbox[0],
        top_left_y=bbox[1],
        bottom_right_x=bbox[2],
        bottom_right_y=bbox[3],
        confidence=0.9,
        motion_status=MotionStatus.UNKNOWN,
        landing_status=LandingStatus.NOT_APPLICABLE,
    )


def _diagnostics(valid=True):
    return HomographyDiagnostics(
        valid=valid,
        reason="ok" if valid else "ransac_failed",
        match_count=100,
        inlier_count=90 if valid else 0,
        inlier_ratio=0.9 if valid else 0.0,
        quality_accepted=valid,
        quality_level="high" if valid else "low",
    )


class _Homography:
    inner_crop_ratio = 0.0
    min_valid_pixels = 10

    def __init__(self, field):
        self.field = field

    def analyze_pair(self, _previous, _current, _exclusions):
        diagnostics = self.field.diagnostics if self.field is not None else _diagnostics(False)
        return HomographyComputation(self.field, diagnostics)

    @staticmethod
    def to_grayscale(image):
        return image.copy()

    @staticmethod
    def is_frozen(_previous, _current):
        return False


def _field(background=(2.0, 1.0), vehicle=(2.0, 1.0), bbox=(40, 40, 60, 60)):
    flow = np.empty((100, 100, 2), np.float32)
    flow[:, :, 0] = background[0]
    flow[:, :, 1] = background[1]
    x1, y1, x2, y2 = bbox
    flow[y1:y2, x1:x2, 0] = vehicle[0]
    flow[y1:y2, x1:x2, 1] = vehicle[1]
    diagnostics = _diagnostics()
    return HomographyMotionField(
        flow=flow,
        valid_mask=np.ones((100, 100), bool),
        homography=np.eye(3),
        scale_x=1.0,
        scale_y=1.0,
        valid_pixel_count=10000,
        diagnostics=diagnostics,
    )


def _analyzer(field, **changes):
    options = {
        "ring_expansion_ratio": 0.5,
        "min_background_pixels": 100,
        "stationary_threshold_px": 2.0,
        "moving_threshold_px": 6.0,
        "min_valid_ratio": 0.5,
    }
    options.update(changes)
    return HomographyLocalMotionAnalyzer(_Homography(field), **options)


def _run(analyzer, detections):
    gray = np.zeros((100, 100), np.uint8)
    return analyzer.analyze(gray, gray, detections, [])


def test_local_parallax_shared_by_vehicle_and_background_is_stationary():
    bbox = (40, 40, 60, 60)
    measurement = _run(
        _analyzer(_field(background=(8, -4), vehicle=(8.5, -3.5), bbox=bbox)),
        [_detection(bbox)],
    ).measurements[0]
    assert measurement.final_result is MotionStatus.STATIONARY
    assert measurement.corrected_residual_magnitude == pytest.approx(2**-0.5)


def test_vehicle_motion_distinct_from_background_is_moving():
    bbox = (40, 40, 60, 60)
    measurement = _run(
        _analyzer(_field(background=(2, 1), vehicle=(10, 1), bbox=bbox)),
        [_detection(bbox)],
    ).measurements[0]
    assert measurement.final_result is MotionStatus.MOVING
    assert measurement.corrected_residual_magnitude == pytest.approx(8.0)


def test_corrected_residual_in_hysteresis_is_unknown():
    bbox = (40, 40, 60, 60)
    measurement = _run(
        _analyzer(_field(background=(1, 0), vehicle=(5, 0), bbox=bbox)),
        [_detection(bbox)],
    ).measurements[0]
    assert measurement.final_result is MotionStatus.UNKNOWN
    assert measurement.decision_reason == "corrected_hysteresis"


def test_insufficient_background_pixels_is_unknown():
    bbox = (40, 40, 60, 60)
    blocker = _detection((0, 0, 100, 100), ObjectClass.UAP)
    measurement = _run(
        _analyzer(_field(bbox=bbox)),
        [_detection(bbox), blocker],
    ).measurements[0]
    assert measurement.final_result is MotionStatus.UNKNOWN
    assert measurement.decision_reason == "insufficient_background_pixels"


def test_seriously_clipped_corner_ring_is_unknown():
    bbox = (0, 0, 20, 20)
    measurement = _run(
        _analyzer(_field(bbox=bbox)),
        [_detection(bbox)],
    ).measurements[0]
    assert measurement.final_result is MotionStatus.UNKNOWN
    assert measurement.decision_reason == "insufficient_background_ratio"


def test_other_detection_is_masked_from_background_ring():
    bbox = (40, 40, 60, 60)
    field = _field(background=(1, 0), vehicle=(1, 0), bbox=bbox)
    field.flow[30:40, 30:70, 0] = 100
    other = _detection((30, 30, 70, 40), ObjectClass.INSAN)
    measurement = _run(_analyzer(field), [_detection(bbox), other]).measurements[0]
    assert measurement.background_statistics is not None
    assert measurement.background_statistics.median_x == pytest.approx(1.0)
    assert measurement.background_statistics.valid_pixel_count == 800


def test_invalid_homography_is_unknown():
    measurement = _run(_analyzer(None), [_detection((40, 40, 60, 60))]).measurements[0]
    assert measurement.final_result is MotionStatus.UNKNOWN
    assert measurement.decision_reason == "invalid_homography"


class _Runtime:
    confidence = 0.25


def test_local_is_selectable_and_global_default_is_unchanged(monkeypatch):
    settings = replace(
        get_settings(),
        detection_enabled=True,
        detection_motion_enabled=True,
        detection_landing_enabled=False,
        matching_enabled=False,
    )
    local = YoloDetectionService(
        replace(settings, detection_motion_method="homography_local"), runtime=_Runtime()
    )
    default = YoloDetectionService(
        replace(settings, detection_motion_method="global_median"), runtime=_Runtime()
    )
    assert isinstance(local._motion_analyzer, HomographyLocalMotionAnalyzer)
    assert isinstance(default._motion_analyzer, MotionAnalyzer)
    monkeypatch.delenv("DETECTION_MOTION_METHOD", raising=False)
    get_settings.cache_clear()
    try:
        assert get_settings().detection_motion_method == "global_median"
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "changes",
    [
        {"detection_motion_local_ring_expansion_ratio": 0},
        {"detection_motion_local_min_background_pixels": 0},
        {"detection_motion_local_stationary_threshold_px": -1},
        {
            "detection_motion_local_stationary_threshold_px": 6,
            "detection_motion_local_moving_threshold_px": 6,
        },
        {"detection_motion_local_min_valid_ratio": 1.1},
    ],
)
def test_local_config_validation(changes):
    with pytest.raises(ValueError):
        replace(get_settings(), **changes).validate_detection_motion()
