from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import DetectedObject, LandingStatus, MotionStatus, ObjectClass
from app.services.common import FrameContext
from app.services.detection.homography_bbox_motion import HomographyBBoxMotionAnalyzer
from app.services.detection.homography_motion import (
    HomographyComputation,
    HomographyDiagnostics,
)
from app.services.detection.motion_analyzer import MotionAnalyzer
from app.services.detection.service import YoloDetectionService


def _detection(bbox, cls=ObjectClass.TASIT) -> DetectedObject:
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


def _projected_bbox(bbox, matrix):
    x1, y1, x2, y2 = bbox
    points = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], np.float32)
    projected = cv2.perspectiveTransform(points, matrix)[0]
    return (
        float(projected[:, 0].min()),
        float(projected[:, 1].min()),
        float(projected[:, 0].max()),
        float(projected[:, 1].max()),
    )


class _SharedHomography:
    def __init__(self, matrix=None, valid=True):
        self.matrix = np.eye(3, dtype=np.float64) if matrix is None else matrix
        self.valid = valid

    def analyze_pair(self, _previous, _current, _exclusions):
        diagnostics = HomographyDiagnostics(
            self.valid,
            "ok" if self.valid else "ransac_failed",
            40,
            35 if self.valid else 0,
            0.875 if self.valid else 0.0,
            1.0 if self.valid else None,
        )
        field = SimpleNamespace(homography=self.matrix) if self.valid else None
        return HomographyComputation(field, diagnostics)

    @staticmethod
    def to_grayscale(image):
        return image[:, :, 0].copy() if image.ndim == 3 else image.copy()

    @staticmethod
    def is_frozen(_previous, _current):
        return False


def _analyzer(matrix=None, valid=True, **changes):
    options = {
        "match_min_iou": 0.1,
        "match_max_center_distance_ratio": 1.5,
        "match_min_score": 0.25,
        "stationary_threshold_px": 3.0,
        "moving_threshold_px": 8.0,
        "min_size_ratio": 0.5,
        "max_size_ratio": 2.0,
        "min_visible_ratio": 0.75,
    }
    options.update(changes)
    return HomographyBBoxMotionAnalyzer(_SharedHomography(matrix, valid), **options)


def _run(analyzer, previous, current):
    gray = np.zeros((100, 100), np.uint8)
    return analyzer.analyze(gray, gray, previous, current, [])


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[1, 0, 5], [0, 1, 3], [0, 0, 1]], np.float64),
        np.vstack([cv2.getRotationMatrix2D((50, 50), 8, 1), [0, 0, 1]]),
        np.array([[1, 0.02, 2], [-0.01, 1, 1], [0.0002, -0.0001, 1]], np.float64),
    ],
    ids=["translation", "rotation", "perspective"],
)
def test_stationary_vehicle_under_camera_transform(matrix):
    previous_bbox = (30, 30, 50, 50)
    current_bbox = _projected_bbox(previous_bbox, matrix)
    result = _run(
        _analyzer(matrix),
        [_detection(previous_bbox)],
        [_detection(current_bbox)],
    )
    measurement = result.measurements[0]
    assert measurement.status is MotionStatus.STATIONARY
    assert measurement.center_residual_px <= 3.0
    assert measurement.iou == pytest.approx(1.0)


def test_moving_vehicle_after_camera_compensation():
    matrix = np.array([[1, 0, 5], [0, 1, 0], [0, 0, 1]], np.float64)
    previous = (30, 30, 50, 50)
    predicted = _projected_bbox(previous, matrix)
    current = (predicted[0] + 10, predicted[1], predicted[2] + 10, predicted[3])
    measurement = _run(
        _analyzer(matrix), [_detection(previous)], [_detection(current)]
    ).measurements[0]
    assert measurement.status is MotionStatus.MOVING
    assert measurement.center_residual_px == pytest.approx(10.0)


def test_vehicle_in_hysteresis_zone_is_unknown():
    previous = (30, 30, 50, 50)
    current = (35, 30, 55, 50)
    measurement = _run(
        _analyzer(), [_detection(previous)], [_detection(current)]
    ).measurements[0]
    assert measurement.status is MotionStatus.UNKNOWN
    assert measurement.reason == "hysteresis"


def test_partly_outside_frame_vehicle_is_unknown():
    matrix = np.array([[1, 0, -15], [0, 1, 0], [0, 0, 1]], np.float64)
    measurement = _run(
        _analyzer(matrix),
        [_detection((0, 20, 20, 40))],
        [_detection((0, 20, 5, 40))],
    ).measurements[0]
    assert measurement.status is MotionStatus.UNKNOWN
    assert measurement.reason == "edge_unreliable"


def test_new_vehicle_is_unknown_and_exited_vehicle_has_no_current_result():
    previous = [_detection((10, 10, 30, 30)), _detection((70, 70, 90, 90))]
    current = [_detection((10, 10, 30, 30)), _detection((40, 40, 60, 60))]
    result = _run(_analyzer(), previous, current)
    assert result.measurements[0].status is MotionStatus.STATIONARY
    assert result.measurements[1].status is MotionStatus.UNKNOWN
    assert result.measurements[1].previous_index is None
    assert _run(_analyzer(), previous, []).measurements == ()


def test_two_close_vehicles_use_one_to_one_association():
    previous = [_detection((20, 20, 40, 40)), _detection((42, 20, 62, 40))]
    current = [_detection((42, 20, 62, 40)), _detection((20, 20, 40, 40))]
    result = _run(_analyzer(), previous, current)
    matched = [item.previous_index for item in result.measurements]
    assert matched == [1, 0]
    assert len(set(matched)) == 2


def test_large_bbox_size_change_is_unknown():
    result = _run(
        _analyzer(),
        [_detection((30, 30, 50, 50))],
        [_detection((20, 20, 80, 80))],
    )
    assert result.measurements[0].status is MotionStatus.UNKNOWN
    assert result.measurements[0].reason == "unmatched"


def test_invalid_homography_makes_all_current_vehicles_unknown():
    current = [_detection((20, 20, 40, 40)), _detection((50, 50, 70, 70))]
    result = _run(_analyzer(valid=False), [_detection((20, 20, 40, 40))], current)
    assert result.homography_valid is False
    assert all(item.status is MotionStatus.UNKNOWN for item in result.measurements)
    assert all(item.reason == "invalid_homography" for item in result.measurements)


def _settings(**changes):
    defaults = {
        "detection_enabled": True,
        "detection_motion_enabled": True,
        "detection_motion_method": "homography_bbox",
        "detection_motion_warmup_frames": 1,
        "detection_landing_enabled": False,
        "matching_enabled": False,
    }
    defaults.update(changes)
    return replace(get_settings(), **defaults)


class _Runtime:
    confidence = 0.25

    def predict(self, _image):
        box = SimpleNamespace(cls=[0], conf=[0.9], xyxy=[[30, 30, 50, 50]])
        return [SimpleNamespace(boxes=[box])]


@pytest.mark.asyncio
async def test_service_uses_previous_last_result_for_bbox_motion():
    images = {
        "first": np.zeros((100, 100, 3), np.uint8),
        "second": np.ones((100, 100, 3), np.uint8),
    }

    async def reader(source, _timeout):
        return source.encode()

    service = YoloDetectionService(
        _settings(),
        runtime=_Runtime(),
        image_reader=reader,
        image_decoder=lambda content: images[content.decode()].copy(),
        motion_analyzer=_analyzer(),
    )
    first = FrameContext("f1", "first", "v", "s", None, None, None, None, 0)
    second = FrameContext("f2", "second", "v", "s", None, None, None, None, 1)
    assert (await service.process_frame(first))[0].motion_status is MotionStatus.UNKNOWN
    assert (await service.process_frame(second))[0].motion_status is MotionStatus.STATIONARY


def test_motion_method_selection_keeps_global_default_and_adds_bbox():
    global_service = YoloDetectionService(_settings(detection_motion_method="global_median"), runtime=_Runtime())
    bbox_service = YoloDetectionService(_settings(), runtime=_Runtime())
    assert isinstance(global_service._motion_analyzer, MotionAnalyzer)
    assert isinstance(bbox_service._motion_analyzer, HomographyBBoxMotionAnalyzer)


@pytest.mark.parametrize(
    "changes",
    [
        {"detection_motion_bbox_match_min_iou": -0.1},
        {"detection_motion_bbox_match_max_center_distance_ratio": 0},
        {"detection_motion_bbox_match_min_score": 1.1},
        {"detection_motion_bbox_stationary_threshold_px": 8, "detection_motion_bbox_moving_threshold_px": 8},
        {"detection_motion_bbox_min_size_ratio": 2, "detection_motion_bbox_max_size_ratio": 1},
        {"detection_motion_bbox_min_visible_ratio": float("nan")},
    ],
)
def test_bbox_config_validation(changes):
    with pytest.raises(ValueError):
        _settings(**changes).validate_detection_motion()
