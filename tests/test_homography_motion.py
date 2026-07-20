from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import MotionStatus
from app.services.common import FrameContext
from app.services.detection.homography_motion import HomographyMotionAnalyzer
from app.services.detection.motion_analyzer import MotionAnalyzer
from app.services.detection.service import YoloDetectionService


def _settings(**changes):
    defaults = {
        "detection_enabled": True,
        "detection_motion_enabled": True,
        "detection_motion_method": "homography",
        "detection_motion_min_valid_pixels": 9,
        "detection_motion_inner_crop_ratio": 0.0,
        "detection_motion_max_frame_gap": 1,
        "detection_motion_warmup_frames": 1,
        "detection_motion_flow_downscale": 1.0,
        "detection_motion_freeze_threshold": 0.0,
        "detection_motion_homography_min_features": 8,
        "detection_motion_homography_min_inliers": 6,
        "detection_motion_homography_min_inlier_ratio": 0.5,
        "detection_motion_homography_ransac_threshold": 3.0,
        "detection_motion_homography_max_condition_number": 100000.0,
        "detection_motion_homography_residual_threshold_px": 2.0,
        "detection_landing_enabled": False,
        "matching_enabled": False,
    }
    defaults.update(changes)
    return replace(get_settings(), **defaults)


def _points() -> np.ndarray:
    return np.array(
        [[x, y] for y in (10, 25, 40, 55, 70, 85) for x in (10, 25, 40, 55, 70, 85)],
        dtype=np.float32,
    )


def _tracker_for(matrix: np.ndarray):
    previous = _points()
    current = cv2.perspectiveTransform(previous.reshape(1, -1, 2), matrix)[0]
    return lambda _previous, _current: (previous.copy(), current.copy())


def _exact_estimator(matrix: np.ndarray):
    def estimate(previous, _current, _threshold):
        return matrix.copy(), np.ones((len(previous), 1), np.uint8)

    return estimate


def _flow_with_vehicle_motion(amount: float = 0.0):
    def calculate(_previous, current):
        flow = np.zeros((*current.shape, 2), np.float32)
        if amount:
            flow[30:60, 30:60, 0] = amount
        return flow

    return calculate


def _analyzer(matrix: np.ndarray, *, motion: float = 0.0) -> HomographyMotionAnalyzer:
    return HomographyMotionAnalyzer(
        min_features=8,
        min_inliers=6,
        min_inlier_ratio=0.5,
        ransac_threshold=3.0,
        max_condition_number=100000.0,
        residual_threshold_px=2.0,
        min_valid_pixels=9,
        inner_crop_ratio=0.0,
        flow_downscale=1.0,
        freeze_threshold=0.0,
        feature_tracker=_tracker_for(matrix),
        homography_estimator=_exact_estimator(matrix),
        flow_calculator=_flow_with_vehicle_motion(motion),
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[1, 0, 4], [0, 1, 3], [0, 0, 1]], np.float64),
        cv2.getRotationMatrix2D((50, 50), 7, 1.0).tolist() + [[0, 0, 1]],
        np.array([[1.0, 0.02, 2], [-0.01, 1.0, 1], [0.0002, -0.0001, 1]], np.float64),
    ],
    ids=["translation", "rotation", "perspective"],
)
def test_camera_transform_is_compensated_and_stationary(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    analyzer = _analyzer(matrix)
    computation = analyzer.analyze_pair(
        np.zeros((100, 100), np.uint8),
        np.ones((100, 100), np.uint8),
        [],
    )
    assert computation.diagnostics.valid is True
    measurement = analyzer.measure_vehicle(computation.field, (30, 30, 60, 60))
    assert measurement.status is MotionStatus.STATIONARY
    assert measurement.residual_motion_magnitude == pytest.approx(0.0)


def test_real_vehicle_residual_is_moving():
    # Analizör yarışma için sabitlenmiş eşik kullanır (residual_threshold_px=4.0)
    # ve 30px kutuda dinamik eşik 4.0 + sqrt(30)*0.25 ≈ 5.37'dir; 6.0 > 5.37.
    matrix = np.eye(3, dtype=np.float64)
    analyzer = _analyzer(matrix, motion=6.0)
    computation = analyzer.analyze_pair(
        np.zeros((100, 100), np.uint8),
        np.ones((100, 100), np.uint8),
        [],
    )
    measurement = analyzer.measure_vehicle(computation.field, (30, 30, 60, 60))
    assert measurement.status is MotionStatus.MOVING
    assert measurement.residual_motion_magnitude == pytest.approx(6.0)


def test_insufficient_features_returns_stationary_without_fake_motion():
    # Şartname araçlarda 0/1 ister; ölçüm yapılamayan karede -1 (unknown) puan
    # yakar. Analizör bu yüzden ölçülemeyen aracı stationary olarak raporlar.
    analyzer = HomographyMotionAnalyzer(
        min_features=8,
        min_inliers=6,
        min_inlier_ratio=0.5,
        ransac_threshold=3,
        max_condition_number=100000,
        residual_threshold_px=2,
        min_valid_pixels=9,
        inner_crop_ratio=0,
        flow_downscale=1,
        freeze_threshold=0,
        feature_tracker=lambda _a, _b: (_points()[:3], _points()[:3]),
    )
    computation = analyzer.analyze_pair(
        np.zeros((100, 100), np.uint8), np.ones((100, 100), np.uint8), []
    )
    assert computation.field is None
    assert computation.diagnostics.reason == "insufficient_features"
    assert analyzer.classify_vehicle(None, (30, 30, 60, 60)) is MotionStatus.STATIONARY


def test_bad_homography_returns_unknown():
    matrix = np.zeros((3, 3), np.float64)
    analyzer = _analyzer(matrix)
    computation = analyzer.analyze_pair(
        np.zeros((100, 100), np.uint8), np.ones((100, 100), np.uint8), []
    )
    assert computation.field is None
    assert computation.diagnostics.valid is False
    assert computation.diagnostics.reason in {"singular_normalization", "implausible_determinant"}


def _frame(frame_id: str, index: int, source: str, *, video: str = "v") -> FrameContext:
    return FrameContext(frame_id, source, video, "s", None, None, None, None, index)


class _Runtime:
    confidence = 0.25

    def __init__(self):
        self.calls = 0

    def predict(self, _image):
        self.calls += 1
        box = SimpleNamespace(cls=[0], conf=[0.9], xyxy=[[30, 30, 60, 60]])
        return [SimpleNamespace(boxes=[box])]


def _service(images: dict[str, np.ndarray], settings=None) -> YoloDetectionService:
    async def reader(source, _timeout):
        return source.encode()

    return YoloDetectionService(
        settings or _settings(),
        runtime=_Runtime(),
        image_reader=reader,
        image_decoder=lambda content: images[content.decode()].copy(),
    )


@pytest.mark.asyncio
async def test_first_frame_and_duplicate_are_stationary_and_cached():
    # Sartname araclarda 0/1 ister; olculemeyen ilk kare stationary gonderilir.
    image = np.zeros((100, 100, 3), np.uint8)
    service = _service({"a": image})
    first = await service.process_frame(_frame("f1", 0, "a"))
    duplicate = await service.process_frame(_frame("f1", 0, "a"))
    assert first[0].motion_status is MotionStatus.STATIONARY
    assert duplicate == first
    assert service._runtime.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second",
    [
        _frame("f2", 5, "b"),
        _frame("f2", 1, "b", video="other"),
    ],
    ids=["frame_gap", "video_change"],
)
async def test_discontinuity_returns_stationary(second):
    images = {
        "a": np.zeros((100, 100, 3), np.uint8),
        "b": np.ones((100, 100, 3), np.uint8),
    }
    service = _service(images)
    await service.process_frame(_frame("f1", 0, "a"))
    result = await service.process_frame(second)
    assert result[0].motion_status is MotionStatus.STATIONARY


@pytest.mark.asyncio
async def test_resolution_change_returns_stationary():
    service = _service(
        {
            "a": np.zeros((100, 100, 3), np.uint8),
            "b": np.ones((80, 120, 3), np.uint8),
        }
    )
    await service.process_frame(_frame("f1", 0, "a"))
    result = await service.process_frame(_frame("f2", 1, "b"))
    assert result[0].motion_status is MotionStatus.STATIONARY


def test_default_and_explicit_motion_analyzer_selection():
    default_service = _service({"a": np.zeros((10, 10, 3), np.uint8)}, _settings(detection_motion_method="global_median"))
    homography_service = _service({"a": np.zeros((10, 10, 3), np.uint8)})
    assert isinstance(default_service._motion_analyzer, MotionAnalyzer)
    assert isinstance(homography_service._motion_analyzer, HomographyMotionAnalyzer)


@pytest.mark.parametrize(
    "changes",
    [
        {"detection_motion_method": "invalid"},
        {"detection_motion_homography_min_features": 3},
        {"detection_motion_homography_min_inliers": 3},
        {"detection_motion_homography_min_features": 8, "detection_motion_homography_min_inliers": 9},
        {"detection_motion_homography_min_inlier_ratio": 0},
        {"detection_motion_homography_ransac_threshold": 0},
        {"detection_motion_homography_max_condition_number": float("inf")},
        {"detection_motion_homography_residual_threshold_px": 0},
    ],
)
def test_homography_config_validation(changes):
    with pytest.raises(ValueError):
        _settings(**changes).validate_detection_motion()


def test_compare_tool_has_no_competition_or_prediction_calls():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "scripts" / "compare_task1_motion.py").read_text(encoding="utf-8")
    forbidden = ("send_prediction", "prediction/", "competition.runner", "requests.post", "httpx.post")
    assert all(token not in source for token in forbidden)
