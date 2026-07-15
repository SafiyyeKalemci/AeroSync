from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from app.core.config import get_settings
from app.schemas import DetectedObject, LandingStatus, MotionStatus, ObjectClass
from app.services.detection.homography_adaptive_motion import (
    HomographyAdaptiveMotionAnalyzer,
    SceneQuality,
)
from app.services.detection.homography_motion import (
    HomographyComputation,
    HomographyDiagnostics,
    HomographyMotionField,
)
from app.services.detection.service import YoloDetectionService


def _vehicle() -> DetectedObject:
    return DetectedObject(
        cls=ObjectClass.TASIT, top_left_x=2, top_left_y=2,
        bottom_right_x=4, bottom_right_y=4, confidence=0.9,
        motion_status=MotionStatus.UNKNOWN,
        landing_status=LandingStatus.NOT_APPLICABLE,
    )


class _Homography:
    def __init__(self, flow: np.ndarray, valid_mask: np.ndarray | None = None,
                 *, valid: bool = True) -> None:
        diagnostics = HomographyDiagnostics(
            valid, "ok" if valid else "ransac_failed", 100, 90, 0.9, 1.0,
            valid, "high" if valid else None, 0.5 if valid else None,
            0.4 if valid else None, 0.9 if valid else None)
        field = HomographyMotionField(
            flow, np.ones(flow.shape[:2], bool) if valid_mask is None else valid_mask,
            np.eye(3), 1.0, 1.0, int(flow.shape[0] * flow.shape[1]), diagnostics
        ) if valid else None
        self.computation = HomographyComputation(field, diagnostics)

    def analyze_pair(self, *_args):
        return self.computation

    @staticmethod
    def classify_vehicle(_field, _bbox):
        return MotionStatus.STATIONARY

    @staticmethod
    def to_grayscale(image):
        return image

    @staticmethod
    def is_frozen(_previous, _current):
        return False


class _BBox:
    def __init__(self, result: MotionStatus = MotionStatus.MOVING) -> None:
        self.result = result

    def analyze(self, *_args, **_kwargs):
        return SimpleNamespace(measurements=(SimpleNamespace(current_index=0, status=self.result),))


def _analyzer(flow: np.ndarray, valid_mask: np.ndarray | None = None,
              *, valid: bool = True) -> HomographyAdaptiveMotionAnalyzer:
    homography = _Homography(flow, valid_mask, valid=valid)
    return HomographyAdaptiveMotionAnalyzer(
        homography, _BBox(), background_median_max=2.0,
        background_p90_max=5.0, grid_spread_max=3.0,
        min_valid_background_ratio=0.2)


def _run(analyzer: HomographyAdaptiveMotionAnalyzer):
    image = np.zeros((8, 8), np.uint8)
    return analyzer.analyze(image, image, [_vehicle()], [_vehicle()], [])


def test_low_parallax_selects_existing_homography():
    result = _run(_analyzer(np.ones((8, 8, 2), np.float32) * 0.5))
    assert result.scene.scene_quality is SceneQuality.LOW_PARALLAX
    assert result.scene.selected_method == "homography"
    assert result.measurements[0].status is MotionStatus.STATIONARY


def test_high_parallax_selects_existing_bbox_method():
    flow = np.zeros((8, 8, 2), np.float32)
    flow[:, 4:, 0] = 12
    result = _run(_analyzer(flow))
    assert result.scene.scene_quality is SceneQuality.HIGH_PARALLAX
    assert result.scene.selected_method == "homography_bbox"
    assert result.measurements[0].status is MotionStatus.MOVING


def test_invalid_homography_is_safe_unknown():
    result = _run(_analyzer(np.zeros((8, 8, 2), np.float32), valid=False))
    assert result.scene.scene_quality is SceneQuality.UNRELIABLE
    assert result.measurements[0].status is MotionStatus.UNKNOWN


def test_insufficient_background_is_unknown():
    valid = np.zeros((8, 8), bool)
    valid[0, 0] = True
    result = _run(_analyzer(np.zeros((8, 8, 2), np.float32), valid))
    assert result.scene.selection_reason == "insufficient_background"
    assert result.measurements[0].status is MotionStatus.UNKNOWN


def test_scene_decisions_do_not_leak_between_calls():
    analyzer = _analyzer(np.ones((8, 8, 2), np.float32) * 0.5)
    assert _run(analyzer).scene.selected_method == "homography"
    analyzer.homography_analyzer.computation.field.flow[:, 4:, 0] = 12
    assert _run(analyzer).scene.selected_method == "homography_bbox"


def test_adaptive_is_selectable_and_default_remains_unchanged(monkeypatch):
    settings = get_settings()
    service = YoloDetectionService._build_motion_analyzer(
        replace(settings, detection_motion_method="homography_adaptive"))
    assert isinstance(service, HomographyAdaptiveMotionAnalyzer)
    monkeypatch.delenv("DETECTION_MOTION_METHOD", raising=False)
    get_settings.cache_clear()
    try:
        assert get_settings().detection_motion_method == "global_median"
    finally:
        get_settings.cache_clear()


def test_selector_has_no_network_or_prediction_submission_calls():
    source = "\n".join(
        inspect.getsource(__import__(module, fromlist=["*"]))
        for module in (
            "app.services.detection.homography_adaptive_motion",
            "scripts.sweep_task1_adaptive",
        )
    )
    assert "requests." not in source
    assert "httpx." not in source
    assert "send_prediction" not in source
    assert "prediction/" not in source
    assert "competition.runner" not in source
