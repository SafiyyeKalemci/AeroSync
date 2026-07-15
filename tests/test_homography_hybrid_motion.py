from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.core.config import get_settings
from app.schemas import MotionStatus
from app.services.detection.homography_bbox_motion import BBoxMotionMeasurement
from app.services.detection.homography_hybrid_motion import HomographyHybridMotionAnalyzer
from app.services.detection.homography_motion import (
    HomographyDiagnostics,
    VehicleMotionMeasurement,
)
from app.services.detection.motion_analyzer import MotionAnalyzer
from app.services.detection.service import YoloDetectionService


def _analyzer() -> HomographyHybridMotionAnalyzer:
    bbox_analyzer = SimpleNamespace(homography_analyzer=SimpleNamespace())
    return HomographyHybridMotionAnalyzer(
        bbox_analyzer,
        strong_moving_residual_px=8.0,
        min_association_score=0.35,
        min_iou=0.20,
    )


def _bbox(
    status: MotionStatus,
    *,
    reason: str | None = None,
) -> BBoxMotionMeasurement:
    return BBoxMotionMeasurement(
        current_index=0,
        status=status,
        previous_index=0,
        iou=0.8,
        center_residual_px=1.0 if status is MotionStatus.STATIONARY else 10.0,
        size_ratio=1.0,
        association_score=0.9,
        reason=reason or status.value,
    )


def _flow(status: MotionStatus, residual: float | None = None) -> VehicleMotionMeasurement:
    return VehicleMotionMeasurement(status, residual, 100)


def _diagnostics(*, valid: bool = True, quality: str = "high") -> HomographyDiagnostics:
    return HomographyDiagnostics(
        valid=valid,
        reason="ok" if valid else "ransac_failed",
        match_count=100,
        inlier_count=90 if valid else 0,
        inlier_ratio=0.9 if valid else 0.0,
        quality_accepted=valid,
        quality_level=quality,
    )


@pytest.mark.parametrize(
    ("bbox_status", "flow_status", "expected"),
    [
        (MotionStatus.STATIONARY, MotionStatus.STATIONARY, MotionStatus.STATIONARY),
        (MotionStatus.MOVING, MotionStatus.MOVING, MotionStatus.MOVING),
        (MotionStatus.STATIONARY, MotionStatus.MOVING, MotionStatus.UNKNOWN),
        (MotionStatus.MOVING, MotionStatus.STATIONARY, MotionStatus.UNKNOWN),
    ],
)
def test_hybrid_decision_table(bbox_status, flow_status, expected):
    result, _ = _analyzer()._decide(
        _bbox(bbox_status),
        _flow(flow_status, 12.0 if flow_status is MotionStatus.MOVING else 1.0),
        _diagnostics(),
    )
    assert result is expected


def test_bbox_unknown_and_weak_flow_moving_is_unknown():
    result, _ = _analyzer()._decide(
        _bbox(MotionStatus.UNKNOWN, reason="hysteresis"),
        _flow(MotionStatus.MOVING, 7.9),
        _diagnostics(),
    )
    assert result is MotionStatus.UNKNOWN


def test_bbox_unknown_and_strong_flow_with_high_quality_is_moving():
    result, reason = _analyzer()._decide(
        _bbox(MotionStatus.UNKNOWN, reason="hysteresis"),
        _flow(MotionStatus.MOVING, 12.0),
        _diagnostics(quality="high"),
    )
    assert result is MotionStatus.MOVING
    assert reason == "strong_flow_with_high_quality_homography"


def test_unmatched_bbox_and_strong_flow_with_high_quality_is_moving():
    bbox = BBoxMotionMeasurement(
        current_index=0,
        status=MotionStatus.UNKNOWN,
        reason="unmatched",
    )
    result, reason = _analyzer()._decide(
        bbox,
        _flow(MotionStatus.MOVING, 12.0),
        _diagnostics(quality="high"),
    )
    assert result is MotionStatus.MOVING
    assert reason == "strong_flow_with_high_quality_homography"


def test_invalid_homography_is_unknown():
    result, reason = _analyzer()._decide(
        _bbox(MotionStatus.MOVING),
        _flow(MotionStatus.MOVING, 20.0),
        _diagnostics(valid=False),
    )
    assert result is MotionStatus.UNKNOWN
    assert reason == "invalid_homography"


def test_frame_edge_is_safely_unknown():
    result, reason = _analyzer()._decide(
        _bbox(MotionStatus.UNKNOWN, reason="edge_unreliable"),
        _flow(MotionStatus.MOVING, 20.0),
        _diagnostics(),
    )
    assert result is MotionStatus.UNKNOWN
    assert reason == "frame_edge"


class _Runtime:
    confidence = 0.25


def test_hybrid_is_selectable_and_default_stays_global_median(monkeypatch):
    settings = replace(
        get_settings(),
        detection_enabled=True,
        detection_motion_enabled=True,
        detection_landing_enabled=False,
        matching_enabled=False,
    )
    default = YoloDetectionService(
        replace(settings, detection_motion_method="global_median"), runtime=_Runtime()
    )
    hybrid = YoloDetectionService(
        replace(settings, detection_motion_method="homography_hybrid"), runtime=_Runtime()
    )
    assert isinstance(default._motion_analyzer, MotionAnalyzer)
    assert isinstance(hybrid._motion_analyzer, HomographyHybridMotionAnalyzer)
    monkeypatch.delenv("DETECTION_MOTION_METHOD", raising=False)
    get_settings.cache_clear()
    try:
        assert get_settings().detection_motion_method == "global_median"
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "changes",
    [
        {"detection_motion_hybrid_strong_moving_residual_px": 0},
        {"detection_motion_hybrid_min_association_score": -0.1},
        {"detection_motion_hybrid_min_iou": 1.1},
    ],
)
def test_hybrid_config_validation(changes):
    with pytest.raises(ValueError):
        replace(get_settings(), **changes).validate_detection_motion()
