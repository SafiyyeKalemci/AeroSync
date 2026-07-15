from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest

from app.core.config import get_settings
from app.services.detection.homography_motion import HomographyMotionAnalyzer
from app.services.detection.homography_quality import HomographyQualityGate
from scripts.benchmark_task1_motion import (
    QUALITY_CSV_COLUMNS,
    BenchmarkOptions,
    PairAnalysis,
    calculate_quality_summary,
    run_benchmark,
)


def _points() -> np.ndarray:
    return np.array(
        [[x, y] for y in (10, 25, 40, 55, 70, 85) for x in (10, 25, 40, 55, 70, 85)],
        dtype=np.float32,
    )


def _gate(mode="adaptive", **changes) -> HomographyQualityGate:
    options = {
        "mode": mode,
        "fixed_min_inlier_ratio": 0.50,
        "high_inlier_ratio": 0.50,
        "low_inlier_ratio": 0.35,
        "min_matches": 8,
        "min_inliers": 6,
        "max_condition_number": 100000.0,
        "max_reprojection_error_px": 2.0,
        "min_spatial_coverage": 0.08,
        "min_projected_overlap_ratio": 0.50,
    }
    options.update(changes)
    return HomographyQualityGate(**options)


def _decision(
    ratio: float,
    *,
    gate=None,
    points=None,
    current=None,
    matrix=None,
):
    previous = _points() if points is None else np.asarray(points, np.float32)
    current = previous.copy() if current is None else np.asarray(current, np.float32)
    mask = np.zeros(len(previous), np.uint8)
    mask[: int(len(previous) * ratio)] = 1
    return (gate or _gate()).evaluate(
        np.eye(3) if matrix is None else matrix,
        previous,
        current,
        mask,
        frame_width=100,
        frame_height=100,
    )


def test_high_ratio_valid_geometry_is_accepted_high():
    result = _decision(0.60)
    assert result.accepted is True
    assert result.quality_level == "high"
    assert result.reason == "adaptive_high_accepted"


def test_intermediate_strong_metrics_is_accepted():
    result = _decision(0.45)
    assert result.accepted is True
    assert result.quality_level == "intermediate"
    assert result.reason == "adaptive_intermediate_accepted"
    assert result.reprojection_error == pytest.approx(0.0)
    assert result.spatial_coverage is not None and result.spatial_coverage > 0.08


def test_intermediate_low_spatial_coverage_is_rejected():
    clustered = np.array(
        [[40 + x, 40 + y] for y in (0, 1, 2, 3, 4, 5) for x in (0, 1, 2, 3, 4, 5)],
        np.float32,
    )
    result = _decision(0.45, points=clustered)
    assert result.accepted is False
    assert result.reason == "low_spatial_coverage"


def test_intermediate_high_reprojection_error_is_rejected():
    previous = _points()
    current = previous.copy()
    current[:16, 0] += 5.0
    result = _decision(0.45, current=current)
    assert result.accepted is False
    assert result.reason == "high_reprojection_error"


def test_intermediate_bad_condition_number_is_rejected():
    matrix = np.array([[1e-6, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
    result = _decision(0.45, matrix=matrix)
    assert result.accepted is False
    assert result.reason == "excessive_condition_number"


def test_ratio_below_low_is_rejected_low():
    result = _decision(0.30)
    assert result.accepted is False
    assert result.quality_level == "low"
    assert result.reason == "low_inlier_ratio"


def test_insufficient_absolute_inliers_is_rejected():
    result = _decision(0.40, gate=_gate(min_inliers=20))
    assert result.accepted is False
    assert result.reason == "insufficient_inliers"


def test_invalid_homography_is_rejected():
    result = _decision(0.60, matrix=np.zeros((3, 3)))
    assert result.accepted is False
    assert result.reason == "invalid_homography"


def test_fixed_mode_preserves_threshold_behavior():
    fixed = _gate(mode="fixed", fixed_min_inlier_ratio=0.50)
    assert _decision(0.49, gate=fixed).accepted is False
    assert _decision(0.50, gate=fixed).accepted is True


def test_analyzer_fixed_rejects_and_adaptive_accepts_same_intermediate_estimate():
    previous_points = _points()
    current_points = previous_points.copy()
    mask = np.zeros(len(previous_points), np.uint8)
    mask[:16] = 1

    def tracker(_previous, _current):
        return previous_points.copy(), current_points.copy()

    def estimator(_previous, _current, _threshold):
        return np.eye(3), mask.copy()

    common = dict(
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
        feature_tracker=tracker,
        homography_estimator=estimator,
        flow_calculator=lambda _a, b: np.zeros((*b.shape, 2), np.float32),
    )
    fixed = HomographyMotionAnalyzer(**common, quality_gate=_gate(mode="fixed"))
    adaptive = HomographyMotionAnalyzer(**common, quality_gate=_gate(mode="adaptive"))
    previous = np.zeros((100, 100), np.uint8)
    current = np.ones((100, 100), np.uint8)
    fixed_result = fixed.analyze_pair(previous, current, [])
    adaptive_result = adaptive.analyze_pair(previous, current, [])
    assert fixed_result.field is None
    assert fixed_result.diagnostics.reason == "low_inlier_ratio"
    assert adaptive_result.field is not None
    assert adaptive_result.diagnostics.quality_level == "intermediate"


@pytest.mark.asyncio
async def test_quality_benchmark_csv_and_summary(tmp_path):
    images = tmp_path / "images"
    output = tmp_path / "output"
    images.mkdir()
    (images / "frame_1.jpg").write_bytes(b"x")
    (images / "frame_2.jpg").write_bytes(b"y")
    quality_row = {
        "previous_frame": "frame_1.jpg",
        "current_frame": "frame_2.jpg",
        "matches": 100,
        "inliers": 45,
        "inlier_ratio": 0.45,
        "fixed_050_accepted": False,
        "fixed_045_accepted": True,
        "fixed_040_accepted": True,
        "adaptive_accepted": True,
        "adaptive_quality_level": "intermediate",
        "adaptive_reason": "adaptive_intermediate_accepted",
        "condition_number": 1.0,
        "reprojection_error": 0.5,
        "spatial_coverage": 0.25,
        "projected_overlap": 1.0,
    }

    async def processor(_pair):
        return PairAnalysis((), False, quality_row=quality_row)

    report = await run_benchmark(
        get_settings(), BenchmarkOptions(images, output), processor=processor, emit=lambda _: None
    )
    quality_path = output / "homography_quality_benchmark.csv"
    with quality_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == QUALITY_CSV_COLUMNS
        assert len(list(reader)) == 1
    assert report["quality_summary"]["adaptive_intermediate_accepted"] == 1
    assert report["quality_summary"]["fixed_050_rejected"] == 1
    assert calculate_quality_summary([quality_row])["fixed_040_accepted"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"detection_motion_homography_quality_gate": "invalid"},
        {"detection_motion_homography_adaptive_low_inlier_ratio": 0.5},
        {"detection_motion_homography_adaptive_high_inlier_ratio": 0.3},
        {"detection_motion_homography_adaptive_max_reprojection_error_px": 0},
        {"detection_motion_homography_adaptive_min_spatial_coverage": 1.1},
        {"detection_motion_homography_adaptive_min_projected_overlap_ratio": -0.1},
    ],
)
def test_adaptive_config_validation(changes):
    with pytest.raises(ValueError):
        replace(get_settings(), **changes).validate_detection_motion()
