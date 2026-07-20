from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import DetectedObject, LandingStatus, MotionStatus, ObjectClass
from app.services.detection.homography_motion import HomographyMotionAnalyzer
from app.services.detection.homography_quality import quality_gate_from_settings
from scripts.debug_task1_homography_motion import DebugOptions, run_debug


def _settings():
    return replace(
        get_settings(),
        detection_motion_min_valid_pixels=9,
        detection_motion_inner_crop_ratio=0.0,
        detection_motion_flow_downscale=1.0,
        detection_motion_homography_min_features=8,
        detection_motion_homography_min_inliers=6,
        detection_motion_homography_min_inlier_ratio=0.5,
        detection_motion_homography_residual_threshold_px=2.0,
        detection_motion_homography_quality_gate="fixed",
    )


def _vehicle() -> DetectedObject:
    return DetectedObject(
        cls=ObjectClass.TASIT,
        top_left_x=30,
        top_left_y=30,
        bottom_right_x=60,
        bottom_right_y=60,
        confidence=0.9,
        motion_status=MotionStatus.UNKNOWN,
        landing_status=LandingStatus.NOT_APPLICABLE,
    )


def _analyzer(settings, *, valid=True):
    points = np.array(
        [[x, y] for y in (10, 20, 70, 80) for x in (10, 20, 70, 80)],
        np.float32,
    )

    def tracker(_previous, _current):
        return points.copy(), points.copy()

    def estimator(previous, _current, _threshold):
        if not valid:
            return None, None
        return np.eye(3), np.ones((len(previous), 1), np.uint8)

    def flow(_previous, current):
        # 30px kutuda sabitlenmiş dinamik eşik ≈ 5.37 (4.0 + sqrt(30)*0.25);
        # 6.0 bunun üstünde kalır ve "moving" beklentisini korur.
        result = np.zeros((*current.shape, 2), np.float32)
        result[30:60, 30:60, 0] = 6.0
        return result

    return HomographyMotionAnalyzer(
        min_features=settings.detection_motion_homography_min_features,
        min_inliers=settings.detection_motion_homography_min_inliers,
        min_inlier_ratio=settings.detection_motion_homography_min_inlier_ratio,
        ransac_threshold=settings.detection_motion_homography_ransac_threshold,
        max_condition_number=settings.detection_motion_homography_max_condition_number,
        residual_threshold_px=settings.detection_motion_homography_residual_threshold_px,
        min_valid_pixels=settings.detection_motion_min_valid_pixels,
        inner_crop_ratio=settings.detection_motion_inner_crop_ratio,
        flow_downscale=settings.detection_motion_flow_downscale,
        freeze_threshold=settings.detection_motion_freeze_threshold,
        feature_tracker=tracker,
        homography_estimator=estimator,
        flow_calculator=flow,
        quality_gate=quality_gate_from_settings(settings),
    )


def _images(tmp_path):
    previous = np.zeros((100, 100, 3), np.uint8)
    current = np.ones((100, 100, 3), np.uint8)
    previous_path = tmp_path / "frame_1.png"
    current_path = tmp_path / "frame_2.png"
    assert cv2.imwrite(str(previous_path), previous)
    assert cv2.imwrite(str(current_path), current)
    return previous_path, current_path


async def _detections(_settings, _previous, _current):
    return [[_vehicle()], [_vehicle()]]


@pytest.mark.asyncio
async def test_debug_matches_production_result_and_writes_json_and_images(tmp_path):
    settings = _settings()
    previous, current = _images(tmp_path)
    output = tmp_path / "output"
    analyzer = _analyzer(settings)
    expected = analyzer.classify_vehicle(
        analyzer.compute_flow(
            np.zeros((100, 100), np.uint8),
            np.ones((100, 100), np.uint8),
            [],
        ),
        (30, 30, 60, 60),
    )
    report = await run_debug(
        settings,
        DebugOptions(previous, current, output),
        detection_provider=_detections,
        analyzer=analyzer,
        emit=lambda _: None,
    )
    assert report["vehicles"][0]["motion_result"] == expected.value == "moving"
    json_path = output / "homography_motion_debug.json"
    assert json_path.is_file()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["vehicles"][0]["motion_result"] == "moving"
    for name in (
        "current_frame_detections.jpg",
        "residual_flow_heatmap.jpg",
        "residual_flow_vectors.jpg",
    ):
        assert (output / name).is_file()
    assert (output / "per_vehicle_debug" / "vehicle_000.jpg").is_file()


@pytest.mark.asyncio
async def test_residual_statistics_are_finite(tmp_path):
    settings = _settings()
    previous, current = _images(tmp_path)
    report = await run_debug(
        settings,
        DebugOptions(previous, current, tmp_path / "stats"),
        detection_provider=_detections,
        analyzer=_analyzer(settings),
        emit=lambda _: None,
    )
    vehicle = report["vehicles"][0]
    for key in (
        "residual_median_x",
        "residual_median_y",
        "residual_magnitude_px",
        "residual_flow_magnitude_p50",
        "residual_flow_magnitude_p75",
        "residual_flow_magnitude_p90",
        "residual_flow_magnitude_p95",
        "residual_flow_magnitude_max",
    ):
        assert math.isfinite(vehicle[key])
    assert vehicle["residual_median_x"] == pytest.approx(6.0)
    assert vehicle["residual_magnitude_px"] == pytest.approx(6.0)
    # Analizör yarışma sabiti inner_crop_ratio=0.10 kullanır: 30px kutu her
    # kenardan 3px kırpılır -> 24x24 = 576 gecerli piksel.
    assert vehicle["valid_residual_pixel_count"] == 576


@pytest.mark.asyncio
async def test_invalid_homography_is_safe_stationary(tmp_path):
    # Homografi kurulamayan karede araç -1 (unknown) yerine stationary gider:
    # şartname araçlarda 0/1 ister, -1 puan yakar.
    settings = _settings()
    previous, current = _images(tmp_path)
    output = tmp_path / "invalid"
    report = await run_debug(
        settings,
        DebugOptions(previous, current, output),
        detection_provider=_detections,
        analyzer=_analyzer(settings, valid=False),
        emit=lambda _: None,
    )
    assert report["homography"]["valid"] is False
    assert report["vehicles"][0]["motion_result"] == "stationary"
    assert report["vehicles"][0]["residual_magnitude_px"] is None
    assert (output / "homography_motion_debug.json").is_file()
    assert (output / "residual_flow_heatmap.jpg").is_file()


def test_debug_tool_has_no_prediction_server_or_post_calls():
    source = (
        Path(__file__).parents[1] / "scripts" / "debug_task1_homography_motion.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "send_prediction",
        "prediction/",
        "competition.runner",
        "requests.post",
        "httpx.post",
        ".post(",
    )
    assert all(token not in source for token in forbidden)
