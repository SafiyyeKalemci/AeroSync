from __future__ import annotations

import ast
import asyncio
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import DetectedTranslation
from scripts.validate_task1_detection import load_local_image
from scripts.validate_task2_localization import (
    ValidationOptions,
    camera_intrinsics_report,
    process_loaded_sequence,
    run_synthetic_vo,
    run_validation,
    synthetic_gps_calibration,
    validate_official_compatibility,
)


def _settings(**changes):
    values = {
        "localization_enabled": True,
        "localization_vo_enabled": True,
        "localization_min_features": 8,
        "localization_max_features": 150,
        "localization_feature_quality_level": 0.01,
        "localization_feature_min_distance": 3.0,
        "localization_lk_win_size": 15,
        "localization_lk_max_level": 2,
        "localization_lk_fb_error_threshold": 1.0,
        "localization_ransac_iterations": 100,
        "localization_ransac_residual_threshold": 1.0,
        "localization_min_inliers": 6,
        "localization_min_inlier_ratio": 0.5,
        "localization_max_frame_gap": 1,
        "localization_warmup_frames": 1,
        "localization_freeze_threshold": 0.0,
        "localization_camera_width": 320,
        "localization_camera_height": 240,
        "localization_camera_fx": 250.0,
        "localization_camera_fy": 250.0,
        "localization_camera_cx": 160.0,
        "localization_camera_cy": 120.0,
        "localization_camera_distortion": "0,0,0,0,0",
        "localization_camera_calibration_path": None,
        "localization_calibration_enabled": True,
        "localization_calibration_min_samples": 4,
        "localization_calibration_max_samples": 20,
        "localization_calibration_min_camera_step_px": 0.5,
        "localization_calibration_min_gps_step": 0.001,
        "localization_calibration_max_rms_residual": 0.1,
        "localization_calibration_min_inlier_ratio": 0.7,
        "localization_calibration_min_directional_diversity": 0.1,
        "localization_calibration_scale_min": 0.0001,
        "localization_calibration_scale_max": 2.0,
        "localization_calibration_outlier_mad_factor": 3.5,
        "localization_calibration_expected_max_frame": 450,
    }
    values.update(changes)
    return replace(get_settings(), **values)


def test_camera_intrinsics_validation():
    report = camera_intrinsics_report(_settings())
    assert report["valid"] is True
    assert all(report["checks"].values())


def test_synthetic_vo_initialization_identical_translation_rotation_and_failures():
    report = asyncio.run(run_synthetic_vo(_settings()))
    cases = report["cases"]
    assert cases["identical_frame"]["transform_valid"] is False
    assert cases["identical_frame"]["failure_reason"] == "freeze"
    assert cases["small_translation"]["transform_valid"] is True
    assert abs(cases["small_translation"]["translation_x_px"]) > 1
    assert abs(cases["small_translation"]["translation_y_px"]) > 1
    assert cases["rotation"]["transform_valid"] is True
    assert abs(cases["rotation"]["rotation_yaw_rad"]) > 0.001
    assert cases["insufficient_features"]["failure_reason"] == "insufficient_features"
    assert cases["resolution_change"]["failure_reason"] == "shape_changed"


def test_continuity_frame_gap_video_resolution_and_session_reset():
    continuity = asyncio.run(run_synthetic_vo(_settings()))["continuity"]
    assert continuity["frame_gap"] == {"action": "reset", "reason": "localization_frame_gap"}
    assert continuity["video_change"] == {"action": "reset", "reason": "localization_video_changed"}
    assert continuity["resolution_change"] == {"action": "reset", "reason": "localization_shape_changed"}
    assert continuity["session_reset"] == {"action": "first", "reason": "localization_first_frame"}
    assert continuity["identical_image"]["action"] == "repeated_image"


def test_synthetic_gps_uses_production_calibration_and_recovers_scale():
    result = synthetic_gps_calibration(_settings())
    assert result["scale_ready"] is True
    assert result["sample_count"] >= 4
    assert result["rejected_sample_count"] == 1
    assert result["rejected_reasons"] == ["camera_step_too_small"]
    assert result["estimated_scale"] == pytest.approx(result["target_scale"])
    assert result["scale_confidence"] == 1.0


def test_first_frame_initializes_and_unknown_gps_returns_safe_none(tmp_path):
    image = tmp_path / "frame.jpg"
    canvas = np.zeros((240, 320, 3), np.uint8)
    for x in range(20, 300, 30):
        cv2.circle(canvas, (x, 120), 4, (255, 255, 255), -1)
    assert cv2.imwrite(str(image), canvas)
    frames, outputs, calls = asyncio.run(
        process_loaded_sequence(_settings(), [load_local_image(image)])
    )
    assert frames[0]["initialization_state"] == "initialized"
    assert frames[0]["previous_frame_available"] is False
    assert frames[0]["current_frame_accepted"] is True
    assert outputs == [None]
    assert calls == []


def test_disabled_production_config_is_not_forced_on(tmp_path):
    image = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image), np.zeros((240, 320, 3), np.uint8))
    frames, outputs, calls = asyncio.run(
        process_loaded_sequence(
            _settings(localization_enabled=False), [load_local_image(image)], respect_enabled=True
        )
    )
    assert frames[0]["status"] == "disabled"
    assert frames[0]["failure_reason"] == "localization_disabled_by_configuration"
    assert outputs == [None]
    assert calls == []


def test_json_csv_finite_outputs_and_gps_inactive(tmp_path):
    report = asyncio.run(
        run_validation(
            _settings(localization_enabled=False),
            ValidationOptions(output_dir=tmp_path, synthetic=True),
            emit=lambda _: None,
        )
    )
    json_path = tmp_path / "task2_localization_validation.json"
    csv_path = tmp_path / "task2_localization_sequence.csv"
    assert json_path.is_file() and csv_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["gps_scale_diagnostics"]["active"] is True
    assert payload["gps_scale_diagnostics"]["input_available"] is False
    assert payload["gps_scale_diagnostics"]["calibration_reason"] == "gps_input_unavailable_in_offline_frames"
    assert payload["final_summary"]["prediction_submission"] == "DISABLED"
    assert _all_finite(payload)
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and {"frame_index", "vo_valid", "dx", "cumulative_x", "status"} <= rows[0].keys()
    assert report["official_result_compatibility"]["compatible"] is True


def test_official_result_schema_compatibility():
    translation = DetectedTranslation(translation_x=1.5, translation_y=-2.0, translation_z=3.0)
    report = validate_official_compatibility([None, translation])
    assert report["compatible"] is True
    assert report["serialized_results"][1] == {
        "translation_x": 1.5,
        "translation_y": -2.0,
        "translation_z": 3.0,
    }


def _all_finite(value):
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def test_validator_has_no_prediction_server_or_post_calls():
    source_path = Path(__file__).parents[1] / "scripts" / "validate_task2_localization.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not ({"requests", "httpx", "competition"} & imported_roots)
    for token in (
        "send_prediction",
        "prediction/",
        "competition.runner",
        "requests.post",
        "httpx.post",
        ".post(",
    ):
        assert token not in source.casefold()
