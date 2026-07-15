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
from scripts.validate_task2_gps_loss_recovery import (
    ValidationOptions,
    build_parser,
    make_partition,
    run_controlled_scenario,
    run_validation,
    validate_options,
)


def _settings(**changes):
    values = {
        "localization_enabled": True,
        "localization_vo_enabled": True,
        "localization_min_features": 8,
        "localization_max_features": 150,
        "localization_min_inliers": 6,
        "localization_min_inlier_ratio": 0.25,
        "localization_camera_width": 64,
        "localization_camera_height": 64,
        "localization_camera_fx": 64.0,
        "localization_camera_fy": 64.0,
        "localization_camera_cx": 32.0,
        "localization_camera_cy": 32.0,
        "localization_camera_distortion": "",
        "localization_camera_calibration_path": None,
        "localization_calibration_enabled": True,
        "localization_calibration_min_samples": 8,
        "localization_calibration_max_samples": 450,
        "localization_calibration_min_camera_step_px": 1.0,
        "localization_calibration_min_gps_step": 0.02,
        "localization_calibration_max_rms_residual": 0.5,
        "localization_calibration_min_inlier_ratio": 0.7,
        "localization_calibration_min_directional_diversity": 0.1,
        "localization_calibration_scale_min": 0.000001,
        "localization_calibration_scale_max": 10.0,
        "localization_calibration_outlier_mad_factor": 3.5,
        "localization_calibration_expected_max_frame": 450,
        "localization_max_delta_per_frame": 5.0,
        "localization_z_policy": "hold_last_valid_z",
        "localization_recovery_min_healthy_frames": 1,
    }
    values.update(changes)
    return replace(get_settings(), **values)


@pytest.fixture(scope="module")
def nominal_report():
    return asyncio.run(run_controlled_scenario(_settings()))


def test_minimum_samples_become_ready_and_known_transform_is_recovered(nominal_report):
    summary = nominal_report["final_summary"]
    assert summary["calibration_ready_frame"] == 8
    assert summary["calibration_samples"] == 11
    assert summary["calibration_ready_before_loss"] is True
    assert summary["estimated_scale"] == pytest.approx(0.01, abs=1e-12)
    assert summary["estimated_rotation_deg"] == pytest.approx(15.0, abs=1e-9)


def test_healthy_ground_truth_passthrough_and_gps_loss_anchor(nominal_report):
    rows = nominal_report["frames"]
    for row in rows[:12]:
        assert row["status"] == "gps_ground_truth"
        assert row["predicted_translation_x"] == pytest.approx(row["ground_truth_x"])
        assert row["predicted_translation_y"] == pytest.approx(row["ground_truth_y"])
    transition = nominal_report["gps_loss_transition"]
    assert transition["last_valid_gps_anchor"] == pytest.approx(
        [rows[11]["ground_truth_x"], rows[11]["ground_truth_y"], rows[11]["ground_truth_z"]]
    )
    assert transition["frozen_calibration_snapshot"]["ready"] is True


def test_first_unhealthy_motion_prediction_and_z_policy(nominal_report):
    summary = nominal_report["final_summary"]
    first = nominal_report["gps_loss_predictions"][0]
    assert summary["first_unhealthy_frame_movement_preserved"] is True
    assert first["predicted_translation_x"] is not None
    assert first["position_error"] == pytest.approx(0.0, abs=1e-10)
    assert summary["z_policy"] == "hold_last_valid_z"
    assert summary["z_policy_correct"] is True


def test_invalid_vo_during_loss_returns_none_then_valid_state_continues(nominal_report):
    loss = nominal_report["gps_loss_predictions"]
    invalid = next(row for row in loss if not row["vo_valid"])
    assert invalid["failure_reason"] == "low_quality"
    assert invalid["predicted_translation_x"] is None
    assert invalid["stale_result"] is False
    following = loss[loss.index(invalid) + 1]
    assert following["vo_valid"] is True
    assert following["predicted_translation_x"] is not None


def test_recovery_is_immediate_finite_schema_compatible_and_not_stale(nominal_report):
    summary = nominal_report["final_summary"]
    assert summary["recovery_first_frame_returned_ground_truth"] is True
    assert summary["stale_prediction_detected"] is False
    assert summary["finite_outputs"] is True
    assert nominal_report["recovery"]["anchor_renewed"] is True
    assert nominal_report["official_schema_compatibility"]["compatible"] is True
    for row in nominal_report["frames"]:
        for key in ("predicted_translation_x", "predicted_translation_y", "predicted_translation_z"):
            value = row[key]
            assert value is None or math.isfinite(value)


def test_insufficient_calibration_never_fabricates_gps_loss_position():
    report = asyncio.run(
        run_controlled_scenario(
            _settings(), frame_count=9, healthy_count=4, loss_count=3,
            invalid_frame=None, session_id="insufficient-test",
        )
    )
    assert report["final_summary"]["calibration_ready_before_loss"] is False
    assert report["final_summary"]["calibration_samples"] == 3
    assert all(row["predicted_translation_x"] is None for row in report["gps_loss_predictions"])


def test_robust_calibration_rejects_outlier_steps_and_still_recovers_transform():
    report = asyncio.run(
        run_controlled_scenario(
            _settings(), invalid_frame=None, outliers=(5,), session_id="outlier-test"
        )
    )
    calibration = report["calibration_history"][11]
    assert calibration["sample_count"] == 11
    assert calibration["inlier_count"] == 9
    assert calibration["ready"] is True
    assert calibration["scale"] == pytest.approx(0.01, abs=1e-12)
    assert calibration["rotation_deg"] == pytest.approx(15.0, abs=1e-9)


def test_json_csv_outputs_and_required_sections(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    for index in range(24):
        matrix = np.float32([[1, 0, index % 4], [0, 1, (index * 2) % 4]])
        frame = cv2.warpAffine(base, matrix, (64, 64))
        assert cv2.imwrite(str(images / f"frame_{index:03d}.png"), frame)
    output = tmp_path / "output"
    report = asyncio.run(
        run_validation(
            _settings(),
            ValidationOptions(
                images,
                output,
                healthy_frames=20,
                loss_frames=4,
                recovery_frames=0,
            ),
            emit=lambda _message: None,
        )
    )
    json_path = output / "task2_gps_loss_recovery.json"
    csv_path = output / "task2_gps_loss_recovery_sequence.csv"
    assert json_path.is_file() and csv_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    required = {
        "config", "synthetic_truth_config", "frames", "vo_diagnostics",
        "calibration_history", "gps_loss_transition", "gps_loss_predictions",
        "recovery", "error_metrics", "final_summary",
    }
    assert required <= payload.keys()
    assert payload["controlled_fixture_validation"]["final_summary"]["final_result"] == "PASS"
    assert payload["input"]["frame_split"] == {
        "healthy_frames": 20,
        "loss_frames": 4,
        "recovery_frames": 0,
        "source": "cli",
    }
    assert payload["final_summary"]["healthy_gps_frames"] == 20
    assert payload["final_summary"]["gps_loss_frames"] == 4
    assert payload["final_summary"]["recovery_frames"] == 0
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24
    assert set(("frame_index", "gps_healthy", "vo_valid", "gt_x", "pred_x", "status")) <= rows[0].keys()
    assert report["final_summary"]["prediction_submission"] == "DISABLED"


def test_cli_parser_accepts_custom_frame_split():
    args = build_parser().parse_args(
        [
            "--images-dir", "frames",
            "--output-dir", "output",
            "--healthy-frames", "20",
            "--loss-frames", "4",
            "--recovery-frames", "0",
        ]
    )
    assert (args.healthy_frames, args.loss_frames, args.recovery_frames) == (20, 4, 0)
    options = ValidationOptions(
        args.images_dir,
        args.output_dir,
        healthy_frames=args.healthy_frames,
        loss_frames=args.loss_frames,
        recovery_frames=args.recovery_frames,
    )
    validate_options(options)
    assert make_partition(24, 20, 4, 0).recovery_start == 24


def test_custom_split_rejects_partial_excess_and_unassigned_frames():
    with pytest.raises(ValueError, match="must be provided together"):
        validate_options(ValidationOptions(Path("frames"), Path("output"), healthy_frames=20))
    with pytest.raises(ValueError, match="exceeds available frame count"):
        make_partition(24, 20, 5, 0)
    with pytest.raises(ValueError, match="must equal available frame count"):
        make_partition(24, 19, 4, 0)


def test_default_split_remains_backward_compatible():
    assert make_partition(24) == type(make_partition(24))(12, 6, 6)


def test_validator_has_no_prediction_server_or_post_calls():
    source_path = Path(__file__).parents[1] / "scripts" / "validate_task2_gps_loss_recovery.py"
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
        "send_prediction", "prediction/", "competition.runner", "requests.post",
        "httpx.post", ".post(",
    ):
        assert token not in source.casefold()
