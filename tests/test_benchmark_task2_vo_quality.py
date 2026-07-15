from __future__ import annotations

import ast
import asyncio
import csv
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from scripts.benchmark_task2_vo_quality import (
    BenchmarkOptions,
    candidate_assessment,
    detailed_rows,
    discover_and_validate_frames,
    mark_mad_jumps,
    numeric_stats,
    quality_sweep,
    run_benchmark,
    summarize_threshold,
    validate_ratios,
)


def _settings(**changes):
    values = {
        "localization_camera_width": 160,
        "localization_camera_height": 120,
        "localization_camera_fx": 100.0,
        "localization_camera_fy": 100.0,
        "localization_camera_cx": 80.0,
        "localization_camera_cy": 60.0,
        "localization_camera_calibration_path": None,
        "localization_min_features": 8,
        "localization_max_features": 100,
        "localization_min_inliers": 6,
    }
    values.update(changes)
    return replace(get_settings(), **values)


def _frame(index, ratio, inlier_ratio=None, dx=None, dy=None, yaw=None):
    initialization = index == 0
    valid = not initialization and inlier_ratio is not None and inlier_ratio >= ratio
    failure = "initialization_or_reset" if initialization else None if valid else "low_quality"
    cumulative = (0.0, 0.0, 0.0)
    if valid:
        cumulative = (float(dx), float(dy), float(yaw))
    return {
        "frame_index": index,
        "frame_name": f"frame_{index}.jpg",
        "vo_diagnostics": {
            "transform_valid": valid,
            "tracked_points": 50 if not initialization else 0,
            "inlier_count": round(50 * (inlier_ratio or 0)),
            "inlier_ratio": inlier_ratio or 0.0,
            "reprojection_error": 0.2 if not initialization else None,
            "translation_x_px": dx if valid else None,
            "translation_y_px": dy if valid else None,
            "rotation_yaw_rad": yaw if valid else None,
            "failure_reason": failure,
        },
        "pose": {
            "cumulative_dx_px": cumulative[0],
            "cumulative_dy_px": cumulative[1],
            "cumulative_yaw_rad": cumulative[2],
        },
    }


class FakeSequenceProcessor:
    def __init__(self):
        self.sessions = []

    async def __call__(self, settings, _images, *, session_id, video_name, respect_enabled):
        self.sessions.append((session_id, settings.localization_min_inlier_ratio, respect_enabled))
        ratio = settings.localization_min_inlier_ratio
        frames = [
            _frame(0, ratio),
            _frame(1, ratio, 0.42, 1.0, 0.0, 0.01),
            _frame(2, ratio, 0.80, 2.0, 0.0, 0.02),
        ]
        return frames, [None] * 3, []


def test_threshold_sweep_and_independent_state():
    processor = FakeSequenceProcessor()
    detailed, summaries, frames = asyncio.run(
        quality_sweep(_settings(), [object(), object(), object()], [0.50, 0.40], sequence_processor=processor)
    )
    by_ratio = {row["threshold"]: row for row in summaries}
    assert by_ratio[0.50]["valid_count"] == 1
    assert by_ratio[0.50]["low_quality_count"] == 1
    assert by_ratio[0.40]["valid_count"] == 2
    assert all(items[0]["vo_diagnostics"]["failure_reason"] == "initialization_or_reset" for items in frames.values())
    assert processor.sessions == [
        ("task2-vo-quality-050", 0.50, False),
        ("task2-vo-quality-040", 0.40, False),
    ]
    assert len(detailed) == 6


def test_failure_grouping_translation_yaw_and_cumulative_trajectory():
    rows = detailed_rows(
        0.4,
        [_frame(0, 0.4), _frame(1, 0.4, 0.8, 3.0, 4.0, 0.1), _frame(2, 0.4, 0.9, 6.0, 8.0, 0.2)],
    )
    summary = summarize_threshold(0.4, rows)
    assert summary["initialization_reset_count"] == 1
    assert summary["valid_count"] == 2
    assert summary["median_translation_px"] == pytest.approx(7.5)
    assert summary["trajectory_length_px"] == pytest.approx(15.0)
    assert summary["net_displacement_px"] == pytest.approx(10.0)
    assert summary["cumulative_yaw_rad"] == pytest.approx(0.2)


def test_mad_jump_detection():
    rows = detailed_rows(
        0.25,
        [
            _frame(1, 0.25, 0.9, 1.0, 0.0, 0.01),
            _frame(2, 0.25, 0.9, 1.0, 0.0, 0.01),
            _frame(3, 0.25, 0.9, 10.0, 0.0, 0.5),
        ],
    )
    mark_mad_jumps(rows)
    assert [row["translation_jump"] for row in rows] == [False, False, True]
    assert [row["yaw_jump"] for row in rows] == [False, False, True]


def test_numeric_statistics_and_candidate_assessment():
    stats = numeric_stats([1, 2, 3, 10])
    assert stats["median"] == 2.5
    summaries = [
        {"threshold": 0.5, "valid_ratio": 0.4, "valid_count": 4, "translation_jump_count": 0, "yaw_jump_count": 0},
        {"threshold": 0.4, "valid_ratio": 0.7, "valid_count": 7, "translation_jump_count": 1, "yaw_jump_count": 0},
        {"threshold": 0.3, "valid_ratio": 0.8, "valid_count": 8, "translation_jump_count": 3, "yaw_jump_count": 1},
    ]
    result = candidate_assessment(summaries)
    assert result["highest_valid_ratio"][0]["threshold"] == 0.3
    assert result["lowest_jump_valid_ratio_ge_050"]["threshold"] == 0.4
    assert "valid_ratio" in result["heuristic"]


def test_resolution_validation_excludes_mismatch(tmp_path):
    assert cv2.imwrite(str(tmp_path / "ok.jpg"), np.zeros((120, 160, 3), np.uint8))
    assert cv2.imwrite(str(tmp_path / "bad.jpg"), np.zeros((100, 100, 3), np.uint8))
    loaded, rejected = discover_and_validate_frames(tmp_path, _settings())
    assert [item.path.name for item in loaded] == ["ok.jpg"]
    assert rejected[0]["frame_name"] == "bad.jpg"
    assert rejected[0]["reason"] == "camera_resolution_mismatch"


def test_ratio_validation():
    assert validate_ratios([0.5, 0.25]) == (0.5, 0.25)
    with pytest.raises(ValueError):
        validate_ratios([0])
    with pytest.raises(ValueError):
        validate_ratios([0.5, 0.5])


def test_benchmark_creates_csv_and_never_generates_gps_result(tmp_path):
    images = tmp_path / "images"
    output = tmp_path / "output"
    images.mkdir()
    for index in range(3):
        assert cv2.imwrite(str(images / f"frame_{index}.jpg"), np.zeros((120, 160, 3), np.uint8))
    report = asyncio.run(
        run_benchmark(
            _settings(localization_min_inlier_ratio=0.5),
            BenchmarkOptions(images, output, (0.5, 0.4)),
            emit=lambda _: None,
            sequence_processor=FakeSequenceProcessor(),
        )
    )
    assert report["detailed_csv"].is_file()
    assert report["summary_csv"].is_file()
    with report["summary_csv"].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    payload = json.loads(report["report_json"].read_text(encoding="utf-8"))
    assert payload["gps_scale"] == "NOT EVALUATED"
    assert payload["prediction_submission"] == "DISABLED"
    assert payload["production_threshold"] == 0.5
    assert payload["production_threshold_changed"] is False


def test_benchmark_has_no_prediction_server_or_post_calls():
    source_path = Path(__file__).parents[1] / "scripts" / "benchmark_task2_vo_quality.py"
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
