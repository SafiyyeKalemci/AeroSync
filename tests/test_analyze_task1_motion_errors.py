from __future__ import annotations

import ast
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.analyze_task1_motion_errors import (
    AnalysisOptions,
    GROUP_FALSE_MOVING,
    GROUP_TRUE_MOVING,
    GROUP_UNKNOWN,
    build_samples,
    classification_group,
    classification_metrics,
    feature_separation,
    group_statistics,
    local_threshold_sweep,
    manual_auc,
    numeric_statistics,
    run_analysis,
)


def _evaluation_row(gt="stationary", prediction="moving", *, matched="True", ignored="False"):
    row = {
        "previous_frame": "frame_1.jpg",
        "current_frame": "frame_2.jpg",
        "gt_object_index": "0",
        "gt_motion": gt,
        "gt_bbox": "[10,10,30,30]",
        "matched_detection_index": "0",
        "detection_bbox": "[10,10,30,30]",
        "detection_confidence": "0.9",
        "matched": matched,
        "ignored": ignored,
        "homography_valid": "True",
        "homography_quality_level": "high",
        "homography_inlier_ratio": "0.8",
        "bbox_iou": "0.7",
        "bbox_center_residual": "8.0",
        "flow_residual": "10.0",
        "local_corrected_residual": "7.0",
    }
    for method in (
        "global_median",
        "homography",
        "homography_bbox",
        "homography_hybrid",
        "homography_local",
    ):
        row[f"{method}_prediction"] = prediction
    return row


def _diagnostic():
    return {
        "frame_previous": "frame_1.jpg",
        "frame_current": "frame_2.jpg",
        "vehicle_index": "0",
        "homography_matches": "100",
        "homography_inliers": "80",
        "homography_inlier_ratio": "0.8",
        "homography_residual_px": "10",
        "bbox_projected_bbox": "[11,10,31,30]",
        "bbox_iou": "0.7",
        "bbox_center_residual_px": "8",
        "bbox_size_ratio": "1.0",
        "bbox_association_score": "0.9",
        "hybrid_bbox_result": "moving",
        "hybrid_flow_result": "moving",
        "homography_hybrid_result": "moving",
        "hybrid_decision_reason": "agree",
        "hybrid_homography_quality_level": "high",
        "local_vehicle_residual_x": "9",
        "local_vehicle_residual_y": "1",
        "local_vehicle_residual_magnitude": "9.05",
        "local_background_residual_x": "2",
        "local_background_residual_y": "1",
        "local_background_residual_magnitude": "2.24",
        "local_corrected_residual_x": "7",
        "local_corrected_residual_y": "0",
        "local_corrected_residual_magnitude": "7",
        "local_vehicle_valid_pixels": "200",
        "local_background_valid_pixels": "300",
        "local_background_valid_ratio": "0.8",
        "local_decision_reason": "corrected_moving",
    }


def test_false_moving_true_moving_and_unknown_grouping():
    assert classification_group("stationary", "moving") == GROUP_FALSE_MOVING
    assert classification_group("moving", "moving") == GROUP_TRUE_MOVING
    assert classification_group("stationary", "unknown") == GROUP_UNKNOWN


def test_numeric_group_statistics_and_percentiles():
    stats = numeric_statistics([1, 2, 3, 4])
    assert stats["count"] == 4
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(2.5)
    assert stats["p25"] == pytest.approx(1.75)
    samples, _, _ = build_samples([_evaluation_row()], [_diagnostic()])
    grouped = group_statistics(samples)
    residual = next(
        row
        for row in grouped
        if row["method"] == "homography"
        and row["group"] == GROUP_FALSE_MOVING
        and row["feature"] == "flow_residual"
    )
    assert residual["count"] == 1
    assert residual["median"] == 10.0


def test_threshold_sweep_unknown_coverage_macro_f1_and_balanced_accuracy():
    base = [
        {"gt_motion": "stationary", "local_corrected_residual": 1.0},
        {"gt_motion": "moving", "local_corrected_residual": 10.0},
        {"gt_motion": "moving", "local_corrected_residual": 5.0},
        {"gt_motion": "stationary", "local_corrected_residual": ""},
    ]
    sweep = local_threshold_sweep(base)
    row = next(
        item
        for item in sweep
        if item["stationary_threshold"] == 2
        and item["moving_threshold"] == 8
    )
    assert row["unknown_pred"] == 2
    assert row["coverage"] == pytest.approx(0.5)
    assert row["moving_f1"] == pytest.approx(2 / 3)
    assert row["stationary_f1"] == pytest.approx(2 / 3)
    assert row["macro_f1"] == pytest.approx(2 / 3)
    assert row["balanced_accuracy"] == pytest.approx(0.5)


def test_classification_metrics_unknown_is_strict_error():
    result = classification_metrics(
        ["moving", "stationary", "stationary"],
        ["moving", "stationary", "unknown"],
    )
    assert result["strict_accuracy"] == pytest.approx(2 / 3)
    assert result["decided_only_accuracy"] == 1.0
    assert result["coverage"] == pytest.approx(2 / 3)


def test_manual_auc_and_feature_separation():
    assert manual_auc([0, 0, 1, 1], [1, 2, 3, 4]) == 1.0
    rows = [
        {"gt_motion": "stationary", "flow_residual": 1, "bbox_center_residual": 4, "local_corrected_residual": 1},
        {"gt_motion": "stationary", "flow_residual": 2, "bbox_center_residual": 3, "local_corrected_residual": 2},
        {"gt_motion": "moving", "flow_residual": 8, "bbox_center_residual": 2, "local_corrected_residual": 7},
        {"gt_motion": "moving", "flow_residual": 9, "bbox_center_residual": 1, "local_corrected_residual": 8},
    ]
    result = {row["feature"]: row for row in feature_separation(rows)}
    assert result["flow_residual"]["auc"] == 1.0
    assert result["bbox_center_residual"]["auc"] == 0.0
    assert result["bbox_center_residual"]["separation_auc"] == 1.0
    assert result["local_corrected_residual"]["moving_median"] == 7.5


def test_malformed_rows_are_skipped():
    malformed = _evaluation_row()
    malformed["detection_bbox"] = "bad"
    samples, errors, base = build_samples([malformed])
    assert samples == []
    assert base == []
    assert errors and "invalid matched detection" in errors[0]


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_analysis_creates_all_outputs(tmp_path):
    images, labels, output = tmp_path / "images", tmp_path / "labels", tmp_path / "output"
    images.mkdir()
    labels.mkdir()
    assert cv2.imwrite(str(images / "frame_2.jpg"), np.zeros((100, 100, 3), np.uint8))
    (labels / "frame_2.xml").write_text("<annotation/>", encoding="utf-8")
    evaluation = tmp_path / "evaluation.csv"
    diagnostics = tmp_path / "diagnostics.csv"
    quality = tmp_path / "quality.csv"
    _write_csv(evaluation, [_evaluation_row(), _evaluation_row("moving", "moving")])
    _write_csv(diagnostics, [_diagnostic()])
    _write_csv(
        quality,
        [
            {
                "previous_frame": "frame_1.jpg",
                "current_frame": "frame_2.jpg",
                "condition_number": "100",
                "reprojection_error": "1.2",
                "spatial_coverage": "0.3",
                "projected_overlap": "0.9",
            }
        ],
    )
    report = run_analysis(
        AnalysisOptions(images, labels, evaluation, output, diagnostics, quality),
        emit=lambda _: None,
    )
    assert report["malformed_rows"] == []
    for filename in (
        "motion_error_samples.csv",
        "motion_error_group_statistics.csv",
        "motion_feature_separation.csv",
        "homography_local_threshold_sweep.csv",
        "top_false_moving.csv",
        "top_false_stationary.csv",
        "most_ambiguous_samples.csv",
        "motion_error_analysis_summary.txt",
    ):
        assert (output / filename).is_file()
    assert report["samples"]
    assert report["local_sweep"]


def test_error_analysis_has_no_prediction_server_or_post_calls():
    source_path = Path(__file__).parents[1] / "scripts" / "analyze_task1_motion_errors.py"
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
