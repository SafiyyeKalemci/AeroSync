from __future__ import annotations

import ast
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.optimize_task1_homography_bbox import (
    BBoxConfig,
    BBoxSample,
    OptimizationOptions,
    calculate_metrics,
    cross_validate_profiles,
    grouped_folds,
    leaderboard_rows,
    load_samples,
    pareto_frontier,
    run_optimization,
    simulate_bbox_decision,
    threshold_sweep,
)


def _config(**changes) -> BBoxConfig:
    values = {
        "stationary_threshold": 3.0,
        "moving_threshold": 8.0,
        "min_iou": 0.10,
        "min_association_score": 0.25,
        "min_size_ratio": 0.50,
        "max_size_ratio": 2.0,
    }
    values.update(changes)
    return BBoxConfig(**values)


def _sample(
    gt="stationary",
    residual=2.0,
    *,
    pair="a.jpg->b.jpg",
    iou=0.8,
    score=0.9,
    size=1.0,
    valid=True,
    edge=False,
    projected_edge=False,
) -> BBoxSample:
    previous, current = pair.split("->")
    return BBoxSample(
        pair,
        previous,
        current,
        "0",
        "0",
        gt,
        (10, 10, 30, 30),
        (10, 10, 30, 30),
        0.9,
        valid,
        "high",
        residual,
        iou,
        score,
        size,
        1.0,
        edge,
        projected_edge,
        "unknown",
    )


def test_threshold_decision_and_hysteresis():
    config = _config()
    assert simulate_bbox_decision(_sample(residual=3), config) == "stationary"
    assert simulate_bbox_decision(_sample(residual=8), config) == "moving"
    assert simulate_bbox_decision(_sample(residual=5), config) == "unknown"


@pytest.mark.parametrize(
    ("sample", "config"),
    (
        (_sample(iou=0.09), _config()),
        (_sample(score=0.24), _config()),
        (_sample(size=0.49), _config()),
        (_sample(size=2.01), _config()),
        (_sample(valid=False), _config()),
        (_sample(edge=True), _config()),
        (_sample(projected_edge=True), _config()),
    ),
)
def test_safety_filters_produce_unknown(sample, config):
    assert simulate_bbox_decision(sample, config) == "unknown"


def test_metrics_unknown_is_strict_error_and_coverage_is_decided_fraction():
    samples = [_sample(), _sample("moving", 10), _sample("moving", 5)]
    metrics = calculate_metrics(samples, ["stationary", "moving", "unknown"])
    assert metrics["strict_accuracy"] == pytest.approx(2 / 3)
    assert metrics["decided_only_accuracy"] == 1.0
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(5 / 6)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)


def test_sweep_leaderboards_and_pareto_selection():
    samples = [_sample(), _sample("moving", 10)]
    sweep = threshold_sweep(samples, [_config(), _config(stationary_threshold=1, moving_threshold=12)])
    leaders = leaderboard_rows(sweep)
    assert any(row["leaderboard"] == "highest_macro_f1" for row in leaders)
    frontier = pareto_frontier(sweep)
    assert frontier
    assert max(row["macro_f1"] for row in frontier) == 1.0


def test_grouped_cv_has_no_frame_pair_leakage_and_reports_aggregates():
    samples = [
        _sample(pair="a.jpg->b.jpg"),
        _sample("moving", 10, pair="b.jpg->c.jpg"),
        _sample(pair="x.jpg->y.jpg"),
        _sample("moving", 10, pair="y.jpg->z.jpg"),
    ]
    folds = grouped_folds(samples, 3)
    assert set.union(*folds) == {sample.pair_id for sample in samples}
    assert sum(len(fold) for fold in folds) == len(set.union(*folds))
    profile_row = threshold_sweep(samples, [_config()])[0]
    rows = cross_validate_profiles(samples, {"Balanced": profile_row}, 3)
    assert any(row["fold"] == "aggregate" for row in rows)
    for fold in folds:
        assert not ({sample.pair_id for sample in samples if sample.pair_id in fold} & {sample.pair_id for sample in samples if sample.pair_id not in fold})


def test_malformed_csv_rows_are_skipped():
    evaluation = [_evaluation_row("a.jpg", "b.jpg", "stationary")]
    evaluation[0]["detection_bbox"] = "bad"
    samples, malformed = load_samples(evaluation, [], {}, _config())
    assert samples == []
    assert malformed and "invalid matched evaluation row" in malformed[0]


def _evaluation_row(previous, current, gt):
    return {
        "previous_frame": previous,
        "current_frame": current,
        "gt_object_index": "0",
        "gt_motion": gt,
        "gt_bbox": "[10,10,30,30]",
        "matched_detection_index": "0",
        "detection_bbox": "[10,10,30,30]",
        "detection_confidence": "0.9",
        "homography_bbox_prediction": "stationary" if gt == "stationary" else "moving",
        "homography_valid": "True",
        "homography_quality_level": "high",
        "matched": "True",
        "ignored": "False",
    }


def _diagnostic_row(previous, current, residual):
    return {
        "frame_previous": previous,
        "frame_current": current,
        "vehicle_index": "0",
        "bbox_center_residual_px": str(residual),
        "bbox_iou": "0.8",
        "bbox_association_score": "0.9",
        "bbox_size_ratio": "1.0",
    }


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_run_creates_all_required_outputs(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    output = tmp_path / "output"
    images.mkdir()
    labels.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        assert cv2.imwrite(str(images / name), np.zeros((60, 80, 3), np.uint8))
    evaluation = tmp_path / "evaluation.csv"
    diagnostics = tmp_path / "diagnostics.csv"
    _write_csv(
        evaluation,
        [
            _evaluation_row("a.jpg", "b.jpg", "stationary"),
            _evaluation_row("b.jpg", "c.jpg", "moving"),
        ],
    )
    _write_csv(
        diagnostics,
        [
            _diagnostic_row("a.jpg", "b.jpg", 2),
            _diagnostic_row("b.jpg", "c.jpg", 10),
        ],
    )
    report = run_optimization(
        OptimizationOptions(images, labels, evaluation, output, diagnostics, folds=2),
        emit=lambda _: None,
        configs=[_config(), _config(stationary_threshold=5, moving_threshold=12)],
    )
    for filename in (
        "homography_bbox_threshold_sweep.csv",
        "homography_bbox_top_configs.csv",
        "homography_bbox_pareto.csv",
        "homography_bbox_parameter_sensitivity.csv",
        "homography_bbox_feature_distribution.csv",
        "homography_bbox_cross_validation.csv",
        "homography_bbox_optimization_summary.txt",
    ):
        assert (output / filename).is_file()
    assert report["sweep"] and report["cross_validation"]


def test_optimizer_has_no_prediction_server_or_post_calls():
    source_path = Path(__file__).parents[1] / "scripts" / "optimize_task1_homography_bbox.py"
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
