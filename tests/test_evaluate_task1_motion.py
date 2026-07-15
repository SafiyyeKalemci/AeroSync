from __future__ import annotations

import ast
import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import DetectedObject, LandingStatus, MotionStatus, ObjectClass
from scripts.benchmark_task1_motion import PairAnalysis, build_frame_pairs, discover_frames
from scripts.evaluate_task1_motion import (
    EvaluationOptions,
    GroundTruthObject,
    associate_objects,
    calculate_metrics,
    normalize_motion_label,
    parse_annotation,
    run_evaluation,
    validate_dataset,
)


def _xml(path: Path, objects: list[tuple[str, tuple[int, int, int, int]]], *, filename=None):
    entries = "".join(
        f"<object><name>{motion}</name><bndbox><xmin>{bbox[0]}</xmin>"
        f"<ymin>{bbox[1]}</ymin><xmax>{bbox[2]}</xmax><ymax>{bbox[3]}</ymax>"
        "</bndbox></object>"
        for motion, bbox in objects
    )
    path.write_text(
        f"<annotation><filename>{filename or path.stem + '.jpg'}</filename>"
        f"<size><width>100</width><height>100</height></size>{entries}</annotation>",
        encoding="utf-8",
    )


def _image(path: Path):
    assert cv2.imwrite(str(path), np.zeros((100, 100, 3), np.uint8))


def _detection(bbox, confidence=0.9):
    return DetectedObject(
        cls=ObjectClass.TASIT,
        top_left_x=bbox[0],
        top_left_y=bbox[1],
        bottom_right_x=bbox[2],
        bottom_right_y=bbox[3],
        confidence=confidence,
        motion_status=MotionStatus.UNKNOWN,
        landing_status=LandingStatus.NOT_APPLICABLE,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MOVING", "moving"),
        ("hareketli", "moving"),
        ("STATIONARY", "stationary"),
        ("Hareketsiz", "stationary"),
        ("IGNORE", "ignore"),
        ("yoksay", "ignore"),
        ("invalid", None),
    ],
)
def test_motion_label_normalization(raw, expected):
    assert normalize_motion_label(raw) == expected


def test_xml_and_bbox_parse(tmp_path):
    image = tmp_path / "frame_1.jpg"
    xml = tmp_path / "frame_1.xml"
    _image(image)
    _xml(xml, [("MOVING", (10, 20, 30, 40)), ("IGNORE", (50, 50, 70, 70))])
    parsed = parse_annotation(xml, image)
    assert parsed.errors == ()
    assert parsed.objects[0].motion == "moving"
    assert parsed.objects[0].bbox == (10.0, 20.0, 30.0, 40.0)
    assert parsed.objects[1].ignored is True


def test_malformed_xml_and_missing_image_are_reported(tmp_path):
    malformed = tmp_path / "bad.xml"
    malformed.write_text("<annotation>", encoding="utf-8")
    assert "malformed_xml" in parse_annotation(malformed).errors[0]
    valid = tmp_path / "frame_1.xml"
    _xml(valid, [("MOVING", (10, 10, 20, 20))])
    assert "missing_image" in parse_annotation(valid).errors


def test_duplicate_invalid_label_and_out_of_bounds_are_reported(tmp_path):
    image = tmp_path / "frame_1.jpg"
    xml = tmp_path / "frame_1.xml"
    _image(image)
    _xml(
        xml,
        [
            ("MOVING", (10, 10, 20, 20)),
            ("MOVING", (10, 10, 20, 20)),
            ("UNKNOWN_LABEL", (30, 30, 40, 40)),
            ("STATIONARY", (90, 90, 110, 110)),
        ],
    )
    parsed = parse_annotation(xml, image)
    assert len(parsed.objects) == 1
    assert any("duplicate_object" in error for error in parsed.errors)
    assert any("invalid_motion_label" in error for error in parsed.errors)
    assert any("bbox_out_of_bounds" in error for error in parsed.errors)


def test_iou_association_is_one_to_one_and_reports_unmatched():
    ground_truth = [
        GroundTruthObject(0, "moving", (10, 10, 30, 30)),
        GroundTruthObject(1, "stationary", (12, 10, 32, 30)),
        GroundTruthObject(2, "stationary", (70, 70, 90, 90)),
    ]
    detections = [_detection((10, 10, 30, 30)), _detection((40, 40, 60, 60))]
    result = associate_objects(ground_truth, detections, 0.5)
    assert result.gt_to_detection == {0: (0, 1.0)}
    assert result.unmatched_gt == (1, 2)
    assert result.unmatched_detections == (1,)


def _metric_row(gt, prediction, *, ignored=False, matched=True):
    row = {"gt_motion": gt, "ignored": ignored, "matched": matched}
    for method in (
        "global_median",
        "homography",
        "homography_bbox",
        "homography_hybrid",
        "homography_local",
    ):
        row[f"{method}_prediction"] = prediction
    return row


def test_unknown_metrics_strict_decided_coverage_prf_and_confusion():
    rows = [
        _metric_row("moving", "moving"),
        _metric_row("moving", "unknown"),
        _metric_row("stationary", "stationary"),
        _metric_row("stationary", "moving"),
        _metric_row("moving", "stationary", ignored=True),
        _metric_row("moving", "moving", matched=False),
    ]
    summary, confusion = calculate_metrics(
        rows, total_gt=6, matched_gt=4, unmatched_gt=1, ignored_gt=1
    )
    result = summary[0]
    assert result["total_evaluated"] == 4
    assert result["strict_accuracy"] == pytest.approx(0.5)
    assert result["decided_only_accuracy"] == pytest.approx(2 / 3)
    assert result["coverage"] == pytest.approx(0.75)
    assert result["moving_precision"] == pytest.approx(0.5)
    assert result["moving_recall"] == pytest.approx(0.5)
    assert result["stationary_precision"] == pytest.approx(1.0)
    assert result["stationary_recall"] == pytest.approx(0.5)
    assert result["balanced_score"] == pytest.approx(0.375)
    selected = [
        row for row in confusion
        if row["method"] == "global_median"
        and row["gt_motion"] == "moving"
        and row["predicted_motion"] == "unknown"
    ]
    assert selected[0]["count"] == 1


def _analysis(current_image):
    vehicles = (
        _detection((10, 10, 30, 30), 0.91),
        _detection((60, 60, 80, 80), 0.82),
    )
    rows = []
    for index, result in enumerate(("moving", "unknown")):
        rows.append(
            {
                "vehicle_index": index,
                "global_median_result": result,
                "homography_result": result,
                "homography_bbox_result": result,
                "homography_hybrid_result": result,
                "homography_local_result": result,
                "homography_valid": True,
                "homography_inlier_ratio": 0.9,
                "homography_residual_px": 3.0,
                "bbox_iou": 0.8,
                "bbox_center_residual_px": 2.0,
                "hybrid_homography_quality_level": "high",
                "local_corrected_residual_magnitude": 1.0,
            }
        )
    return PairAnalysis(tuple(rows), False, current_image=current_image, vehicles=vehicles)


@pytest.mark.asyncio
async def test_evaluation_writes_csvs_and_reports_ignore_and_unmatched(tmp_path):
    images, labels, output = tmp_path / "images", tmp_path / "labels", tmp_path / "output"
    images.mkdir()
    labels.mkdir()
    for number in (1, 2):
        _image(images / f"frame_{number}.jpg")
    _xml(labels / "frame_1.xml", [("STATIONARY", (10, 10, 30, 30))])
    _xml(
        labels / "frame_2.xml",
        [
            ("MOVING", (10, 10, 30, 30)),
            ("STATIONARY", (35, 35, 50, 50)),
            ("IGNORE", (60, 60, 80, 80)),
        ],
    )

    async def processor(_pair):
        return _analysis(np.zeros((100, 100, 3), np.uint8))

    report = await run_evaluation(
        get_settings(),
        EvaluationOptions(images, labels, output, min_iou=0.5),
        processor=processor,
        emit=lambda _: None,
    )
    assert report["pairs"] == 1
    assert report["matched_ground_truth"] == 1
    assert report["unmatched_ground_truth"] == 1
    assert report["unmatched_detections"] == 0
    assert report["summary"][0]["total_evaluated"] == 1
    for filename in (
        "motion_evaluation_detailed.csv",
        "motion_evaluation_summary.csv",
        "motion_evaluation_confusion.csv",
        "unmatched_ground_truth.csv",
        "unmatched_detections.csv",
        "xml_validation_errors.csv",
    ):
        assert (output / filename).is_file()
    with (output / "motion_evaluation_detailed.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        detailed = list(csv.DictReader(handle))
    assert len(detailed) == 3
    assert sum(row["ignored"] == "True" for row in detailed) == 1
    assert sum(row["matched"] == "False" for row in detailed) == 1


def test_missing_xml_and_non_consecutive_frames_are_reported(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    _image(images / "frame_1.jpg")
    _image(images / "frame_3.jpg")
    _xml(labels / "frame_1.xml", [("MOVING", (1, 1, 10, 10))])
    _, issues = validate_dataset(images, labels)
    assert any(issue["error"] == "missing_xml" for issue in issues)
    assert build_frame_pairs(discover_frames(images)) == []


def test_evaluation_has_no_prediction_server_post_or_runner_dependency():
    source_path = Path(__file__).parents[1] / "scripts" / "evaluate_task1_motion.py"
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
