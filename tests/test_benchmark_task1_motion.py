from __future__ import annotations

import ast
import csv
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import DetectedObject, LandingStatus, MotionStatus, ObjectClass
from scripts.compare_task1_motion import (
    _bbox,
    _bbox_analyzer,
    _global_analyzer,
    _homography_analyzer,
    _hybrid_analyzer,
    _local_analyzer,
)
from scripts.benchmark_task1_motion import (
    CSV_COLUMNS,
    BenchmarkOptions,
    FramePair,
    OfflineMotionPairProcessor,
    PairAnalysis,
    PairDebug,
    build_frame_pairs,
    calculate_disagreements,
    calculate_summary,
    calculate_transitions,
    discover_frames,
    pairing_diagnostics,
    run_benchmark,
)
from scripts.validate_task1_detection import load_local_image


def _files(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"test")


def _row(
    global_result="moving",
    homography="stationary",
    bbox="unknown",
    hybrid="unknown",
    local="unknown",
):
    return {
        "frame_previous": "frame_1.jpg",
        "frame_current": "frame_2.jpg",
        "vehicle_index": 0,
        "bbox": "[1,2,3,4]",
        "global_median_result": global_result,
        "homography_result": homography,
        "homography_bbox_result": bbox,
        "homography_hybrid_result": hybrid,
        "homography_local_result": local,
        "homography_valid": True,
        "homography_reason": "ok",
        "homography_matches": 40,
        "homography_inliers": 35,
        "homography_inlier_ratio": 0.875,
        "homography_residual_px": 1.0,
        "bbox_previous_index": 0,
        "bbox_projected_bbox": "[1,2,3,4]",
        "bbox_iou": 0.9,
        "bbox_center_residual_px": 1.0,
        "bbox_size_ratio": 1.0,
        "bbox_association_score": 0.9,
        "hybrid_bbox_result": bbox,
        "hybrid_flow_result": homography,
        "hybrid_flow_residual_px": 1.0,
        "hybrid_homography_quality_level": "high",
        "hybrid_decision_reason": "test",
        "local_homography_quality_level": "high",
        "local_vehicle_residual_x": 1.0,
        "local_vehicle_residual_y": 0.0,
        "local_vehicle_residual_magnitude": 1.0,
        "local_vehicle_magnitude_p50": 1.0,
        "local_vehicle_magnitude_p75": 1.0,
        "local_vehicle_magnitude_p90": 1.0,
        "local_vehicle_valid_pixels": 100,
        "local_background_residual_x": 1.0,
        "local_background_residual_y": 0.0,
        "local_background_residual_magnitude": 1.0,
        "local_background_magnitude_p50": 1.0,
        "local_background_magnitude_p75": 1.0,
        "local_background_magnitude_p90": 1.0,
        "local_background_valid_pixels": 100,
        "local_background_valid_ratio": 1.0,
        "local_corrected_residual_x": 0.0,
        "local_corrected_residual_y": 0.0,
        "local_corrected_residual_magnitude": 0.0,
        "local_stationary_threshold": 2.0,
        "local_moving_threshold": 6.0,
        "local_decision_reason": "test",
    }


def test_numeric_frame_sorting(tmp_path):
    _files(tmp_path, "frame_10.jpg", "frame_2.jpg", "frame_1.jpg")
    frames = discover_frames(tmp_path)
    assert [item.path.name for item in frames] == [
        "frame_1.jpg",
        "frame_2.jpg",
        "frame_10.jpg",
    ]


def test_consecutive_frame_pairs_are_created(tmp_path):
    _files(tmp_path, "frame_241.jpg", "frame_242.jpg", "frame_243.jpg")
    pairs = build_frame_pairs(discover_frames(tmp_path))
    assert [(p.previous.frame_number, p.current.frame_number) for p in pairs] == [
        (241, 242),
        (242, 243),
    ]


def test_gap_and_different_sequences_are_not_paired(tmp_path):
    _files(tmp_path, "a_1.jpg", "a_3.jpg", "b_4.jpg", "b_5.jpg")
    pairs = build_frame_pairs(discover_frames(tmp_path))
    assert [(p.previous.path.name, p.current.path.name) for p in pairs] == [
        ("b_4.jpg", "b_5.jpg")
    ]


def test_two_numeric_ranges_create_eight_pairs_and_report_gap(tmp_path):
    names = [
        f"2022_2_4_{number:06d}.jpg"
        for number in (*range(241, 246), *range(631, 636))
    ]
    _files(tmp_path, *names)
    frames = discover_frames(tmp_path)
    pairs = build_frame_pairs(frames)
    assert len(pairs) == 8
    assert any(
        "000245.jpg -> 2022_2_4_000631.jpg" in message
        for message in pairing_diagnostics(frames)
    )


@pytest.mark.asyncio
async def test_csv_columns_are_exact(tmp_path):
    images = tmp_path / "images"
    output = tmp_path / "output"
    _files(images, "frame_1.jpg", "frame_2.jpg")

    async def processor(_pair):
        return PairAnalysis((_row(),), False)

    await run_benchmark(
        get_settings(), BenchmarkOptions(images, output), processor=processor, emit=lambda _: None
    )
    with (output / "motion_benchmark.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
        assert len(list(reader)) == 1


def test_summary_counts_and_percentages():
    rows = [
        _row("moving", "stationary", "unknown"),
        _row("stationary", "stationary", "unknown"),
        _row("unknown", "moving", "moving"),
        _row("unknown", "unknown", "stationary"),
    ]
    summary = {item["method"]: item for item in calculate_summary(rows)}
    assert summary["global_median"]["total_detections"] == 4
    assert summary["global_median"]["moving_count"] == 1
    assert summary["global_median"]["stationary_count"] == 1
    assert summary["global_median"]["unknown_count"] == 2
    assert summary["global_median"]["unknown_percentage"] == 50.0


def test_unknown_summary_for_empty_and_unknown_rows():
    empty = {item["method"]: item for item in calculate_summary([])}
    unknown = {item["method"]: item for item in calculate_summary([_row("unknown", "unknown", "unknown")])}
    assert all(item["total_detections"] == 0 for item in empty.values())
    assert all(item["unknown_count"] == 1 for item in unknown.values())
    assert all(item["unknown_percentage"] == 100.0 for item in unknown.values())


def test_method_disagreement_counts():
    rows = [
        _row("moving", "stationary", "unknown"),
        _row("moving", "moving", "moving"),
        _row("unknown", "unknown", "stationary"),
    ]
    assert calculate_disagreements(rows) == {
        "global_median_vs_homography": 1,
        "global_median_vs_homography_bbox": 2,
        "homography_vs_homography_bbox": 2,
        "global_median_vs_homography_hybrid": 2,
        "homography_vs_homography_hybrid": 2,
        "homography_bbox_vs_homography_hybrid": 2,
        "global_median_vs_homography_local": 2,
        "homography_vs_homography_local": 2,
        "homography_bbox_vs_homography_local": 2,
        "homography_hybrid_vs_homography_local": 0,
    }


def test_transition_counts():
    rows = [
        _row("moving", "stationary", "unknown"),
        _row("moving", "unknown", "unknown"),
        _row("stationary", "moving", "unknown"),
    ]
    assert calculate_transitions(rows) == {
        "global_median_moving_to_homography_stationary": 1,
        "global_median_moving_to_homography_unknown": 1,
        "homography_stationary_to_bbox_unknown": 1,
        "homography_moving_to_bbox_unknown": 1,
    }


@pytest.mark.asyncio
async def test_pair_without_detections_still_completes(tmp_path):
    images = tmp_path / "images"
    output = tmp_path / "output"
    _files(images, "frame_1.jpg", "frame_2.jpg")

    async def processor(_pair):
        return PairAnalysis((), False)

    report = await run_benchmark(
        get_settings(), BenchmarkOptions(images, output), processor=processor, emit=lambda _: None
    )
    assert report["frame_pairs_tested"] == 1
    assert report["total_vehicle_detections"] == 0
    assert (output / "motion_benchmark.csv").is_file()


@pytest.mark.asyncio
async def test_corrupt_pair_is_reported_and_next_pair_continues(tmp_path):
    images = tmp_path / "images"
    output = tmp_path / "output"
    _files(images, "frame_1.jpg", "frame_2.jpg", "frame_3.jpg")

    async def processor(pair):
        if pair.previous.frame_number == 1:
            raise ValueError("corrupt image")
        return PairAnalysis((_row(),), False)

    messages: list[str] = []
    report = await run_benchmark(
        get_settings(), BenchmarkOptions(images, output), processor=processor, emit=messages.append
    )
    assert report["frame_pairs_discovered"] == 2
    assert report["frame_pairs_tested"] == 1
    assert len(report["errors"]) == 1
    assert any("PAIR ERROR" in message for message in messages)


@pytest.mark.asyncio
async def test_max_pairs_limits_processing(tmp_path):
    images = tmp_path / "images"
    output = tmp_path / "output"
    _files(images, "frame_1.jpg", "frame_2.jpg", "frame_3.jpg", "frame_4.jpg")
    calls = []

    async def processor(pair):
        calls.append(pair)
        return PairAnalysis((), False)

    report = await run_benchmark(
        get_settings(),
        BenchmarkOptions(images, output, max_pairs=2),
        processor=processor,
        emit=lambda _: None,
    )
    assert len(calls) == 2
    assert report["frame_pairs_discovered"] == 2


@pytest.mark.asyncio
async def test_first_pair_debug_is_printed(tmp_path):
    images = tmp_path / "images"
    output = tmp_path / "output"
    _files(images, "frame_1.jpg", "frame_2.jpg")
    debug = PairDebug(
        images / "frame_1.jpg",
        images / "frame_2.jpg",
        "a" * 64,
        "b" * 64,
        False,
        (1080, 1920, 3),
        (1080, 1920, 3),
        True,
        917,
        909,
        909 / 917,
        "ok",
    )

    async def processor(_pair):
        return PairAnalysis((), False, debug=debug)

    messages: list[str] = []
    await run_benchmark(
        get_settings(), BenchmarkOptions(images, output), processor=processor, emit=messages.append
    )
    output_text = "\n".join(messages)
    assert "previous image SHA-256: " + "a" * 64 in output_text
    assert "current image SHA-256: " + "b" * 64 in output_text
    assert "images_equal: False" in output_text
    assert "homography matches: 917" in output_text
    assert "homography inliers: 909" in output_text


def _vehicle(bbox) -> DetectedObject:
    return DetectedObject(
        cls=ObjectClass.TASIT,
        top_left_x=bbox[0],
        top_left_y=bbox[1],
        bottom_right_x=bbox[2],
        bottom_right_y=bbox[3],
        confidence=0.9,
        motion_status=MotionStatus.UNKNOWN,
        landing_status=LandingStatus.NOT_APPLICABLE,
    )


@pytest.mark.asyncio
async def test_benchmark_matches_compare_algorithm_on_same_fixture_pair(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(42)
    previous_image = rng.integers(0, 256, (180, 240, 3), dtype=np.uint8)
    matrix = np.array([[1, 0, 3], [0, 1, 2]], dtype=np.float32)
    current_image = cv2.warpAffine(previous_image, matrix, (240, 180))
    previous_path = images / "sequence_000001.png"
    current_path = images / "sequence_000002.png"
    assert cv2.imwrite(str(previous_path), previous_image)
    assert cv2.imwrite(str(current_path), current_image)
    previous_detections = [_vehicle((60, 60, 100, 100))]
    current_detections = [_vehicle((63, 62, 103, 102))]

    class Detector:
        async def process_frame(self, frame):
            return previous_detections if frame.frame_index == 1 else current_detections

    settings = replace(
        get_settings(),
        detection_motion_min_valid_pixels=9,
        detection_motion_inner_crop_ratio=0.0,
        detection_motion_flow_downscale=1.0,
        detection_motion_homography_min_features=20,
        detection_motion_homography_min_inliers=12,
        detection_motion_homography_min_inlier_ratio=0.5,
    )
    frames = discover_frames(images)
    pair: FramePair = build_frame_pairs(frames)[0]
    processor = OfflineMotionPairProcessor(settings, images, detector=Detector())
    benchmark = await processor(pair)

    previous_loaded = load_local_image(previous_path)
    current_loaded = load_local_image(current_path)
    global_analyzer = _global_analyzer(settings)
    homography_analyzer = _homography_analyzer(settings)
    bbox_analyzer = _bbox_analyzer(settings, homography_analyzer)
    hybrid_analyzer = _hybrid_analyzer(settings, bbox_analyzer)
    local_analyzer = _local_analyzer(settings, homography_analyzer)
    previous_gray = global_analyzer.to_grayscale(previous_loaded.image)
    current_gray = global_analyzer.to_grayscale(current_loaded.image)
    exclusions = [_bbox(item) for item in current_detections]
    global_field = global_analyzer.compute_flow(previous_gray, current_gray, exclusions)
    comparison = homography_analyzer.analyze_pair(previous_gray, current_gray, exclusions)
    bbox_comparison = bbox_analyzer.analyze(
        previous_gray,
        current_gray,
        previous_detections,
        current_detections,
        exclusions,
        homography_computation=comparison,
    )
    hybrid_comparison = hybrid_analyzer.analyze(
        previous_gray,
        current_gray,
        previous_detections,
        current_detections,
        exclusions,
        homography_computation=comparison,
    )
    local_comparison = local_analyzer.analyze(
        previous_gray,
        current_gray,
        current_detections,
        exclusions,
        homography_computation=comparison,
    )
    compare_global = global_analyzer.classify_vehicle(global_field, _bbox(current_detections[0]))
    compare_homography = homography_analyzer.measure_vehicle(
        comparison.field, _bbox(current_detections[0])
    ).status
    row = benchmark.rows[0]
    assert benchmark.debug is not None
    assert benchmark.debug.homography_valid == comparison.diagnostics.valid
    assert benchmark.debug.homography_matches == comparison.diagnostics.match_count
    assert benchmark.debug.homography_inliers == comparison.diagnostics.inlier_count
    assert benchmark.debug.homography_inlier_ratio == pytest.approx(
        comparison.diagnostics.inlier_ratio
    )
    assert row["global_median_result"] == compare_global.value
    assert row["homography_result"] == compare_homography.value
    assert row["homography_bbox_result"] == bbox_comparison.measurements[0].status.value
    assert (
        row["homography_hybrid_result"]
        == hybrid_comparison.measurements[0].final_result.value
    )
    assert (
        row["homography_local_result"]
        == local_comparison.measurements[0].final_result.value
    )
    assert "local_corrected_residual_magnitude" in CSV_COLUMNS


def test_benchmark_has_no_network_prediction_or_runner_dependency():
    source_path = Path(__file__).parents[1] / "scripts" / "benchmark_task1_motion.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not ({"requests", "httpx", "competition"} & imported_roots)
    assert "send_prediction" not in source
    assert "competition.runner" not in source
    assert ".post(" not in source.casefold()
