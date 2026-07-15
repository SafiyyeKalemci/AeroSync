from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
from pathlib import Path

from app.core.config import Settings, get_settings
from app.schemas import DetectedObject, MotionStatus, ObjectClass
from app.services.detection.homography_motion import HomographyMotionAnalyzer
from app.services.detection.homography_bbox_motion import HomographyBBoxMotionAnalyzer
from app.services.detection.homography_hybrid_motion import HomographyHybridMotionAnalyzer
from app.services.detection.homography_local_motion import (
    HomographyLocalMotionAnalyzer,
    LocalMotionMeasurement,
)
from app.services.detection.homography_quality import quality_gate_from_settings
from app.services.detection.motion_analyzer import BBox, MotionAnalyzer
from app.services.detection.service import YoloDetectionService
from app.services.detection.yolo_runtime import YoloRuntime
from scripts.validate_task1_detection import (
    LoadedImage,
    ValidationOptions,
    _configure_local_runtime_state,
    _frame_context,
    load_local_image,
)


@dataclass(frozen=True, slots=True)
class ComparisonOptions:
    image: Path
    next_image: Path
    save_visualization: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task 1 global_median ve homography motion yöntemlerini yerel karşılaştır."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--next-image", required=True, type=Path)
    parser.add_argument("--save-visualization", type=Path)
    return parser


def _global_analyzer(settings: Settings) -> MotionAnalyzer:
    return MotionAnalyzer(
        threshold_px=settings.detection_motion_threshold_px,
        min_valid_pixels=settings.detection_motion_min_valid_pixels,
        inner_crop_ratio=settings.detection_motion_inner_crop_ratio,
        flow_downscale=settings.detection_motion_flow_downscale,
        freeze_threshold=settings.detection_motion_freeze_threshold,
    )


def _homography_analyzer(settings: Settings) -> HomographyMotionAnalyzer:
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
        quality_gate=quality_gate_from_settings(settings),
    )


def _bbox_analyzer(
    settings: Settings, homography: HomographyMotionAnalyzer
) -> HomographyBBoxMotionAnalyzer:
    return HomographyBBoxMotionAnalyzer(
        homography,
        match_min_iou=settings.detection_motion_bbox_match_min_iou,
        match_max_center_distance_ratio=settings.detection_motion_bbox_match_max_center_distance_ratio,
        match_min_score=settings.detection_motion_bbox_match_min_score,
        stationary_threshold_px=settings.detection_motion_bbox_stationary_threshold_px,
        moving_threshold_px=settings.detection_motion_bbox_moving_threshold_px,
        min_size_ratio=settings.detection_motion_bbox_min_size_ratio,
        max_size_ratio=settings.detection_motion_bbox_max_size_ratio,
        min_visible_ratio=settings.detection_motion_bbox_min_visible_ratio,
    )


def _hybrid_analyzer(
    settings: Settings, bbox: HomographyBBoxMotionAnalyzer
) -> HomographyHybridMotionAnalyzer:
    return HomographyHybridMotionAnalyzer(
        bbox,
        strong_moving_residual_px=(
            settings.detection_motion_hybrid_strong_moving_residual_px
        ),
        min_association_score=settings.detection_motion_hybrid_min_association_score,
        min_iou=settings.detection_motion_hybrid_min_iou,
    )


def _local_analyzer(
    settings: Settings, homography: HomographyMotionAnalyzer
) -> HomographyLocalMotionAnalyzer:
    return HomographyLocalMotionAnalyzer(
        homography,
        ring_expansion_ratio=settings.detection_motion_local_ring_expansion_ratio,
        min_background_pixels=settings.detection_motion_local_min_background_pixels,
        stationary_threshold_px=(
            settings.detection_motion_local_stationary_threshold_px
        ),
        moving_threshold_px=settings.detection_motion_local_moving_threshold_px,
        min_valid_ratio=settings.detection_motion_local_min_valid_ratio,
    )


async def _detect_pair(
    settings: Settings, first: LoadedImage, second: LoadedImage
) -> list[list[DetectedObject]]:
    content = {str(first.path): first.content, str(second.path): second.content}

    async def reader(source: str, _timeout: float) -> bytes:
        if source not in content:
            raise ValueError("Yalnızca doğrulanmış yerel görüntüler kabul edilir.")
        return content[source]

    detection_settings = replace(settings, detection_motion_enabled=False)
    runtime = YoloRuntime(
        settings.detection_model_path,
        settings.detection_confidence,
        settings.detection_iou,
    )
    service = YoloDetectionService(
        detection_settings, runtime=runtime, image_reader=reader
    )
    context_options = ValidationOptions(first.path, second.path)
    return [
        await service.process_frame(_frame_context(context_options, first, 0)),
        await service.process_frame(_frame_context(context_options, second, 1)),
    ]


def _bbox(item: DetectedObject) -> BBox:
    return (
        item.top_left_x,
        item.top_left_y,
        item.bottom_right_x,
        item.bottom_right_y,
    )


def _target_paths(path: Path) -> tuple[Path, Path, Path, Path, Path]:
    resolved = path.expanduser().resolve()
    if resolved.is_dir() or not resolved.suffix:
        resolved.mkdir(parents=True, exist_ok=True)
        return (
            resolved / "motion_global_median.png",
            resolved / "motion_homography.png",
            resolved / "motion_homography_bbox.png",
            resolved / "motion_homography_hybrid.png",
            resolved / "motion_homography_local.png",
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return (
        resolved.with_name(f"{resolved.stem}_global_median.png"),
        resolved.with_name(f"{resolved.stem}_homography.png"),
        resolved.with_name(f"{resolved.stem}_homography_bbox.png"),
        resolved.with_name(f"{resolved.stem}_homography_hybrid.png"),
        resolved.with_name(f"{resolved.stem}_homography_local.png"),
    )


def _save_comparison(
    path: Path,
    image: object,
    vehicles: list[DetectedObject],
    statuses: list[MotionStatus],
    title: str,
    projected_boxes: list[BBox | None] | None = None,
) -> None:
    import cv2

    canvas = image.copy()
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    projected_boxes = projected_boxes or [None] * len(vehicles)
    for item, status, projected in zip(
        vehicles, statuses, projected_boxes, strict=True
    ):
        point1 = (round(item.top_left_x), round(item.top_left_y))
        point2 = (round(item.bottom_right_x), round(item.bottom_right_y))
        color = (0, 255, 0) if status is MotionStatus.STATIONARY else (0, 0, 255)
        if status is MotionStatus.UNKNOWN:
            color = (0, 255, 255)
        cv2.rectangle(canvas, point1, point2, color, 2)
        if projected is not None:
            projected_start = (round(projected[0]), round(projected[1]))
            projected_end = (round(projected[2]), round(projected[3]))
            cv2.rectangle(canvas, projected_start, projected_end, (255, 128, 0), 1)
        cv2.putText(
            canvas,
            status.value,
            (point1[0], max(16, point1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Görselleştirme kaydedilemedi: {path}")


def _save_local_comparison(
    path: Path,
    image: object,
    vehicles: list[DetectedObject],
    measurements: list[LocalMotionMeasurement],
) -> None:
    import cv2

    canvas = image.copy()
    cv2.putText(
        canvas,
        "homography_local",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    for vehicle, measurement in zip(vehicles, measurements, strict=True):
        point1 = (round(vehicle.top_left_x), round(vehicle.top_left_y))
        point2 = (round(vehicle.bottom_right_x), round(vehicle.bottom_right_y))
        color = (
            (0, 255, 0)
            if measurement.final_result is MotionStatus.STATIONARY
            else (0, 0, 255)
            if measurement.final_result is MotionStatus.MOVING
            else (0, 255, 255)
        )
        cv2.rectangle(canvas, point1, point2, color, 2)
        vehicle_residual = _stat_magnitude(measurement.vehicle_statistics)
        background_residual = _stat_magnitude(measurement.background_statistics)
        corrected = _number(measurement.corrected_residual_magnitude)
        labels = (
            measurement.final_result.value,
            f"v={vehicle_residual} bg={background_residual}",
            f"corrected={corrected}",
        )
        start_y = max(16, point1[1] - 36)
        for offset, label in enumerate(labels):
            cv2.putText(
                canvas,
                label,
                (point1[0], start_y + offset * 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Görselleştirme kaydedilemedi: {path}")


def _stat_magnitude(statistics: object | None) -> str:
    return _number(
        getattr(statistics, "vector_magnitude", None) if statistics is not None else None
    )


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}px"


async def run_comparison(
    settings: Settings,
    options: ComparisonOptions,
) -> int:
    _configure_local_runtime_state()
    first = load_local_image(options.image)
    second = load_local_image(options.next_image)
    if first.image.shape[:2] != second.image.shape[:2]:
        print("Frame comparison: FAIL (resolution change)")
        print("Prediction submission: DISABLED")
        return 20
    detections = await _detect_pair(settings, first, second)
    vehicles = [item for item in detections[1] if item.cls is ObjectClass.TASIT]
    exclusions = [
        _bbox(item)
        for item in detections[1]
        if item.cls in {ObjectClass.TASIT, ObjectClass.INSAN}
    ]
    global_analyzer = _global_analyzer(settings)
    homography_analyzer = _homography_analyzer(settings)
    bbox_analyzer = _bbox_analyzer(settings, homography_analyzer)
    hybrid_analyzer = _hybrid_analyzer(settings, bbox_analyzer)
    local_analyzer = _local_analyzer(settings, homography_analyzer)
    previous_gray = global_analyzer.to_grayscale(first.image)
    current_gray = global_analyzer.to_grayscale(second.image)
    global_field = global_analyzer.compute_flow(previous_gray, current_gray, exclusions)
    computation = homography_analyzer.analyze_pair(
        previous_gray, current_gray, exclusions
    )
    diagnostics = computation.diagnostics
    global_statuses = [global_analyzer.classify_vehicle(global_field, _bbox(item)) for item in vehicles]
    measurements = [homography_analyzer.measure_vehicle(computation.field, _bbox(item)) for item in vehicles]
    bbox_analysis = bbox_analyzer.analyze(
        previous_gray,
        current_gray,
        detections[0],
        detections[1],
        exclusions,
        homography_computation=computation,
    )
    bbox_measurements = list(bbox_analysis.measurements)
    hybrid_analysis = hybrid_analyzer.analyze(
        previous_gray,
        current_gray,
        detections[0],
        detections[1],
        exclusions,
        homography_computation=computation,
    )
    hybrid_measurements = list(hybrid_analysis.measurements)
    local_analysis = local_analyzer.analyze(
        previous_gray,
        current_gray,
        detections[1],
        exclusions,
        homography_computation=computation,
    )
    local_measurements = list(local_analysis.measurements)

    print("Task 1 motion A/B/C/D/E comparison")
    print(
        f"Homography valid={diagnostics.valid} reason={diagnostics.reason} "
        f"matches={diagnostics.match_count} inliers={diagnostics.inlier_count} "
        f"inlier_ratio={diagnostics.inlier_ratio:.3f}"
    )
    for index, (item, global_status, measurement, bbox_measurement, hybrid, local) in enumerate(
        zip(
            vehicles,
            global_statuses,
            measurements,
            bbox_measurements,
            hybrid_measurements,
            local_measurements,
            strict=True,
        )
    ):
        residual = (
            "n/a"
            if measurement.residual_motion_magnitude is None
            else f"{measurement.residual_motion_magnitude:.3f}px"
        )
        bbox_residual = (
            "n/a"
            if bbox_measurement.center_residual_px is None
            else f"{bbox_measurement.center_residual_px:.3f}px"
        )
        print(
            f"vehicle={index} bbox={_bbox(item)} global_median={global_status.value} "
            f"homography={measurement.status.value} homography_valid={diagnostics.valid} "
            f"matches={diagnostics.match_count} inliers={diagnostics.inlier_count} "
            f"inlier_ratio={diagnostics.inlier_ratio:.3f} residual={residual} "
            f"homography_bbox={bbox_measurement.status.value} "
            f"projected_bbox={bbox_measurement.projected_bbox} "
            f"previous_index={bbox_measurement.previous_index} "
            f"iou={bbox_measurement.iou} center_residual={bbox_residual} "
            f"size_ratio={bbox_measurement.size_ratio} "
            f"association_score={bbox_measurement.association_score} "
            f"homography_hybrid={hybrid.final_result.value} "
            f"hybrid_bbox_result={hybrid.bbox_result.value} "
            f"hybrid_flow_result={hybrid.flow_result.value} "
            f"hybrid_flow_residual={hybrid.flow_residual_px} "
            f"homography_quality={hybrid.homography_quality_level} "
            f"decision_reason={hybrid.decision_reason} "
            f"homography_local={local.final_result.value} "
            f"local_vehicle_residual={_stat_magnitude(local.vehicle_statistics)} "
            f"local_background_residual={_stat_magnitude(local.background_statistics)} "
            f"local_corrected_residual={_number(local.corrected_residual_magnitude)} "
            f"local_background_valid_ratio={local.background_valid_ratio:.3f} "
            f"local_decision_reason={local.decision_reason} "
            f"comparison={global_status.value}->{measurement.status.value}"
            f"->{bbox_measurement.status.value}->{hybrid.final_result.value}"
            f"->{local.final_result.value}"
        )
    if options.save_visualization is not None:
        global_path, homography_path, bbox_path, hybrid_path, local_path = _target_paths(
            options.save_visualization
        )
        _save_comparison(global_path, second.image, vehicles, global_statuses, "global_median")
        _save_comparison(
            homography_path,
            second.image,
            vehicles,
            [item.status for item in measurements],
            "homography",
        )
        _save_local_comparison(local_path, second.image, vehicles, local_measurements)
        _save_comparison(
            hybrid_path,
            second.image,
            vehicles,
            [item.final_result for item in hybrid_measurements],
            "homography_hybrid",
            [item.projected_bbox for item in hybrid_measurements],
        )
        _save_comparison(
            bbox_path,
            second.image,
            vehicles,
            [item.status for item in bbox_measurements],
            "homography_bbox",
            [item.projected_bbox for item in bbox_measurements],
        )
        print(f"Visualization global_median: {global_path}")
        print(f"Visualization homography: {homography_path}")
        print(f"Visualization homography_bbox: {bbox_path}")
        print(f"Visualization homography_hybrid: {hybrid_path}")
        print(f"Visualization homography_local: {local_path}")
    print("Prediction submission: DISABLED")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(
            run_comparison(
                get_settings(),
                ComparisonOptions(args.image, args.next_image, args.save_visualization),
            )
        )
    except Exception as exc:
        print(f"Frame comparison: FAIL ({type(exc).__name__}: {exc})")
        print("Prediction submission: DISABLED")
        return 20


if __name__ == "__main__":
    raise SystemExit(main())
