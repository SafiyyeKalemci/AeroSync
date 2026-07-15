from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings, get_settings
from app.schemas import DetectedObject, MotionStatus, ObjectClass
from app.services.detection.homography_motion import (
    HomographyMotionAnalyzer,
    HomographyMotionField,
)
from app.services.detection.homography_quality import quality_gate_from_settings
from app.services.detection.motion_analyzer import BBox
from scripts.compare_task1_motion import _bbox, _detect_pair, _homography_analyzer
from scripts.validate_task1_detection import (
    LoadedImage,
    _configure_local_runtime_state,
    load_local_image,
)


@dataclass(frozen=True, slots=True)
class DebugOptions:
    image: Path
    next_image: Path
    output_dir: Path


DetectionProvider = Callable[
    [Settings, LoadedImage, LoadedImage], Awaitable[list[list[DetectedObject]]]
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production homography residual-flow sonucunu tamamen offline incele."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--next-image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


async def _default_detection_provider(
    settings: Settings, previous: LoadedImage, current: LoadedImage
) -> list[list[DetectedObject]]:
    return await _detect_pair(settings, previous, current)


async def run_debug(
    settings: Settings,
    options: DebugOptions,
    *,
    detection_provider: DetectionProvider = _default_detection_provider,
    analyzer: HomographyMotionAnalyzer | None = None,
    emit: Callable[[str], None] = print,
) -> dict[str, object]:
    _configure_local_runtime_state()
    output = options.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous = load_local_image(options.image)
    current = load_local_image(options.next_image)
    if previous.image.shape[:2] != current.image.shape[:2]:
        raise ValueError("Frame çözünürlükleri eşit olmalıdır.")
    detections = await detection_provider(settings, previous, current)
    previous_detections, current_detections = detections
    vehicles = [item for item in current_detections if item.cls is ObjectClass.TASIT]
    exclusions = [
        _bbox(item)
        for item in current_detections
        if item.cls in {ObjectClass.TASIT, ObjectClass.INSAN}
    ]
    motion = analyzer or _homography_analyzer(settings)
    previous_gray = motion.to_grayscale(previous.image)
    current_gray = motion.to_grayscale(current.image)
    computation = motion.analyze_pair(previous_gray, current_gray, exclusions)
    quality = motion.evaluate_quality_gates(
        previous_gray,
        current_gray,
        exclusions,
        {"configured": quality_gate_from_settings(settings)},
    )["configured"]
    diagnostics = computation.diagnostics
    quality_metrics = {
        "quality_level": diagnostics.quality_level or quality.quality_level,
        "reprojection_error": _first_not_none(
            diagnostics.reprojection_error, quality.reprojection_error
        ),
        "spatial_coverage": _first_not_none(
            diagnostics.spatial_coverage, quality.spatial_coverage
        ),
        "projected_overlap": _first_not_none(
            diagnostics.projected_overlap, quality.projected_overlap
        ),
        "metrics_source": (
            "production_diagnostics"
            if diagnostics.reprojection_error is not None
            else "diagnostic_re_evaluation"
        ),
    }
    field = computation.field
    vehicle_reports = [
        _vehicle_report(index, vehicle, motion, field, diagnostics, quality_metrics)
        for index, vehicle in enumerate(vehicles)
    ]
    warped_previous = _warped_previous(previous.image, field, current.image.shape[:2])
    magnitude, full_valid = _full_magnitude(field, current.image.shape[:2])
    heatmap = _heatmap(magnitude, full_valid)
    files = {
        "current_frame_detections": output / "current_frame_detections.jpg",
        "residual_flow_heatmap": output / "residual_flow_heatmap.jpg",
        "residual_flow_vectors": output / "residual_flow_vectors.jpg",
        "json": output / "homography_motion_debug.json",
    }
    _write_detection_visual(files["current_frame_detections"], current.image, vehicle_reports)
    _write_image(files["residual_flow_heatmap"], heatmap)
    _write_vector_visual(files["residual_flow_vectors"], current.image, field)
    roi_dir = output / "per_vehicle_debug"
    roi_dir.mkdir(parents=True, exist_ok=True)
    roi_files = []
    for report in vehicle_reports:
        roi_path = roi_dir / f"vehicle_{report['vehicle_index']:03d}.jpg"
        _write_vehicle_roi(
            roi_path,
            current.image,
            warped_previous,
            heatmap,
            report,
        )
        roi_files.append(str(roi_path))
    payload: dict[str, object] = {
        "frame_metadata": {
            "previous_path": str(previous.path),
            "current_path": str(current.path),
            "previous_sha256": hashlib.sha256(previous.content).hexdigest(),
            "current_sha256": hashlib.sha256(current.content).hexdigest(),
            "images_equal": previous.content == current.content,
            "previous_shape": list(previous.image.shape),
            "current_shape": list(current.image.shape),
        },
        "homography": {
            "valid": diagnostics.valid,
            "reason": diagnostics.reason,
            "quality_level": quality_metrics["quality_level"],
            "matches": diagnostics.match_count,
            "inliers": diagnostics.inlier_count,
            "inlier_ratio": diagnostics.inlier_ratio,
            "condition_number": diagnostics.condition_number,
            "reprojection_error": quality_metrics["reprojection_error"],
            "spatial_coverage": quality_metrics["spatial_coverage"],
            "projected_overlap": quality_metrics["projected_overlap"],
            "quality_metrics_source": quality_metrics["metrics_source"],
        },
        "configured_motion_threshold_px": motion.residual_threshold_px,
        "masking_policy": _masking_policy(motion),
        "vehicles": vehicle_reports,
        "outputs": {**{key: str(value) for key, value in files.items()}, "per_vehicle": roi_files},
        "prediction_submission": "DISABLED",
    }
    files["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _emit_report(emit, payload)
    emit("Prediction submission: DISABLED")
    return payload


def _vehicle_report(
    index: int,
    vehicle: DetectedObject,
    analyzer: HomographyMotionAnalyzer,
    field: HomographyMotionField | None,
    diagnostics: object,
    quality_metrics: dict[str, object],
) -> dict[str, object]:
    measurement = analyzer.measure_vehicle(field, _bbox(vehicle))
    stats = _roi_statistics(field, _bbox(vehicle), analyzer.inner_crop_ratio)
    if measurement.residual_motion_magnitude is not None:
        stats["residual_magnitude_px"] = measurement.residual_motion_magnitude
    return {
        "vehicle_index": index,
        "bbox": list(_bbox(vehicle)),
        "confidence": vehicle.confidence,
        "motion_result": measurement.status.value,
        "residual_median_x": stats["residual_median_x"],
        "residual_median_y": stats["residual_median_y"],
        "residual_magnitude_px": stats["residual_magnitude_px"],
        "configured_motion_threshold_px": analyzer.residual_threshold_px,
        "valid_residual_pixel_count": stats["valid_residual_pixel_count"],
        "invalid_residual_pixel_count": stats["invalid_residual_pixel_count"],
        "invalid_residual_fraction": stats["invalid_residual_fraction"],
        "residual_flow_magnitude_p50": stats["p50"],
        "residual_flow_magnitude_p75": stats["p75"],
        "residual_flow_magnitude_p90": stats["p90"],
        "residual_flow_magnitude_p95": stats["p95"],
        "residual_flow_magnitude_max": stats["max"],
        "homography_valid": diagnostics.valid,
        "homography_quality_level": quality_metrics["quality_level"],
        "homography_matches": diagnostics.match_count,
        "homography_inliers": diagnostics.inlier_count,
        "homography_inlier_ratio": diagnostics.inlier_ratio,
        "homography_reprojection_error": quality_metrics["reprojection_error"],
        "homography_spatial_coverage": quality_metrics["spatial_coverage"],
        "homography_projected_overlap": quality_metrics["projected_overlap"],
        "diagnostic_signals": _diagnostic_signals(vehicle, stats),
    }


def _roi_statistics(
    field: HomographyMotionField | None, bbox: BBox, inner_crop_ratio: float
) -> dict[str, object]:
    import numpy as np

    empty = {
        "residual_median_x": None,
        "residual_median_y": None,
        "residual_magnitude_px": None,
        "valid_residual_pixel_count": 0,
        "invalid_residual_pixel_count": 0,
        "invalid_residual_fraction": None,
        "p50": None,
        "p75": None,
        "p90": None,
        "p95": None,
        "max": None,
    }
    if field is None:
        return empty
    flow = np.asarray(field.flow)
    mask = np.asarray(field.valid_mask, dtype=bool)
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    x1 += width * inner_crop_ratio
    x2 -= width * inner_crop_ratio
    y1 += height * inner_crop_ratio
    y2 -= height * inner_crop_ratio
    left = max(0, min(flow.shape[1], int(math.floor(x1 * field.scale_x))))
    top = max(0, min(flow.shape[0], int(math.floor(y1 * field.scale_y))))
    right = max(0, min(flow.shape[1], int(math.ceil(x2 * field.scale_x))))
    bottom = max(0, min(flow.shape[0], int(math.ceil(y2 * field.scale_y))))
    if right <= left or bottom <= top:
        return empty
    roi = flow[top:bottom, left:right]
    valid = mask[top:bottom, left:right] & np.isfinite(roi).all(axis=2)
    valid_count = int(valid.sum())
    total = int(valid.size)
    if valid_count == 0:
        return {
            **empty,
            "invalid_residual_pixel_count": total,
            "invalid_residual_fraction": 1.0,
        }
    x_values = roi[:, :, 0][valid] / field.scale_x
    y_values = roi[:, :, 1][valid] / field.scale_y
    magnitudes = np.hypot(x_values, y_values)
    median_x = float(np.median(x_values))
    median_y = float(np.median(y_values))
    percentiles = np.percentile(magnitudes, [50, 75, 90, 95])
    return {
        "residual_median_x": median_x,
        "residual_median_y": median_y,
        "residual_magnitude_px": math.hypot(median_x, median_y),
        "valid_residual_pixel_count": valid_count,
        "invalid_residual_pixel_count": total - valid_count,
        "invalid_residual_fraction": (total - valid_count) / total,
        "p50": float(percentiles[0]),
        "p75": float(percentiles[1]),
        "p90": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "max": float(np.max(magnitudes)),
    }


def _full_magnitude(
    field: HomographyMotionField | None, shape: tuple[int, int]
) -> tuple[object, object]:
    import cv2
    import numpy as np

    height, width = shape
    if field is None:
        return np.zeros((height, width), np.float32), np.zeros((height, width), bool)
    flow = np.asarray(field.flow)
    magnitude = np.hypot(flow[:, :, 0] / field.scale_x, flow[:, :, 1] / field.scale_y)
    valid = np.asarray(field.valid_mask, bool) & np.isfinite(magnitude)
    if magnitude.shape != (height, width):
        magnitude = cv2.resize(magnitude, (width, height), interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(valid.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return magnitude, valid


def _heatmap(magnitude: object, valid: object) -> object:
    import cv2
    import numpy as np

    values = np.asarray(magnitude, np.float32)
    mask = np.asarray(valid, bool)
    upper = float(np.percentile(values[mask], 99)) if mask.any() else 1.0
    normalized = np.clip(values / max(upper, 1e-6), 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~mask] = 0
    return colored


def _warped_previous(previous: object, field: HomographyMotionField | None, shape) -> object:
    import cv2
    import numpy as np

    height, width = shape
    if field is None:
        return np.zeros((height, width, 3), np.uint8)
    return cv2.warpPerspective(previous, field.homography, (width, height), flags=cv2.INTER_LINEAR)


def _write_detection_visual(path: Path, image: object, reports: list[dict[str, object]]) -> None:
    import cv2

    canvas = image.copy()
    for report in reports:
        x1, y1, x2, y2 = report["bbox"]
        status = report["motion_result"]
        color = (0, 0, 255) if status == MotionStatus.MOVING.value else (0, 255, 0)
        if status == MotionStatus.UNKNOWN.value:
            color = (0, 255, 255)
        cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        residual = report["residual_magnitude_px"]
        residual_text = "n/a" if residual is None else f"{float(residual):.2f}px"
        label = (
            f"v{report['vehicle_index']} {status} residual={residual_text} "
            f"thr={float(report['configured_motion_threshold_px']):.2f}px"
        )
        cv2.putText(canvas, label, (round(x1), max(18, round(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    _write_image(path, canvas)


def _write_vector_visual(path: Path, image: object, field: HomographyMotionField | None) -> None:
    import cv2
    import numpy as np

    canvas = image.copy()
    if field is not None:
        flow = np.asarray(field.flow)
        valid = np.asarray(field.valid_mask, bool)
        step = max(16, min(flow.shape[:2]) // 30)
        for y in range(step // 2, flow.shape[0], step):
            for x in range(step // 2, flow.shape[1], step):
                if not valid[y, x] or not np.isfinite(flow[y, x]).all():
                    continue
                start = (round(x / field.scale_x), round(y / field.scale_y))
                end = (
                    round(start[0] + float(flow[y, x, 0]) / field.scale_x),
                    round(start[1] + float(flow[y, x, 1]) / field.scale_y),
                )
                cv2.arrowedLine(canvas, start, end, (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.25)
    _write_image(path, canvas)


def _write_vehicle_roi(
    path: Path,
    current: object,
    warped: object,
    heatmap: object,
    report: dict[str, object],
) -> None:
    import cv2
    import numpy as np

    height, width = current.shape[:2]
    x1, y1, x2, y2 = report["bbox"]
    left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
    right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
    if right <= left or bottom <= top:
        panel = np.zeros((120, 360, 3), np.uint8)
    else:
        panels = [current[top:bottom, left:right], warped[top:bottom, left:right], heatmap[top:bottom, left:right]]
        target_height = max(120, max(item.shape[0] for item in panels))
        resized = [
            cv2.resize(item, (max(1, round(item.shape[1] * target_height / item.shape[0])), target_height))
            for item in panels
        ]
        panel = np.concatenate(resized, axis=1)
    label = (
        f"vehicle={report['vehicle_index']} motion={report['motion_result']} "
        f"residual={report['residual_magnitude_px']} threshold={report['configured_motion_threshold_px']}"
    )
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(panel, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    _write_image(path, panel)


def _write_image(path: Path, image: object) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Görsel yazılamadı: {path}")


def _diagnostic_signals(vehicle: DetectedObject, stats: dict[str, object]) -> list[str]:
    signals = [
        "ROI includes all visual content inside the YOLO box after configured inner crop; background/shadow may remain."
    ]
    if vehicle.top_left_x <= 1 or vehicle.top_left_y <= 1:
        signals.append("bbox_touches_frame_edge")
    invalid_fraction = stats["invalid_residual_fraction"]
    if invalid_fraction is not None and float(invalid_fraction) > 0.05:
        signals.append("warp_border_or_invalid_pixels_present")
    p95 = stats["p95"]
    residual = stats["residual_magnitude_px"]
    if p95 is not None and residual is not None and float(p95) > 2 * max(float(residual), 1e-6):
        signals.append("high_tail_residual_possible_parallax_or_bbox_edges")
    return signals


def _masking_policy(analyzer: HomographyMotionAnalyzer) -> dict[str, object]:
    return {
        "homography_feature_exclusions": "Current-frame TASIT and INSAN bboxes are excluded from homography feature tracks.",
        "residual_flow_mask": "Warped previous-frame source support and finite flow only; object bboxes are not removed from residual flow.",
        "vehicle_roi_policy": "Production median uses the bbox after symmetric inner crop.",
        "inner_crop_ratio": analyzer.inner_crop_ratio,
        "warp_border_pixels": "Pixels outside warped source support are excluded by valid_mask.",
        "warp_interpolation_pixels": "Interpolated pixels inside warped source support remain included.",
        "frame_edge_policy": "ROI is clipped to the flow array; supported finite edge pixels remain included.",
        "production_statistic": "Magnitude of component-wise median residual vector, not median of per-pixel magnitudes.",
    }


def _emit_report(emit: Callable[[str], None], payload: dict[str, object]) -> None:
    homography = payload["homography"]
    emit("===== TASK 1 HOMOGRAPHY MOTION DEBUG =====")
    emit(
        f"Homography valid={homography['valid']} reason={homography['reason']} "
        f"matches={homography['matches']} inliers={homography['inliers']} "
        f"ratio={homography['inlier_ratio']:.6f}"
    )
    for report in payload["vehicles"]:
        emit(
            f"vehicle={report['vehicle_index']} motion={report['motion_result']} "
            f"median=({report['residual_median_x']}, {report['residual_median_y']}) "
            f"magnitude={report['residual_magnitude_px']} "
            f"threshold={report['configured_motion_threshold_px']} "
            f"valid_pixels={report['valid_residual_pixel_count']} "
            f"p50={report['residual_flow_magnitude_p50']} "
            f"p95={report['residual_flow_magnitude_p95']}"
        )


def _first_not_none(first: object | None, second: object | None) -> object | None:
    return first if first is not None else second


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(
            run_debug(
                get_settings(),
                DebugOptions(args.image, args.next_image, args.output_dir),
            )
        )
    except Exception as exc:
        print(f"Homography motion debug: FAIL ({type(exc).__name__}: {exc})")
        print("Prediction submission: DISABLED")
        return 10
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
