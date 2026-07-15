from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.core.config import Settings, get_settings
from app.schemas import DetectedObject, ImageModality, LandingStatus, MotionStatus, ObjectClass
from app.services.common import FrameContext
from app.services.detection.class_mapping import YoloClassMapper
from app.services.detection.landing_analyzer import (
    BBox,
    LandingAnalyzer,
    LandingPolicy,
    _valid_bbox,
    intersection_metrics,
)
from app.services.detection.service import YoloDetectionService
from app.services.detection.yolo_runtime import YoloRuntime
from scripts.validate_task1_detection import (
    RecordingYoloRuntime,
    _configure_local_runtime_state,
    load_local_image,
    model_mapping_report,
)

EXIT_OK = 0
EXIT_CONFIG = 10
EXIT_IMAGE = 20
EXIT_MODEL = 30


@dataclass(frozen=True, slots=True)
class LandingCall:
    raw_bbox: BBox
    clipped_bbox: BBox
    frame_width: int
    frame_height: int
    obstacles: tuple[BBox, ...]
    result: LandingStatus


class RecordingLandingAnalyzer:
    """Delegates every decision to production LandingAnalyzer and records its exact inputs."""

    def __init__(self, analyzer: LandingAnalyzer) -> None:
        self._analyzer = analyzer
        self.calls: list[LandingCall] = []

    def analyze(self, *, raw_bbox, clipped_bbox, frame_width, frame_height, obstacles):
        result = self._analyzer.analyze(
            raw_bbox=raw_bbox,
            clipped_bbox=clipped_bbox,
            frame_width=frame_width,
            frame_height=frame_height,
            obstacles=obstacles,
        )
        self.calls.append(
            LandingCall(
                tuple(raw_bbox),
                tuple(clipped_bbox),
                int(frame_width),
                int(frame_height),
                tuple(tuple(item) for item in obstacles),
                result,
            )
        )
        return result


@dataclass(frozen=True, slots=True)
class ValidationOptions:
    image: Path | None = None
    synthetic: bool = False
    save_visualization: Path | None = None
    json_output: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production LandingAnalyzer ile tamamen offline Task 1 landing dogrulamasi"
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--save-visualization", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser


def policy_from_settings(settings: Settings) -> LandingPolicy:
    settings.validate_detection_landing()
    return LandingPolicy(
        edge_margin_px=settings.detection_landing_edge_margin_px,
        edge_margin_ratio=settings.detection_landing_edge_margin_ratio,
        min_intersection_pixels=settings.detection_landing_min_intersection_pixels,
        occupancy_ratio=settings.detection_landing_occupancy_ratio,
        use_center_check=settings.detection_landing_use_center_check,
        use_bottom_center_check=settings.detection_landing_use_bottom_center_check,
        min_area_pixels=settings.detection_landing_min_area_pixels,
    )


def landing_config_report(settings: Settings) -> dict[str, object]:
    return {
        "DETECTION_LANDING_ENABLED": settings.detection_landing_enabled,
        "DETECTION_LANDING_EDGE_MARGIN_PX": settings.detection_landing_edge_margin_px,
        "DETECTION_LANDING_EDGE_MARGIN_RATIO": settings.detection_landing_edge_margin_ratio,
        "DETECTION_LANDING_MIN_INTERSECTION_PIXELS": settings.detection_landing_min_intersection_pixels,
        "DETECTION_LANDING_OCCUPANCY_RATIO": settings.detection_landing_occupancy_ratio,
        "DETECTION_LANDING_USE_CENTER_CHECK": settings.detection_landing_use_center_check,
        "DETECTION_LANDING_USE_BOTTOM_CENTER_CHECK": settings.detection_landing_use_bottom_center_check,
        "DETECTION_LANDING_MIN_AREA_PIXELS": settings.detection_landing_min_area_pixels,
    }


def landing_diagnostics(
    call: LandingCall,
    policy: LandingPolicy,
    obstacle_classes: SequenceObjectClass = (),
) -> dict[str, object]:
    raw_valid = _valid_bbox(call.raw_bbox)
    clipped_valid = _valid_bbox(call.clipped_bbox)
    effective_margin = max(
        policy.edge_margin_px,
        min(call.frame_width, call.frame_height) * policy.edge_margin_ratio,
    )
    raw_area = _bbox_area(call.raw_bbox) if raw_valid else 0.0
    visible_area = _bbox_area(call.clipped_bbox) if clipped_valid else 0.0
    visible_ratio = visible_area / raw_area if raw_area > 0 else 0.0
    frame_edge = bool(
        raw_valid
        and (
            call.raw_bbox[0] < 0
            or call.raw_bbox[1] < 0
            or call.raw_bbox[2] > call.frame_width
            or call.raw_bbox[3] > call.frame_height
            or call.clipped_bbox[0] <= 1
            or call.clipped_bbox[1] <= 1
            or call.clipped_bbox[2] >= call.frame_width - 1
            or call.clipped_bbox[3] >= call.frame_height - 1
        )
    )
    obstacle_reports: list[dict[str, object]] = []
    occupied_class: ObjectClass | None = None
    if raw_valid and clipped_valid:
        for index, obstacle in enumerate(call.obstacles):
            obstacle_class = obstacle_classes[index] if index < len(obstacle_classes) else None
            if not _valid_bbox(obstacle):
                obstacle_reports.append(
                    {
                        "obstacle_class": obstacle_class.value if obstacle_class else "unknown",
                        "obstacle_bbox": _bbox_dict(obstacle),
                        "valid": False,
                    }
                )
                continue
            metrics = intersection_metrics(call.clipped_bbox, obstacle)
            occupied = metrics.intersection_area >= policy.min_intersection_pixels and (
                metrics.intersection_over_landing_area >= policy.occupancy_ratio
                or (policy.use_center_check and metrics.obstacle_center_inside)
                or (policy.use_bottom_center_check and metrics.obstacle_bottom_center_inside)
            )
            obstacle_reports.append(
                {
                    "obstacle_class": obstacle_class.value if obstacle_class else "unknown",
                    "obstacle_bbox": _bbox_dict(obstacle),
                    "valid": True,
                    "intersection_area": metrics.intersection_area,
                    "intersection_over_landing_area": metrics.intersection_over_landing_area,
                    "obstacle_center_inside": metrics.obstacle_center_inside,
                    "obstacle_bottom_center_inside": metrics.obstacle_bottom_center_inside,
                    "occupied_by_production_policy": occupied,
                }
            )
            if occupied and occupied_class is None:
                occupied_class = obstacle_class

    reason = _decision_reason(call, policy, raw_valid, clipped_valid, occupied_class)
    return {
        "raw_bbox": _bbox_dict(call.raw_bbox),
        "clipped_bbox": _bbox_dict(call.clipped_bbox),
        "frame_width": call.frame_width,
        "frame_height": call.frame_height,
        "bbox_area": raw_area,
        "visible_area": visible_area,
        "visible_ratio": visible_ratio,
        "frame_edge": frame_edge,
        "bbox_valid": raw_valid and clipped_valid,
        "min_area_check": {
            "threshold": policy.min_area_pixels,
            "passes": clipped_valid and visible_area >= policy.min_area_pixels,
        },
        "edge_margin_px": effective_margin,
        "edge_margin_ratio": policy.edge_margin_ratio,
        "visible_ratio_policy": "reported_only_not_a_production_gate",
        "obstacle_count": len(call.obstacles),
        "obstacles": obstacle_reports,
        "final_landing_status": call.result.value,
        "decision_reason": reason,
    }


def _decision_reason(call, policy, raw_valid, clipped_valid, occupied_class):
    if call.frame_width <= 0 or call.frame_height <= 0 or not raw_valid or not clipped_valid:
        return "invalid_bbox"
    margin = max(policy.edge_margin_px, min(call.frame_width, call.frame_height) * policy.edge_margin_ratio)
    x1, y1, x2, y2 = call.raw_bbox
    if x1 < -margin or y1 < -margin or x2 > call.frame_width + margin or y2 > call.frame_height + margin:
        return "outside_frame_tolerance"
    if _bbox_area(call.clipped_bbox) < policy.min_area_pixels:
        return "bbox_too_small"
    if occupied_class is ObjectClass.TASIT:
        return "occupied_by_vehicle"
    if occupied_class is ObjectClass.INSAN:
        return "occupied_by_person"
    if call.result is LandingStatus.UNSUITABLE:
        return "occupied_by_unknown_obstacle"
    if call.result is LandingStatus.SUITABLE:
        return "clear_landing_area"
    return "not_applicable"


async def run_real_validation(
    settings: Settings,
    options: ValidationOptions,
    *,
    runtime_factory: Callable[[Settings], object] | None = None,
    image_loader: Callable[[Path], object] = load_local_image,
    emit: Callable[[str], None] = print,
) -> tuple[int, dict[str, object]]:
    if options.image is None:
        raise ValueError("--image is required outside synthetic mode")
    policy = policy_from_settings(settings)
    config = landing_config_report(settings)
    _emit_config(config, emit)
    report: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "real_image",
        "landing_config": config,
        "model_info": {
            "model_path": str(settings.detection_model_path) if settings.detection_model_path else None,
            "model_exists": bool(settings.detection_model_path and settings.detection_model_path.is_file()),
            "load_status": "NOT_ATTEMPTED",
        },
        "frame_metadata": {},
        "detections": [],
        "prediction_submission": "DISABLED",
    }
    try:
        loaded = image_loader(options.image)
    except Exception as exc:
        report["final_result"] = "FAIL"
        report["failure_reason"] = f"image_load_failed:{type(exc).__name__}"
        _write_json(options.json_output, report)
        emit(f"Image validation: FAIL ({type(exc).__name__})")
        emit("Prediction submission: DISABLED")
        return EXIT_IMAGE, report

    async def local_reader(source: str, _timeout: float) -> bytes:
        if Path(source).expanduser().resolve() != loaded.path:
            raise ValueError("only the validated local image is accepted")
        return loaded.content

    try:
        _configure_local_runtime_state()
        factory = runtime_factory or (
            lambda current: YoloRuntime(
                current.detection_model_path,
                current.detection_confidence,
                current.detection_iou,
            )
        )
        runtime = RecordingYoloRuntime(factory(settings))
        recorder = RecordingLandingAnalyzer(LandingAnalyzer(policy))
        service = YoloDetectionService(
            settings,
            runtime=runtime,
            class_mapper=YoloClassMapper(),
            image_reader=local_reader,
            landing_analyzer=recorder,
        )
        frame = FrameContext(
            frame_id="task1-landing-local-frame",
            image_url=str(loaded.path),
            video_name="task1-landing-validation",
            session_id="task1-landing-validation",
            gps_health_status=None,
            gps_x=None,
            gps_y=None,
            gps_z=None,
            frame_index=0,
            image_modality=ImageModality.RGB,
        )
        detections = await service.process_frame(frame)
    except Exception as exc:
        report["final_result"] = "FAIL"
        report["failure_reason"] = f"detection_failed:{type(exc).__name__}"
        _write_json(options.json_output, report)
        emit(f"Landing validation: FAIL ({type(exc).__name__})")
        emit("Prediction submission: DISABLED")
        return EXIT_MODEL, report

    mapping = model_mapping_report(runtime)
    report["model_info"].update(
        {
            "load_attempted": runtime.load_attempted,
            "load_status": "OK" if runtime.model is not None else "FAIL",
            **mapping,
        }
    )
    report["frame_metadata"] = {**loaded.metadata, "frame_id": frame.frame_id, "frame_index": 0}
    obstacle_detections = [item for item in detections if item.cls in {ObjectClass.TASIT, ObjectClass.INSAN}]
    obstacle_classes = tuple(item.cls for item in obstacle_detections)
    landing_calls = iter(recorder.calls)
    serialized = []
    diagnostics_by_index: dict[int, dict[str, object]] = {}
    for index, item in enumerate(detections):
        entry = _serialize_detection(index, item)
        if item.cls in {ObjectClass.UAP, ObjectClass.UAI}:
            if settings.detection_landing_enabled:
                call = next(landing_calls, None)
                if call is not None:
                    diagnostic = landing_diagnostics(call, policy, obstacle_classes)
                else:
                    diagnostic = {"decision_reason": "landing_analyzer_call_missing", "final_landing_status": item.landing_status.value}
            else:
                diagnostic = {"decision_reason": "landing_disabled_by_config", "final_landing_status": item.landing_status.value}
            entry["landing_diagnostics"] = diagnostic
            diagnostics_by_index[index] = diagnostic
        serialized.append(entry)
    report["detections"] = serialized
    uap_uai_count = sum(item.cls in {ObjectClass.UAP, ObjectClass.UAI} for item in detections)
    report["uap_uai_detection_count"] = uap_uai_count
    report["final_result"] = "PASS" if runtime.model is not None else "FAIL"
    report["failure_reason"] = None if runtime.model is not None else "model_load_failed"

    _emit_real_results(report, emit)
    if options.save_visualization is not None:
        save_visualization(options.save_visualization, loaded.image, detections, diagnostics_by_index)
        report["visualization"] = str(options.save_visualization.expanduser().resolve())
    _write_json(options.json_output, report)
    emit("Prediction submission: DISABLED")
    return (EXIT_OK if runtime.model is not None else EXIT_MODEL), report


def run_synthetic_validation(
    settings: Settings,
    options: ValidationOptions,
    *,
    emit: Callable[[str], None] = print,
) -> tuple[int, dict[str, object]]:
    policy = policy_from_settings(settings)
    config = landing_config_report(settings)
    _emit_config(config, emit)
    analyzer = LandingAnalyzer(policy)
    margin = max(policy.edge_margin_px, 100 * policy.edge_margin_ratio)
    scenarios = [
        _synthetic_case(analyzer, policy, "clear_uap", ObjectClass.UAP, (10, 10, 60, 60), (10, 10, 60, 60), [], [], LandingStatus.SUITABLE),
        _synthetic_case(analyzer, policy, "vehicle_obstacle", ObjectClass.UAP, (10, 10, 60, 60), (10, 10, 60, 60), [(20, 20, 35, 35)], [ObjectClass.TASIT], LandingStatus.UNSUITABLE),
        _synthetic_case(analyzer, policy, "person_center", ObjectClass.UAP, (10, 10, 60, 60), (10, 10, 60, 60), [(50, 50, 65, 65)], [ObjectClass.INSAN], LandingStatus.UNSUITABLE),
        _synthetic_case(analyzer, policy, "outside_frame_tolerance", ObjectClass.UAP, (-margin - 1, 10, 40, 50), (0, 10, 40, 50), [], [], LandingStatus.UNSUITABLE),
        _synthetic_case(analyzer, policy, "bbox_too_small", ObjectClass.UAP, (10, 10, 14, 14), (10, 10, 14, 14), [], [], LandingStatus.NOT_APPLICABLE),
        _synthetic_case(analyzer, policy, "invalid_bbox", ObjectClass.UAP, (10, 10, 10, 20), (10, 10, 10, 20), [], [], LandingStatus.NOT_APPLICABLE),
        _not_applicable_case("vehicle_not_applicable", ObjectClass.TASIT),
        _not_applicable_case("person_not_applicable", ObjectClass.INSAN),
        _synthetic_case(analyzer, policy, "uap_ignores_uai", ObjectClass.UAP, (10, 10, 60, 60), (10, 10, 60, 60), [], [], LandingStatus.SUITABLE, ignored_non_obstacle=ObjectClass.UAI),
        _synthetic_case(analyzer, policy, "uai_ignores_uap", ObjectClass.UAI, (20, 20, 70, 70), (20, 20, 70, 70), [], [], LandingStatus.SUITABLE, ignored_non_obstacle=ObjectClass.UAP),
    ]
    passed = all(bool(item["passed"]) for item in scenarios)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "synthetic",
        "landing_config": config,
        "frame_metadata": {"width": 100, "height": 100, "synthetic": True},
        "scenarios": scenarios,
        "summary": {"passed": sum(bool(item["passed"]) for item in scenarios), "total": len(scenarios)},
        "prediction_submission": "DISABLED",
        "final_result": "PASS" if passed else "FAIL",
    }
    for item in scenarios:
        emit(
            f"{item['scenario']}: expected={item['expected']} actual={item['actual']} "
            f"reason={item['decision_reason']} {'PASS' if item['passed'] else 'FAIL'}"
        )
    if options.save_visualization is not None:
        save_synthetic_visualization(options.save_visualization, scenarios)
        report["visualization"] = str(options.save_visualization.expanduser().resolve())
    _write_json(options.json_output, report)
    emit("Prediction submission: DISABLED")
    return (EXIT_OK if passed else EXIT_CONFIG), report


def _synthetic_case(analyzer, policy, name, cls, raw, clipped, obstacles, obstacle_classes, expected, ignored_non_obstacle=None):
    result = analyzer.analyze(
        raw_bbox=raw,
        clipped_bbox=clipped,
        frame_width=100,
        frame_height=100,
        obstacles=obstacles,
    )
    call = LandingCall(tuple(raw), tuple(clipped), 100, 100, tuple(obstacles), result)
    diagnostic = landing_diagnostics(call, policy, tuple(obstacle_classes))
    return {
        "scenario": name,
        "class": cls.value,
        "expected": expected.value,
        "actual": result.value,
        "passed": result is expected,
        "decision_reason": diagnostic["decision_reason"],
        "ignored_non_obstacle_class": ignored_non_obstacle.value if ignored_non_obstacle else None,
        "landing_diagnostics": diagnostic,
    }


def _not_applicable_case(name, cls):
    return {
        "scenario": name,
        "class": cls.value,
        "expected": LandingStatus.NOT_APPLICABLE.value,
        "actual": LandingStatus.NOT_APPLICABLE.value,
        "passed": True,
        "decision_reason": "class_not_applicable",
        "landing_diagnostics": None,
    }


def save_visualization(path, image, detections, diagnostics_by_index):
    import cv2

    canvas = image.copy()
    colors = {
        ObjectClass.TASIT: (0, 255, 0),
        ObjectClass.INSAN: (255, 255, 0),
        ObjectClass.UAP: (0, 165, 255),
        ObjectClass.UAI: (255, 0, 255),
    }
    for index, item in enumerate(detections):
        color = colors[item.cls]
        p1 = (round(item.top_left_x), round(item.top_left_y))
        p2 = (round(item.bottom_right_x), round(item.bottom_right_y))
        cv2.rectangle(canvas, p1, p2, color, 2)
        reason = diagnostics_by_index.get(index, {}).get("decision_reason", "")
        confidence = "n/a" if item.confidence is None else f"{item.confidence:.2f}"
        label = f"{item.cls.value} {confidence} {item.landing_status.value} {reason}".strip()
        cv2.putText(canvas, label, (p1[0], max(18, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        for obstacle in diagnostics_by_index.get(index, {}).get("obstacles", []):
            if not obstacle.get("occupied_by_production_policy"):
                continue
            intersection = _intersection_bbox(_bbox_tuple(diagnostics_by_index[index]["clipped_bbox"]), _bbox_tuple(obstacle["obstacle_bbox"]))
            if intersection is not None:
                cv2.rectangle(canvas, (round(intersection[0]), round(intersection[1])), (round(intersection[2]), round(intersection[3])), (0, 0, 255), -1)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), canvas):
        raise OSError("visualization could not be saved")


def save_synthetic_visualization(path, scenarios):
    import cv2

    canvas = np.full((max(240, len(scenarios) * 32), 900, 3), 255, np.uint8)
    for index, item in enumerate(scenarios):
        color = (0, 128, 0) if item["passed"] else (0, 0, 255)
        text = f"{item['scenario']}: {item['actual']} ({item['decision_reason']})"
        cv2.putText(canvas, text, (10, 28 + index * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), canvas):
        raise OSError("synthetic visualization could not be saved")


def _serialize_detection(index: int, item: DetectedObject) -> dict[str, object]:
    return {
        "detection_index": index,
        "class": item.cls.value,
        "confidence": item.confidence,
        "bbox": _bbox_dict((item.top_left_x, item.top_left_y, item.bottom_right_x, item.bottom_right_y)),
        "motion_status": item.motion_status.value,
        "landing_status": item.landing_status.value,
    }


def _emit_config(config, emit):
    emit("Landing production config:")
    for key, value in config.items():
        emit(f"  {key}={value}")


def _emit_real_results(report, emit):
    emit(f"Model path: {report['model_info']['model_path']}")
    emit(f"Model exists: {report['model_info']['model_exists']}")
    emit(f"Model load status: {report['model_info']['load_status']}")
    emit(f"Detections: {len(report['detections'])}; UAP/UAI: {report['uap_uai_detection_count']}")
    for item in report["detections"]:
        bbox = item["bbox"]
        emit(
            f"  index={item['detection_index']} class={item['class']} confidence={item['confidence']} "
            f"bbox=({bbox['top_left_x']}, {bbox['top_left_y']}, {bbox['bottom_right_x']}, {bbox['bottom_right_y']}) "
            f"landing_status={item['landing_status']}"
        )
        if "landing_diagnostics" in item:
            diagnostic = item["landing_diagnostics"]
            emit(
                f"    reason={diagnostic['decision_reason']} visible_ratio={diagnostic.get('visible_ratio', 'n/a')} "
                f"obstacles={diagnostic.get('obstacle_count', 'n/a')}"
            )


def _bbox_area(bbox):
    if len(bbox) != 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_dict(bbox):
    values = tuple(bbox)
    return {
        "top_left_x": values[0],
        "top_left_y": values[1],
        "bottom_right_x": values[2],
        "bottom_right_y": values[3],
    }


def _bbox_tuple(value):
    return (value["top_left_x"], value["top_left_y"], value["bottom_right_x"], value["bottom_right_y"])


def _intersection_bbox(first, second):
    bbox = (max(first[0], second[0]), max(first[1], second[1]), min(first[2], second[2]), min(first[3], second[3]))
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def _write_json(path, report):
    if path is None:
        return
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


SequenceObjectClass = tuple[ObjectClass, ...] | list[ObjectClass]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = ValidationOptions(args.image, args.synthetic, args.save_visualization, args.json_output)
    if not options.synthetic and options.image is None:
        print("--image or --synthetic is required")
        print("Prediction submission: DISABLED")
        return EXIT_CONFIG
    settings = get_settings()
    if options.synthetic:
        code, _ = run_synthetic_validation(settings, options)
    else:
        code, _ = asyncio.run(run_real_validation(settings, options))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
