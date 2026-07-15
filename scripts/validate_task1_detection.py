from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings, get_settings
from app.schemas import DetectedObject, ImageModality
from app.services.common import FrameContext
from app.services.detection.class_mapping import DEFAULT_YOLO_CLASS_MAPPING
from app.services.detection.interface import DetectionService
from app.services.detection.service import YoloDetectionService
from app.services.detection.yolo_runtime import YoloRuntime
from app.utils.images import detect_image_format

EXIT_OK = 0
EXIT_CONFIG = 10
EXIT_IMAGE = 20
EXIT_MODEL = 30


@dataclass(frozen=True, slots=True)
class ValidationOptions:
    image: Path
    next_image: Path | None = None
    save_visualization: Path | None = None
    json_output: Path | None = None
    session_id: str = "task1-local-validation"
    video_name: str = "task1-local-video"


@dataclass(frozen=True, slots=True)
class LoadedImage:
    path: Path
    content: bytes
    image: object
    metadata: dict[str, object]


RuntimeFactory = Callable[[Settings], object]
ImageLoader = Callable[[Path], LoadedImage]
ImageReader = Callable[[str, float], Awaitable[bytes]]
ServiceFactory = Callable[[Settings, object, ImageReader], DetectionService]


class RecordingYoloRuntime:
    """Transparent instrumentation around the single production YoloRuntime."""

    def __init__(self, runtime: object) -> None:
        self._runtime = runtime
        self.model_path = getattr(runtime, "model_path", None)
        self.confidence = getattr(runtime, "confidence")
        self.iou = getattr(runtime, "iou", None)
        self.detected_class_ids_by_call: list[list[int]] = []

    def predict(self, image: object) -> list[object]:
        results = list(self._runtime.predict(image) or [])
        self.detected_class_ids_by_call.append(_raw_class_ids(results))
        return results

    @property
    def model(self) -> object | None:
        return getattr(self._runtime, "_model", None)

    @property
    def load_attempted(self) -> bool:
        return bool(getattr(self._runtime, "_load_attempted", self.detected_class_ids_by_call))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production Task 1 DetectionService ile guvenli yerel goruntu dogrulamasi."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--next-image", type=Path)
    parser.add_argument("--save-visualization", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser


def load_local_image(path: Path) -> LoadedImage:
    import cv2
    import numpy as np

    resolved = path.expanduser().resolve()
    content = resolved.read_bytes()
    image_format = detect_image_format(content)
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("OpenCV goruntuyu decode edemedi.")
    height, width = image.shape[:2]
    return LoadedImage(
        path=resolved,
        content=content,
        image=image,
        metadata={
            "path": str(resolved),
            "format": image_format,
            "width": int(width),
            "height": int(height),
            "channels": int(image.shape[2]) if image.ndim == 3 else 1,
            "sha256_short": hashlib.sha256(content).hexdigest()[:12],
        },
    )


def _default_runtime_factory(settings: Settings) -> YoloRuntime:
    return YoloRuntime(
        settings.detection_model_path,
        settings.detection_confidence,
        settings.detection_iou,
    )


def _default_service_factory(
    settings: Settings,
    runtime: object,
    image_reader: ImageReader,
) -> DetectionService:
    return YoloDetectionService(settings, runtime=runtime, image_reader=image_reader)


def _configure_local_runtime_state() -> dict[str, str]:
    """Keep optional third-party caches local without changing inference behavior."""
    base = Path(__file__).resolve().parents[1] / "work" / "task1_validation"
    yolo_config = base / "ultralytics"
    matplotlib_config = base / "matplotlib"
    yolo_config.mkdir(parents=True, exist_ok=True)
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    return {
        "yolo_config_dir": os.environ["YOLO_CONFIG_DIR"],
        "matplotlib_config_dir": os.environ["MPLCONFIGDIR"],
    }


def _scalar(value: object) -> object:
    candidate = value
    if hasattr(candidate, "tolist"):
        candidate = candidate.tolist()
    while isinstance(candidate, (list, tuple)) and candidate:
        candidate = candidate[0]
    return candidate


def _raw_class_ids(results: list[object]) -> list[int]:
    values: list[int] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            try:
                numeric = float(_scalar(getattr(box, "cls")))
                if numeric.is_integer():
                    values.append(int(numeric))
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
    return values


def _model_names(model: object | None) -> dict[int, str]:
    if model is None:
        return {}
    names = getattr(model, "names", None)
    if names is None:
        names = getattr(getattr(model, "model", None), "names", None)
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def _canonical_class_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold()).replace("ı", "i")
    ascii_like = "".join(character for character in normalized if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]", "", ascii_like)


def model_mapping_report(runtime: RecordingYoloRuntime) -> dict[str, object]:
    names = _model_names(runtime.model)
    production = {class_id: object_class.value for class_id, object_class in DEFAULT_YOLO_CLASS_MAPPING.items()}
    comparisons = []
    for class_id, expected in production.items():
        actual = names.get(class_id)
        comparisons.append({
            "class_id": class_id,
            "model_name": actual,
            "production_class": expected,
            "matches": actual is not None
            and _canonical_class_name(actual) == _canonical_class_name(expected),
        })
    matches_expected = (
        set(names) == set(production)
        and all(item["matches"] for item in comparisons)
    )
    return {
        "model_class_names": {str(key): value for key, value in sorted(names.items())},
        "production_class_mapping": {str(key): value for key, value in sorted(production.items())},
        "comparison": comparisons,
        "matches_expected_mapping": matches_expected,
    }


def _serialize_detection(index: int, item: DetectedObject) -> dict[str, object]:
    return {
        "detection_index": index,
        "class": item.cls.value,
        "confidence": item.confidence,
        "bbox": {
            "top_left_x": item.top_left_x,
            "top_left_y": item.top_left_y,
            "bottom_right_x": item.bottom_right_x,
            "bottom_right_y": item.bottom_right_y,
        },
        "motion_status": item.motion_status.value,
        "landing_status": item.landing_status.value,
    }


def _frame_context(options: ValidationOptions, loaded: LoadedImage, index: int) -> FrameContext:
    return FrameContext(
        frame_id=f"task1-local-frame-{index}",
        image_url=str(loaded.path),
        video_name=options.video_name,
        session_id=options.session_id,
        gps_health_status=None,
        gps_x=None,
        gps_y=None,
        gps_z=None,
        frame_index=index,
        image_modality=ImageModality.RGB,
    )


def _visualization_paths(path: Path, frame_count: int) -> list[Path]:
    resolved = path.expanduser().resolve()
    if frame_count == 1:
        return [resolved]
    if resolved.is_dir() or not resolved.suffix:
        resolved.mkdir(parents=True, exist_ok=True)
        return [resolved / "task1_frame1.jpg", resolved / "task1_frame2.jpg"]
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return [
        resolved.with_name(f"{resolved.stem}_frame1{resolved.suffix}"),
        resolved.with_name(f"{resolved.stem}_frame2{resolved.suffix}"),
    ]


def save_visualization(path: Path, image: object, detections: list[DetectedObject]) -> None:
    import cv2

    canvas = image.copy()
    colors = {
        "tasit": (0, 255, 0),
        "insan": (255, 255, 0),
        "uap": (0, 165, 255),
        "uai": (255, 0, 255),
    }
    for item in detections:
        color = colors[item.cls.value]
        point1 = (round(item.top_left_x), round(item.top_left_y))
        point2 = (round(item.bottom_right_x), round(item.bottom_right_y))
        cv2.rectangle(canvas, point1, point2, color, 2)
        confidence = "n/a" if item.confidence is None else f"{item.confidence:.3f}"
        label = (
            f"{item.cls.value} {confidence} "
            f"motion={item.motion_status.value} landing={item.landing_status.value}"
        )
        text_y = max(16, point1[1] - 6)
        cv2.putText(canvas, label, (point1[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError("Gorsellestirme kaydedilemedi.")


async def run_validation(
    settings: Settings,
    options: ValidationOptions,
    *,
    runtime_factory: RuntimeFactory = _default_runtime_factory,
    service_factory: ServiceFactory = _default_service_factory,
    image_loader: ImageLoader = load_local_image,
    emit: Callable[[str], None] = print,
) -> tuple[int, dict[str, object]]:
    report: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_info": {
            "model_path": str(settings.detection_model_path) if settings.detection_model_path else None,
            "model_exists": bool(settings.detection_model_path and settings.detection_model_path.is_file()),
            "load_status": "NOT_ATTEMPTED",
            "confidence_threshold": settings.detection_confidence,
            "iou_threshold": settings.detection_iou,
        },
        "frames": [],
        "prediction_submission": "DISABLED",
        "final_result": "FAIL",
        "failure_reason": None,
    }
    emit("Task 1 local validation (production DetectionService)")
    emit(f"Model path: {report['model_info']['model_path']}")
    emit(f"Model exists: {report['model_info']['model_exists']}")
    try:
        loaded_images = [image_loader(options.image)]
        if options.next_image is not None:
            loaded_images.append(image_loader(options.next_image))
    except Exception as exc:
        report["failure_reason"] = f"image_load_failed:{type(exc).__name__}"
        emit(f"Image validation: FAIL ({type(exc).__name__})")
        emit("Prediction submission: DISABLED")
        _write_json(report, options.json_output)
        return EXIT_IMAGE, report

    content_by_path = {str(item.path): item.content for item in loaded_images}

    async def local_reader(source: str, _timeout: float) -> bytes:
        if source not in content_by_path:
            raise ValueError("Yalniz onceden dogrulanmis yerel goruntu yolu kabul edilir.")
        return content_by_path[source]

    try:
        report["local_runtime_state"] = _configure_local_runtime_state()
        inner_runtime = runtime_factory(settings)
        runtime = RecordingYoloRuntime(inner_runtime)
        service = service_factory(settings, runtime, local_reader)
    except Exception as exc:
        report["failure_reason"] = f"service_initialization_failed:{type(exc).__name__}"
        emit(f"Detection service initialization: FAIL ({type(exc).__name__})")
        emit("Prediction submission: DISABLED")
        _write_json(report, options.json_output)
        return EXIT_MODEL, report
    frame_outputs: list[list[DetectedObject]] = []
    try:
        for index, loaded in enumerate(loaded_images):
            detections = await service.process_frame(_frame_context(options, loaded, index))
            frame_outputs.append(detections)
            raw_ids = runtime.detected_class_ids_by_call[index] if index < len(runtime.detected_class_ids_by_call) else []
            frame_report = {
                **loaded.metadata,
                "frame_index": index,
                "frame_id": f"task1-local-frame-{index}",
                "session_id": options.session_id,
                "video_name": options.video_name,
                "detected_class_ids": raw_ids,
                "detections": [
                    _serialize_detection(detection_index, item)
                    for detection_index, item in enumerate(detections)
                ],
            }
            report["frames"].append(frame_report)
    except Exception as exc:
        report["failure_reason"] = f"detection_failed:{type(exc).__name__}"
        emit(f"Detection validation: FAIL ({type(exc).__name__})")
        emit("Prediction submission: DISABLED")
        _write_json(report, options.json_output)
        return EXIT_MODEL, report

    model_loaded = runtime.model is not None
    mapping = model_mapping_report(runtime)
    report["model_info"].update({
        "load_attempted": runtime.load_attempted,
        "load_status": "OK" if model_loaded else "FAIL",
        **mapping,
    })
    emit(f"Model load status: {report['model_info']['load_status']}")
    emit("Model class names:")
    for class_id, name in mapping["model_class_names"].items():
        emit(f"  {class_id} -> {name}")
    emit("Production class mapping:")
    for class_id, name in mapping["production_class_mapping"].items():
        emit(f"  {class_id} -> {name}")
    emit(
        "Class mapping compatibility: "
        + ("OK" if mapping["matches_expected_mapping"] else "WARNING")
    )
    for frame_report in report["frames"]:
        emit(
            f"Frame {frame_report['frame_index']}: detected_class_ids="
            f"{frame_report['detected_class_ids']}; detections={len(frame_report['detections'])}"
        )
        for item in frame_report["detections"]:
            bbox = item["bbox"]
            emit(
                f"  detection={item['detection_index']} class={item['class']} "
                f"confidence={item['confidence']} bbox=({bbox['top_left_x']}, {bbox['top_left_y']}, "
                f"{bbox['bottom_right_x']}, {bbox['bottom_right_y']}) "
                f"motion={item['motion_status']} landing={item['landing_status']}"
            )

    if options.save_visualization is not None:
        try:
            targets = _visualization_paths(options.save_visualization, len(loaded_images))
            for target, loaded, detections in zip(targets, loaded_images, frame_outputs, strict=True):
                save_visualization(target, loaded.image, detections)
            report["visualizations"] = [str(target) for target in targets]
        except Exception as exc:
            report["failure_reason"] = f"visualization_failed:{type(exc).__name__}"
            emit(f"Visualization: FAIL ({type(exc).__name__})")
            emit("Prediction submission: DISABLED")
            _write_json(report, options.json_output)
            return EXIT_IMAGE, report

    report["final_result"] = "PASS" if model_loaded else "FAIL"
    report["failure_reason"] = None if model_loaded else "model_load_failed"
    emit("Prediction submission: DISABLED")
    _write_json(report, options.json_output)
    return (EXIT_OK if model_loaded else EXIT_MODEL), report


def _write_json(report: dict[str, object], path: Path | None) -> None:
    if path is None:
        return
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = ValidationOptions(
        image=args.image,
        next_image=args.next_image,
        save_visualization=args.save_visualization,
        json_output=args.json_output,
    )
    code, _ = asyncio.run(run_validation(get_settings(), options))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
