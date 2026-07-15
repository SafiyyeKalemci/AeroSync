from __future__ import annotations

import argparse
import asyncio
import ast
import csv
import math
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings, get_settings
from app.schemas import DetectedObject, MotionStatus
from scripts.benchmark_task1_motion import (
    METHODS,
    FrameFile,
    FramePair,
    OfflineMotionPairProcessor,
    PairAnalysis,
    build_frame_pairs,
    discover_frames,
    pairing_diagnostics,
)
from scripts.validate_task1_detection import _configure_local_runtime_state

MOVING = MotionStatus.MOVING.value
STATIONARY = MotionStatus.STATIONARY.value
UNKNOWN = MotionStatus.UNKNOWN.value
IGNORE = "ignore"

MOTION_ALIASES = {
    "moving": MOVING,
    "hareketli": MOVING,
    "motion": MOVING,
    "stationary": STATIONARY,
    "hareketsiz": STATIONARY,
    "sabit": STATIONARY,
    "durgun": STATIONARY,
    "ignore": IGNORE,
    "ignored": IGNORE,
    "yoksay": IGNORE,
}

DETAILED_COLUMNS = (
    "previous_frame",
    "current_frame",
    "gt_object_index",
    "gt_class",
    "gt_motion",
    "gt_bbox",
    "matched_detection_index",
    "detection_bbox",
    "detection_confidence",
    "iou",
    "global_median_prediction",
    "homography_prediction",
    "homography_bbox_prediction",
    "homography_hybrid_prediction",
    "homography_local_prediction",
    "homography_adaptive_prediction",
    "adaptive_selected_method",
    "adaptive_scene_quality",
    "adaptive_selection_reason",
    "adaptive_homography_quality",
    "adaptive_background_residual_median",
    "adaptive_background_residual_p90",
    "adaptive_background_residual_p95",
    "adaptive_background_grid_spread",
    "adaptive_background_spatial_variance",
    "adaptive_valid_background_ratio",
    "homography_valid",
    "homography_quality_level",
    "homography_inlier_ratio",
    "bbox_iou",
    "bbox_center_residual",
    "flow_residual",
    "local_corrected_residual",
    "ignored",
    "matched",
)

SUMMARY_COLUMNS = (
    "method",
    "total_ground_truth_objects",
    "matched_ground_truth_objects",
    "unmatched_ground_truth_objects",
    "ignored_ground_truth_objects",
    "total_evaluated",
    "moving_gt",
    "stationary_gt",
    "moving_pred",
    "stationary_pred",
    "unknown_pred",
    "strict_accuracy",
    "decided_only_accuracy",
    "coverage",
    "moving_precision",
    "moving_recall",
    "moving_f1",
    "stationary_precision",
    "stationary_recall",
    "stationary_f1",
    "macro_f1",
    "false_moving",
    "false_stationary",
    "balanced_score",
)

CONFUSION_COLUMNS = ("method", "gt_motion", "predicted_motion", "count")
UNMATCHED_GT_COLUMNS = (
    "previous_frame",
    "current_frame",
    "gt_object_index",
    "gt_motion",
    "gt_bbox",
    "ignored",
    "best_iou",
)
UNMATCHED_DETECTION_COLUMNS = (
    "previous_frame",
    "current_frame",
    "detection_index",
    "detection_bbox",
    "detection_confidence",
    "best_iou",
)
VALIDATION_COLUMNS = ("file", "error")


@dataclass(frozen=True, slots=True)
class EvaluationOptions:
    images_dir: Path
    labels_dir: Path
    output_dir: Path
    min_iou: float = 0.5
    save_visualizations: bool = False


@dataclass(frozen=True, slots=True)
class GroundTruthObject:
    index: int
    motion: str
    bbox: tuple[float, float, float, float]

    @property
    def ignored(self) -> bool:
        return self.motion == IGNORE


@dataclass(frozen=True, slots=True)
class Annotation:
    source: Path
    image_filename: str
    width: int
    height: int
    objects: tuple[GroundTruthObject, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Association:
    gt_to_detection: dict[int, tuple[int, float]]
    unmatched_gt: tuple[int, ...]
    unmatched_detections: tuple[int, ...]
    best_gt_iou: dict[int, float]
    best_detection_iou: dict[int, float]


PairProcessor = Callable[[FramePair], Awaitable[PairAnalysis]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XML ground truth ile beş Task 1 motion yöntemini offline değerlendir."
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--save-visualizations", action="store_true")
    return parser


def normalize_motion_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().casefold().replace("_", " ").split())
    compact = normalized.replace(" ", "")
    return MOTION_ALIASES.get(normalized) or MOTION_ALIASES.get(compact)


def parse_annotation(xml_path: Path, image_path: Path | None = None) -> Annotation:
    errors: list[str] = []
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        return Annotation(xml_path, "", 0, 0, (), (f"malformed_xml: {exc}",))
    filename = (root.findtext("filename") or "").strip()
    width = _integer(root.findtext("size/width"))
    height = _integer(root.findtext("size/height"))
    if not filename:
        errors.append("missing_filename")
    if filename and Path(filename).stem.casefold() != xml_path.stem.casefold():
        errors.append("image_basename_mismatch")
    if image_path is None or not image_path.is_file():
        errors.append("missing_image")
    elif image_path.stem.casefold() != xml_path.stem.casefold():
        errors.append("image_basename_mismatch")
    if width <= 0 or height <= 0:
        errors.append("invalid_image_size")

    objects: list[GroundTruthObject] = []
    seen: set[tuple[str, tuple[float, float, float, float]]] = set()
    for index, node in enumerate(root.findall("object")):
        motion = _motion_from_object(node)
        if motion is None:
            errors.append(f"object[{index}]: invalid_motion_label")
            continue
        bbox = _bbox_from_object(node)
        if bbox is None:
            errors.append(f"object[{index}]: invalid_bbox")
            continue
        if width > 0 and height > 0 and not _bbox_within(bbox, width, height):
            errors.append(f"object[{index}]: bbox_out_of_bounds")
            continue
        key = motion, bbox
        if key in seen:
            errors.append(f"object[{index}]: duplicate_object")
            continue
        seen.add(key)
        objects.append(GroundTruthObject(index, motion, bbox))
    return Annotation(xml_path, filename, width, height, tuple(objects), tuple(errors))


def validate_dataset(
    images_dir: Path, labels_dir: Path
) -> tuple[dict[str, Annotation], list[dict[str, str]]]:
    image_by_stem = {
        item.stem.casefold(): item
        for item in images_dir.iterdir()
        if item.is_file() and item.suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp"}
    }
    annotations: dict[str, Annotation] = {}
    issues: list[dict[str, str]] = []
    for xml_path in sorted(labels_dir.glob("*.xml")):
        annotation = parse_annotation(
            xml_path, image_by_stem.get(xml_path.stem.casefold())
        )
        annotations[xml_path.stem.casefold()] = annotation
        issues.extend({"file": str(xml_path), "error": error} for error in annotation.errors)
    for stem, image_path in image_by_stem.items():
        if stem not in annotations:
            issues.append({"file": str(image_path), "error": "missing_xml"})
    return annotations, issues


def associate_objects(
    ground_truth: Sequence[GroundTruthObject],
    detections: Sequence[DetectedObject],
    min_iou: float,
) -> Association:
    candidates: list[tuple[float, int, int]] = []
    best_gt = {index: 0.0 for index in range(len(ground_truth))}
    best_detection = {index: 0.0 for index in range(len(detections))}
    for gt_position, gt in enumerate(ground_truth):
        for detection_index, detection in enumerate(detections):
            score = bbox_iou(gt.bbox, _detection_bbox(detection))
            best_gt[gt_position] = max(best_gt[gt_position], score)
            best_detection[detection_index] = max(best_detection[detection_index], score)
            if score >= min_iou:
                candidates.append((score, gt_position, detection_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_gt: set[int] = set()
    used_detection: set[int] = set()
    matches: dict[int, tuple[int, float]] = {}
    for score, gt_position, detection_index in candidates:
        if gt_position in used_gt or detection_index in used_detection:
            continue
        used_gt.add(gt_position)
        used_detection.add(detection_index)
        matches[gt_position] = detection_index, score
    return Association(
        gt_to_detection=matches,
        unmatched_gt=tuple(index for index in range(len(ground_truth)) if index not in used_gt),
        unmatched_detections=tuple(
            index for index in range(len(detections)) if index not in used_detection
        ),
        best_gt_iou=best_gt,
        best_detection_iou=best_detection,
    )


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def calculate_metrics(
    detailed_rows: Sequence[dict[str, object]],
    *,
    total_gt: int,
    matched_gt: int,
    unmatched_gt: int,
    ignored_gt: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    evaluated = [
        row for row in detailed_rows if row["matched"] is True and row["ignored"] is False
    ]
    summaries: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for method in METHODS:
        prediction_key = f"{method}_prediction"
        confusion = {
            (gt, prediction): sum(
                    row["gt_motion"] == gt
                    and row.get(prediction_key, UNKNOWN) == prediction
                for row in evaluated
            )
            for gt in (MOVING, STATIONARY)
            for prediction in (MOVING, STATIONARY, UNKNOWN)
        }
        confusion_rows.extend(
            {
                "method": method,
                "gt_motion": gt,
                "predicted_motion": prediction,
                "count": confusion[(gt, prediction)],
            }
            for gt in (MOVING, STATIONARY)
            for prediction in (MOVING, STATIONARY, UNKNOWN)
        )
        total = len(evaluated)
        correct = confusion[(MOVING, MOVING)] + confusion[(STATIONARY, STATIONARY)]
        moving_pred = sum(confusion[(gt, MOVING)] for gt in (MOVING, STATIONARY))
        stationary_pred = sum(
            confusion[(gt, STATIONARY)] for gt in (MOVING, STATIONARY)
        )
        unknown_pred = sum(confusion[(gt, UNKNOWN)] for gt in (MOVING, STATIONARY))
        decided = moving_pred + stationary_pred
        moving_gt = sum(confusion[(MOVING, pred)] for pred in (MOVING, STATIONARY, UNKNOWN))
        stationary_gt = sum(
            confusion[(STATIONARY, pred)] for pred in (MOVING, STATIONARY, UNKNOWN)
        )
        moving_precision = _divide(confusion[(MOVING, MOVING)], moving_pred)
        moving_recall = _divide(confusion[(MOVING, MOVING)], moving_gt)
        stationary_precision = _divide(
            confusion[(STATIONARY, STATIONARY)], stationary_pred
        )
        stationary_recall = _divide(
            confusion[(STATIONARY, STATIONARY)], stationary_gt
        )
        strict = _divide(correct, total)
        coverage = _divide(decided, total)
        moving_f1 = _f1(moving_precision, moving_recall)
        stationary_f1 = _f1(stationary_precision, stationary_recall)
        summaries.append(
            {
                "method": method,
                "total_ground_truth_objects": total_gt,
                "matched_ground_truth_objects": matched_gt,
                "unmatched_ground_truth_objects": unmatched_gt,
                "ignored_ground_truth_objects": ignored_gt,
                "total_evaluated": total,
                "moving_gt": moving_gt,
                "stationary_gt": stationary_gt,
                "moving_pred": moving_pred,
                "stationary_pred": stationary_pred,
                "unknown_pred": unknown_pred,
                "strict_accuracy": strict,
                "decided_only_accuracy": _divide(correct, decided),
                "coverage": coverage,
                "moving_precision": moving_precision,
                "moving_recall": moving_recall,
                "moving_f1": moving_f1,
                "stationary_precision": stationary_precision,
                "stationary_recall": stationary_recall,
                "stationary_f1": stationary_f1,
                "macro_f1": (moving_f1 + stationary_f1) / 2,
                "false_moving": confusion[(STATIONARY, MOVING)],
                "false_stationary": confusion[(MOVING, STATIONARY)],
                "balanced_score": strict * coverage,
            }
        )
    return summaries, confusion_rows


async def run_evaluation(
    settings: Settings,
    options: EvaluationOptions,
    *,
    processor: PairProcessor | None = None,
    emit: Callable[[str], None] = print,
) -> dict[str, object]:
    if not math.isfinite(options.min_iou) or not 0 <= options.min_iou <= 1:
        raise ValueError("--min-iou [0, 1] aralığında olmalıdır")
    images_dir = options.images_dir.expanduser().resolve()
    labels_dir = options.labels_dir.expanduser().resolve()
    output_dir = options.output_dir.expanduser().resolve()
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError("images-dir ve labels-dir mevcut klasörler olmalıdır")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = discover_frames(images_dir)
    pairs = build_frame_pairs(frames)
    annotations, validation_issues = validate_dataset(images_dir, labels_dir)
    if processor is None:
        _configure_local_runtime_state()
        processor = OfflineMotionPairProcessor(settings, images_dir)

    all_objects = [obj for annotation in annotations.values() for obj in annotation.objects]
    emit("XML format: Pascal VOC; motion=object/name; bbox=object/bndbox")
    emit(
        f"Dataset: frames={len(frames)} xml={len(annotations)} pairs={len(pairs)} "
        f"objects={len(all_objects)} moving={sum(o.motion == MOVING for o in all_objects)} "
        f"stationary={sum(o.motion == STATIONARY for o in all_objects)} "
        f"ignore={sum(o.ignored for o in all_objects)}"
    )
    for message in pairing_diagnostics(frames):
        emit(message)

    detailed: list[dict[str, object]] = []
    unmatched_gt_rows: list[dict[str, object]] = []
    unmatched_detection_rows: list[dict[str, object]] = []
    pair_errors: list[str] = []
    current_gt_total = 0
    current_ignored = 0
    current_matched_nonignored = 0
    current_unmatched_nonignored = 0
    processed_pairs = 0
    for pair in pairs:
        annotation = annotations.get(pair.current.path.stem.casefold())
        if annotation is None:
            pair_errors.append(f"{pair.current.path.name}: missing_xml")
            continue
        try:
            analysis = await processor(pair)
        except Exception as exc:
            error = (
                f"{pair.previous.path.name}->{pair.current.path.name}: "
                f"{type(exc).__name__}: {exc}"
            )
            pair_errors.append(error)
            emit(f"PAIR ERROR {error}")
            continue
        processed_pairs += 1
        vehicles = list(analysis.vehicles)
        association = associate_objects(annotation.objects, vehicles, options.min_iou)
        row_by_detection = {
            int(row["vehicle_index"]): row for row in analysis.rows
        }
        current_gt_total += len(annotation.objects)
        current_ignored += sum(item.ignored for item in annotation.objects)
        for gt_position, gt in enumerate(annotation.objects):
            match = association.gt_to_detection.get(gt_position)
            detection_index = match[0] if match else None
            score = match[1] if match else None
            detection = vehicles[detection_index] if detection_index is not None else None
            prediction_row = row_by_detection.get(detection_index, {})
            matched = detection is not None
            if not gt.ignored:
                current_matched_nonignored += int(matched)
                current_unmatched_nonignored += int(not matched)
            row = _detailed_row(
                pair,
                gt,
                detection_index,
                detection,
                score,
                prediction_row,
            )
            detailed.append(row)
            if not matched:
                unmatched_gt_rows.append(
                    {
                        "previous_frame": pair.previous.path.name,
                        "current_frame": pair.current.path.name,
                        "gt_object_index": gt.index,
                        "gt_motion": gt.motion,
                        "gt_bbox": _bbox_text(gt.bbox),
                        "ignored": gt.ignored,
                        "best_iou": association.best_gt_iou[gt_position],
                    }
                )
        for detection_index in association.unmatched_detections:
            detection = vehicles[detection_index]
            unmatched_detection_rows.append(
                {
                    "previous_frame": pair.previous.path.name,
                    "current_frame": pair.current.path.name,
                    "detection_index": detection_index,
                    "detection_bbox": _bbox_text(_detection_bbox(detection)),
                    "detection_confidence": detection.confidence,
                    "best_iou": association.best_detection_iou[detection_index],
                }
            )
        if options.save_visualizations and analysis.current_image is not None:
            _save_visualizations(output_dir, pair, analysis.current_image, detailed[-len(annotation.objects):])

    if pairs and processed_pairs == 0:
        raise RuntimeError("Hiçbir frame çifti değerlendirilemedi")
    summaries, confusion = calculate_metrics(
        detailed,
        total_gt=current_gt_total,
        matched_gt=current_matched_nonignored,
        unmatched_gt=current_unmatched_nonignored,
        ignored_gt=current_ignored,
    )
    adaptive_selector_pairs = {
        method: sorted({
            f"{row['previous_frame']}->{row['current_frame']}"
            for row in detailed
            if row.get("adaptive_selected_method") == method
        })
        for method in ("homography", "homography_bbox", "unknown")
    }
    paths = {
        "detailed_csv": output_dir / "motion_evaluation_detailed.csv",
        "summary_csv": output_dir / "motion_evaluation_summary.csv",
        "confusion_csv": output_dir / "motion_evaluation_confusion.csv",
        "unmatched_gt_csv": output_dir / "unmatched_ground_truth.csv",
        "unmatched_detection_csv": output_dir / "unmatched_detections.csv",
        "validation_csv": output_dir / "xml_validation_errors.csv",
    }
    _write_csv(paths["detailed_csv"], DETAILED_COLUMNS, detailed)
    _write_csv(paths["summary_csv"], SUMMARY_COLUMNS, summaries)
    _write_csv(paths["confusion_csv"], CONFUSION_COLUMNS, confusion)
    _write_csv(paths["unmatched_gt_csv"], UNMATCHED_GT_COLUMNS, unmatched_gt_rows)
    _write_csv(
        paths["unmatched_detection_csv"],
        UNMATCHED_DETECTION_COLUMNS,
        unmatched_detection_rows,
    )
    _write_csv(paths["validation_csv"], VALIDATION_COLUMNS, validation_issues)
    _emit_summary(emit, summaries, current_matched_nonignored, current_unmatched_nonignored, unmatched_detection_rows)
    for method, selected_pairs in adaptive_selector_pairs.items():
        emit(f"Adaptive selected {method}: {len(selected_pairs)} frame pair(s)")
        for selected_pair in selected_pairs:
            emit(f"  {selected_pair}")
    emit("Balanced score formula: strict_accuracy * coverage")
    emit("Prediction submission: DISABLED")
    return {
        "frames": len(frames),
        "pairs": len(pairs),
        "processed_pairs": processed_pairs,
        "dataset_objects": len(all_objects),
        "dataset_moving": sum(item.motion == MOVING for item in all_objects),
        "dataset_stationary": sum(item.motion == STATIONARY for item in all_objects),
        "dataset_ignore": sum(item.ignored for item in all_objects),
        "current_gt_objects": current_gt_total,
        "matched_ground_truth": current_matched_nonignored,
        "unmatched_ground_truth": current_unmatched_nonignored,
        "unmatched_detections": len(unmatched_detection_rows),
        "validation_issues": validation_issues,
        "pair_errors": pair_errors,
        "summary": summaries,
        "adaptive_selector_pairs": adaptive_selector_pairs,
        **paths,
    }


def _detailed_row(
    pair: FramePair,
    gt: GroundTruthObject,
    detection_index: int | None,
    detection: DetectedObject | None,
    score: float | None,
    prediction: dict[str, object],
) -> dict[str, object]:
    return {
        "previous_frame": pair.previous.path.name,
        "current_frame": pair.current.path.name,
        "gt_object_index": gt.index,
        "gt_class": "tasit",
        "gt_motion": gt.motion,
        "gt_bbox": _bbox_text(gt.bbox),
        "matched_detection_index": _optional(detection_index),
        "detection_bbox": _bbox_text(_detection_bbox(detection)) if detection else "",
        "detection_confidence": detection.confidence if detection else "",
        "iou": _optional(score),
        **{
            f"{method}_prediction": prediction.get(f"{method}_result", "")
            for method in METHODS
        },
        "homography_valid": prediction.get("homography_valid", ""),
        "homography_quality_level": prediction.get(
            "hybrid_homography_quality_level", ""
        ),
        "homography_inlier_ratio": prediction.get("homography_inlier_ratio", ""),
        "adaptive_selected_method": prediction.get("adaptive_selected_method", ""),
        "adaptive_scene_quality": prediction.get("adaptive_scene_quality", ""),
        "adaptive_selection_reason": prediction.get("adaptive_selection_reason", ""),
        "adaptive_homography_quality": prediction.get("adaptive_homography_quality", ""),
        "adaptive_background_residual_median": prediction.get("adaptive_background_residual_median", ""),
        "adaptive_background_residual_p90": prediction.get("adaptive_background_residual_p90", ""),
        "adaptive_background_residual_p95": prediction.get("adaptive_background_residual_p95", ""),
        "adaptive_background_grid_spread": prediction.get("adaptive_background_grid_spread", ""),
        "adaptive_background_spatial_variance": prediction.get("adaptive_background_spatial_variance", ""),
        "adaptive_valid_background_ratio": prediction.get("adaptive_valid_background_ratio", ""),
        "bbox_iou": prediction.get("bbox_iou", ""),
        "bbox_center_residual": prediction.get("bbox_center_residual_px", ""),
        "flow_residual": prediction.get("homography_residual_px", ""),
        "local_corrected_residual": prediction.get(
            "local_corrected_residual_magnitude", ""
        ),
        "ignored": gt.ignored,
        "matched": detection is not None,
    }


def _save_visualizations(
    output_dir: Path,
    pair: FramePair,
    image: object,
    rows: Sequence[dict[str, object]],
) -> None:
    import cv2

    for method in METHODS:
        target_dir = output_dir / "visualizations" / method
        target_dir.mkdir(parents=True, exist_ok=True)
        canvas = image.copy()
        for row in rows:
            bbox = ast.literal_eval(str(row["gt_bbox"]))
            gt = str(row["gt_motion"])
            pred = str(row[f"{method}_prediction"] or "unmatched")
            ignored = row["ignored"] is True
            correct = not ignored and gt == pred
            color = (128, 128, 128) if ignored else (0, 255, 0) if correct else (0, 0, 255)
            p1, p2 = (round(bbox[0]), round(bbox[1])), (round(bbox[2]), round(bbox[3]))
            cv2.rectangle(canvas, p1, p2, color, 3)
            label = "IGNORE" if ignored else f"GT:{gt.upper()} | PRED:{pred.upper()}"
            cv2.putText(
                canvas,
                label,
                (p1[0], max(20, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        path = target_dir / f"{pair.previous.path.stem}_to_{pair.current.path.stem}.jpg"
        if not cv2.imwrite(str(path), canvas):
            raise OSError(f"Görselleştirme yazılamadı: {path}")


def _emit_summary(
    emit: Callable[[str], None],
    summaries: Sequence[dict[str, object]],
    matched_gt: int,
    unmatched_gt: int,
    unmatched_detections: Sequence[object],
) -> None:
    emit(
        f"Association: matched_gt={matched_gt} unmatched_gt={unmatched_gt} "
        f"unmatched_detections={len(unmatched_detections)}"
    )
    emit("method strict decided coverage moving_f1 stationary_f1 macro_f1 false_moving false_stationary")
    for row in summaries:
        emit(
            f"{row['method']} {row['strict_accuracy']:.6f} "
            f"{row['decided_only_accuracy']:.6f} {row['coverage']:.6f} "
            f"{row['moving_f1']:.6f} {row['stationary_f1']:.6f} "
            f"{row['macro_f1']:.6f} {row['false_moving']} {row['false_stationary']}"
        )


def _motion_from_object(node: ET.Element) -> str | None:
    for tag in ("motion", "motion_status", "hareket", "name", "class"):
        motion = normalize_motion_label(node.findtext(tag))
        if motion is not None:
            return motion
    return None


def _bbox_from_object(node: ET.Element) -> tuple[float, float, float, float] | None:
    container = node.find("bndbox")
    if container is None:
        container = node.find("bbox")
    if container is None:
        return None
    options = (("xmin", "ymin", "xmax", "ymax"), ("x1", "y1", "x2", "y2"))
    for names in options:
        try:
            values = tuple(float(container.findtext(name, "")) for name in names)
        except ValueError:
            continue
        if all(math.isfinite(value) for value in values) and values[2] > values[0] and values[3] > values[1]:
            return values
    return None


def _bbox_within(bbox: Sequence[float], width: int, height: int) -> bool:
    return 0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height


def _detection_bbox(detection: DetectedObject) -> tuple[float, float, float, float]:
    return (
        detection.top_left_x,
        detection.top_left_y,
        detection.bottom_right_x,
        detection.bottom_right_y,
    )


def _bbox_text(bbox: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.6f}" for value in bbox) + "]"


def _integer(value: str | None) -> int:
    try:
        return int(value or "")
    except ValueError:
        return 0


def _optional(value: object | None) -> object:
    return "" if value is None else value


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = EvaluationOptions(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        min_iou=args.min_iou,
        save_visualizations=args.save_visualizations,
    )
    try:
        asyncio.run(run_evaluation(get_settings(), options))
    except Exception as exc:
        print(f"Motion evaluation: FAIL ({type(exc).__name__}: {exc})")
        print("Prediction submission: DISABLED")
        return 10
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
