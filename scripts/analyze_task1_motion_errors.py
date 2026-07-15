from __future__ import annotations

import argparse
import ast
import csv
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.benchmark_task1_motion import METHODS

MOVING = "moving"
STATIONARY = "stationary"
UNKNOWN = "unknown"

GROUP_FALSE_MOVING = "false_moving"
GROUP_TRUE_STATIONARY = "true_stationary"
GROUP_TRUE_MOVING = "true_moving"
GROUP_FALSE_STATIONARY = "false_stationary"
GROUP_UNKNOWN = "unknown"

STAT_FEATURES = (
    "flow_residual",
    "bbox_center_residual",
    "bbox_iou",
    "association_score",
    "local_corrected_residual",
    "background_residual",
    "homography_inlier_ratio",
    "reprojection_error",
)
SEPARATION_FEATURES = (
    "flow_residual",
    "bbox_center_residual",
    "local_corrected_residual",
)

STATIONARY_THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
MOVING_THRESHOLDS = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)

SAMPLE_COLUMNS = (
    "method",
    "group",
    "previous_frame",
    "current_frame",
    "gt_object_index",
    "detection_index",
    "gt_motion",
    "predicted_motion",
    "gt_bbox",
    "detection_bbox",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "detection_confidence",
    "frame_edge",
    "bbox_visible_ratio",
    "homography_valid",
    "homography_quality_level",
    "homography_matches",
    "homography_inliers",
    "homography_inlier_ratio",
    "reprojection_error",
    "spatial_coverage",
    "projected_overlap",
    "condition_number",
    "residual_x",
    "residual_y",
    "flow_residual",
    "residual_p50",
    "residual_p75",
    "residual_p90",
    "residual_p95",
    "residual_valid_pixels",
    "projected_bbox",
    "bbox_iou",
    "bbox_center_residual",
    "bbox_size_ratio",
    "association_score",
    "local_vehicle_residual_x",
    "local_vehicle_residual_y",
    "local_vehicle_residual",
    "local_background_residual_x",
    "local_background_residual_y",
    "background_residual",
    "local_corrected_residual_x",
    "local_corrected_residual_y",
    "local_corrected_residual",
    "local_vehicle_valid_pixels",
    "local_background_valid_pixels",
    "local_background_valid_ratio",
    "local_decision_reason",
    "hybrid_bbox_result",
    "hybrid_flow_result",
    "hybrid_final_result",
    "hybrid_decision_reason",
)

GROUP_STAT_COLUMNS = (
    "method",
    "group",
    "feature",
    "count",
    "minimum",
    "maximum",
    "mean",
    "median",
    "p25",
    "p75",
    "p90",
    "p95",
)

SWEEP_COLUMNS = (
    "stationary_threshold",
    "moving_threshold",
    "total",
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
    "balanced_accuracy",
)

SEPARATION_COLUMNS = (
    "feature",
    "count",
    "moving_count",
    "stationary_count",
    "auc",
    "separation_auc",
    "moving_median",
    "stationary_median",
)


@dataclass(frozen=True, slots=True)
class AnalysisOptions:
    images_dir: Path
    labels_dir: Path
    evaluation_csv: Path
    output_dir: Path
    diagnostics_csv: Path | None = None
    homography_quality_csv: Path | None = None
    save_visualizations: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task 1 STATIONARY→MOVING hatalarını tamamen offline analiz et."
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--evaluation-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--diagnostics-csv", type=Path)
    parser.add_argument("--homography-quality-csv", type=Path)
    parser.add_argument("--save-visualizations", action="store_true")
    return parser


def classification_group(gt: str, prediction: str) -> str | None:
    if prediction == UNKNOWN:
        return GROUP_UNKNOWN
    mapping = {
        (STATIONARY, MOVING): GROUP_FALSE_MOVING,
        (STATIONARY, STATIONARY): GROUP_TRUE_STATIONARY,
        (MOVING, MOVING): GROUP_TRUE_MOVING,
        (MOVING, STATIONARY): GROUP_FALSE_STATIONARY,
    }
    return mapping.get((gt, prediction))


def numeric_statistics(values: Iterable[object]) -> dict[str, object]:
    data = np.asarray(
        [number for value in values if (number := _number(value)) is not None],
        dtype=np.float64,
    )
    if data.size == 0:
        return {
            "count": 0,
            "minimum": "",
            "maximum": "",
            "mean": "",
            "median": "",
            "p25": "",
            "p75": "",
            "p90": "",
            "p95": "",
        }
    return {
        "count": int(data.size),
        "minimum": float(data.min()),
        "maximum": float(data.max()),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
    }


def group_statistics(samples: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups = sorted({(str(row["method"]), str(row["group"])) for row in samples})
    for method, group in groups:
        selected = [
            row for row in samples if row["method"] == method and row["group"] == group
        ]
        for feature in STAT_FEATURES:
            rows.append(
                {
                    "method": method,
                    "group": group,
                    "feature": feature,
                    **numeric_statistics(row.get(feature) for row in selected),
                }
            )
    return rows


def classification_metrics(
    ground_truth: Sequence[str], predictions: Sequence[str]
) -> dict[str, float | int]:
    pairs = list(zip(ground_truth, predictions, strict=True))
    total = len(pairs)
    tp_moving = sum(gt == MOVING and pred == MOVING for gt, pred in pairs)
    tp_stationary = sum(gt == STATIONARY and pred == STATIONARY for gt, pred in pairs)
    moving_pred = sum(pred == MOVING for _, pred in pairs)
    stationary_pred = sum(pred == STATIONARY for _, pred in pairs)
    unknown_pred = sum(pred == UNKNOWN for _, pred in pairs)
    moving_gt = sum(gt == MOVING for gt, _ in pairs)
    stationary_gt = sum(gt == STATIONARY for gt, _ in pairs)
    decided = moving_pred + stationary_pred
    correct = tp_moving + tp_stationary
    moving_precision, moving_recall = _divide(tp_moving, moving_pred), _divide(
        tp_moving, moving_gt
    )
    stationary_precision, stationary_recall = _divide(
        tp_stationary, stationary_pred
    ), _divide(tp_stationary, stationary_gt)
    moving_f1 = _f1(moving_precision, moving_recall)
    stationary_f1 = _f1(stationary_precision, stationary_recall)
    return {
        "total": total,
        "moving_pred": moving_pred,
        "stationary_pred": stationary_pred,
        "unknown_pred": unknown_pred,
        "strict_accuracy": _divide(correct, total),
        "decided_only_accuracy": _divide(correct, decided),
        "coverage": _divide(decided, total),
        "moving_precision": moving_precision,
        "moving_recall": moving_recall,
        "moving_f1": moving_f1,
        "stationary_precision": stationary_precision,
        "stationary_recall": stationary_recall,
        "stationary_f1": stationary_f1,
        "macro_f1": (moving_f1 + stationary_f1) / 2,
        "balanced_accuracy": (moving_recall + stationary_recall) / 2,
    }


def local_threshold_sweep(base_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    gt = [str(row["gt_motion"]) for row in base_rows]
    for stationary_threshold in STATIONARY_THRESHOLDS:
        for moving_threshold in MOVING_THRESHOLDS:
            if stationary_threshold >= moving_threshold:
                continue
            predictions = [
                _threshold_prediction(
                    _number(row.get("local_corrected_residual")),
                    stationary_threshold,
                    moving_threshold,
                )
                for row in base_rows
            ]
            rows.append(
                {
                    "stationary_threshold": stationary_threshold,
                    "moving_threshold": moving_threshold,
                    **classification_metrics(gt, predictions),
                }
            )
    return rows


def bbox_threshold_sensitivity(base_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    gt = [str(row["gt_motion"]) for row in base_rows]
    for stationary in (2.0, 3.0, 4.0, 5.0):
        for moving in (6.0, 8.0, 10.0, 12.0):
            if stationary >= moving:
                continue
            for min_iou in (0.1, 0.2, 0.3, 0.5):
                for min_score in (0.25, 0.35, 0.5):
                    predictions = [
                        _bbox_prediction(row, stationary, moving, min_iou, min_score)
                        for row in base_rows
                    ]
                    output.append(
                        {
                            "stationary_threshold": stationary,
                            "moving_threshold": moving,
                            "min_iou": min_iou,
                            "min_association_score": min_score,
                            **classification_metrics(gt, predictions),
                        }
                    )
    return output


def manual_auc(labels: Sequence[int], values: Sequence[float]) -> float | None:
    if len(labels) != len(values) or not labels:
        return None
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = ((position + 1) + end) / 2
        for index in order[position:end]:
            ranks[index] = average_rank
        position = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def feature_separation(base_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for feature in SEPARATION_FEATURES:
        pairs = [
            (1 if row["gt_motion"] == MOVING else 0, number)
            for row in base_rows
            if (number := _number(row.get(feature))) is not None
        ]
        labels, values = [item[0] for item in pairs], [item[1] for item in pairs]
        auc = manual_auc(labels, values)
        moving = [value for label, value in pairs if label == 1]
        stationary = [value for label, value in pairs if label == 0]
        output.append(
            {
                "feature": feature,
                "count": len(pairs),
                "moving_count": len(moving),
                "stationary_count": len(stationary),
                "auc": "" if auc is None else auc,
                "separation_auc": "" if auc is None else max(auc, 1 - auc),
                "moving_median": "" if not moving else float(np.median(moving)),
                "stationary_median": "" if not stationary else float(np.median(stationary)),
            }
        )
    return output


def build_samples(
    evaluation_rows: Sequence[dict[str, str]],
    diagnostics_rows: Sequence[dict[str, str]] = (),
    quality_rows: Sequence[dict[str, str]] = (),
    *,
    image_shapes: dict[str, tuple[int, int]] | None = None,
) -> tuple[list[dict[str, object]], list[str], list[dict[str, object]]]:
    diagnostics = {
        (row.get("frame_previous", ""), row.get("frame_current", ""), row.get("vehicle_index", "")): row
        for row in diagnostics_rows
    }
    quality = {
        (row.get("previous_frame", ""), row.get("current_frame", "")): row
        for row in quality_rows
    }
    samples: list[dict[str, object]] = []
    malformed: list[str] = []
    base_rows: list[dict[str, object]] = []
    for line, row in enumerate(evaluation_rows, start=2):
        if not _truth(row.get("matched")) or _truth(row.get("ignored")):
            continue
        gt = str(row.get("gt_motion", "")).strip().casefold()
        if gt not in {MOVING, STATIONARY}:
            malformed.append(f"line {line}: invalid gt_motion")
            continue
        detection_index = str(row.get("matched_detection_index", "")).strip()
        bbox = _parse_bbox(row.get("detection_bbox"))
        if not detection_index or bbox is None:
            malformed.append(f"line {line}: invalid matched detection")
            continue
        diagnostic = diagnostics.get(
            (row.get("previous_frame", ""), row.get("current_frame", ""), detection_index),
            {},
        )
        quality_row = quality.get(
            (row.get("previous_frame", ""), row.get("current_frame", "")), {}
        )
        enriched = _enrich_base_row(row, diagnostic, quality_row, bbox, image_shapes or {})
        enriched["gt_motion"] = gt
        base_rows.append(enriched)
        for method in METHODS:
            prediction = str(
                row.get(f"{method}_prediction", UNKNOWN)
            ).strip().casefold()
            group = classification_group(gt, prediction)
            if group is None:
                malformed.append(f"line {line}: invalid {method} prediction")
                continue
            samples.append(
                {
                    "method": method,
                    "group": group,
                    "gt_motion": gt,
                    "predicted_motion": prediction,
                    **enriched,
                }
            )
    return samples, malformed, base_rows


def run_analysis(
    options: AnalysisOptions, *, emit: Callable[[str], None] = print
) -> dict[str, object]:
    images_dir = options.images_dir.expanduser().resolve()
    labels_dir = options.labels_dir.expanduser().resolve()
    evaluation_csv = options.evaluation_csv.expanduser().resolve()
    output_dir = options.output_dir.expanduser().resolve()
    if not images_dir.is_dir() or not labels_dir.is_dir() or not evaluation_csv.is_file():
        raise ValueError("images-dir, labels-dir ve evaluation-csv mevcut olmalıdır")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_rows = _read_csv(evaluation_csv)
    diagnostic_rows = _read_csv(options.diagnostics_csv) if options.diagnostics_csv else []
    quality_rows = (
        _read_csv(options.homography_quality_csv)
        if options.homography_quality_csv
        else []
    )
    shapes = _load_image_shapes(images_dir)
    samples, malformed, base_rows = build_samples(
        evaluation_rows, diagnostic_rows, quality_rows, image_shapes=shapes
    )
    stats = group_statistics(samples)
    sweep = local_threshold_sweep(base_rows)
    bbox_sweep = bbox_threshold_sensitivity(base_rows)
    separation = feature_separation(base_rows)
    top_false_moving = _top_samples(samples, GROUP_FALSE_MOVING, reverse=True)
    top_false_stationary = _top_samples(samples, GROUP_FALSE_STATIONARY, reverse=False)
    ambiguous = _ambiguous_samples(samples)
    paths = {
        "samples_csv": output_dir / "motion_error_samples.csv",
        "statistics_csv": output_dir / "motion_error_group_statistics.csv",
        "separation_csv": output_dir / "motion_feature_separation.csv",
        "local_sweep_csv": output_dir / "homography_local_threshold_sweep.csv",
        "bbox_sweep_csv": output_dir / "homography_bbox_threshold_sensitivity.csv",
        "top_false_moving_csv": output_dir / "top_false_moving.csv",
        "top_false_stationary_csv": output_dir / "top_false_stationary.csv",
        "ambiguous_csv": output_dir / "most_ambiguous_samples.csv",
        "summary_txt": output_dir / "motion_error_analysis_summary.txt",
    }
    _write_csv(paths["samples_csv"], SAMPLE_COLUMNS, samples)
    _write_csv(paths["statistics_csv"], GROUP_STAT_COLUMNS, stats)
    _write_csv(paths["separation_csv"], SEPARATION_COLUMNS, separation)
    _write_csv(paths["local_sweep_csv"], SWEEP_COLUMNS, sweep)
    _write_csv(paths["bbox_sweep_csv"], tuple(bbox_sweep[0]) if bbox_sweep else (), bbox_sweep)
    _write_csv(paths["top_false_moving_csv"], SAMPLE_COLUMNS, top_false_moving)
    _write_csv(paths["top_false_stationary_csv"], SAMPLE_COLUMNS, top_false_stationary)
    _write_csv(paths["ambiguous_csv"], SAMPLE_COLUMNS, ambiguous)
    summary_lines = _summary_lines(samples, base_rows, separation, sweep, malformed)
    paths["summary_txt"].write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    for line in summary_lines:
        emit(line)
    if options.save_visualizations:
        _save_visualizations(images_dir, output_dir, top_false_moving[:20])
    emit("Prediction submission: DISABLED")
    return {
        "samples": samples,
        "base_rows": base_rows,
        "statistics": stats,
        "separation": separation,
        "local_sweep": sweep,
        "bbox_sweep": bbox_sweep,
        "malformed_rows": malformed,
        **paths,
    }


def _enrich_base_row(row, diagnostic, quality, bbox, shapes):
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    frame_shape = shapes.get(str(row.get("current_frame", "")))
    edge = False
    if frame_shape:
        frame_height, frame_width = frame_shape
        edge = bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= frame_width - 1 or bbox[3] >= frame_height - 1
    return {
        "previous_frame": row.get("previous_frame", ""),
        "current_frame": row.get("current_frame", ""),
        "gt_object_index": row.get("gt_object_index", ""),
        "detection_index": row.get("matched_detection_index", ""),
        "gt_bbox": row.get("gt_bbox", ""),
        "detection_bbox": row.get("detection_bbox", ""),
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": width * height,
        "detection_confidence": _value(row, "detection_confidence"),
        "frame_edge": edge,
        "bbox_visible_ratio": 1.0,
        "homography_valid": row.get("homography_valid", ""),
        "homography_quality_level": diagnostic.get("hybrid_homography_quality_level", row.get("homography_quality_level", "")),
        "homography_matches": _value(diagnostic, "homography_matches"),
        "homography_inliers": _value(diagnostic, "homography_inliers"),
        "homography_inlier_ratio": _value(diagnostic, "homography_inlier_ratio", row.get("homography_inlier_ratio")),
        "reprojection_error": _value(quality, "reprojection_error"),
        "spatial_coverage": _value(quality, "spatial_coverage"),
        "projected_overlap": _value(quality, "projected_overlap"),
        "condition_number": _value(quality, "condition_number"),
        "residual_x": "",
        "residual_y": "",
        "flow_residual": _value(diagnostic, "homography_residual_px", row.get("flow_residual")),
        "residual_p50": "",
        "residual_p75": "",
        "residual_p90": "",
        "residual_p95": "",
        "residual_valid_pixels": "",
        "projected_bbox": diagnostic.get("bbox_projected_bbox", ""),
        "bbox_iou": _value(diagnostic, "bbox_iou", row.get("bbox_iou")),
        "bbox_center_residual": _value(diagnostic, "bbox_center_residual_px", row.get("bbox_center_residual")),
        "bbox_size_ratio": _value(diagnostic, "bbox_size_ratio"),
        "association_score": _value(diagnostic, "bbox_association_score"),
        "local_vehicle_residual_x": _value(diagnostic, "local_vehicle_residual_x"),
        "local_vehicle_residual_y": _value(diagnostic, "local_vehicle_residual_y"),
        "local_vehicle_residual": _value(diagnostic, "local_vehicle_residual_magnitude"),
        "local_background_residual_x": _value(diagnostic, "local_background_residual_x"),
        "local_background_residual_y": _value(diagnostic, "local_background_residual_y"),
        "background_residual": _value(diagnostic, "local_background_residual_magnitude"),
        "local_corrected_residual_x": _value(diagnostic, "local_corrected_residual_x"),
        "local_corrected_residual_y": _value(diagnostic, "local_corrected_residual_y"),
        "local_corrected_residual": _value(diagnostic, "local_corrected_residual_magnitude", row.get("local_corrected_residual")),
        "local_vehicle_valid_pixels": _value(diagnostic, "local_vehicle_valid_pixels"),
        "local_background_valid_pixels": _value(diagnostic, "local_background_valid_pixels"),
        "local_background_valid_ratio": _value(diagnostic, "local_background_valid_ratio"),
        "local_decision_reason": diagnostic.get("local_decision_reason", ""),
        "hybrid_bbox_result": diagnostic.get("hybrid_bbox_result", ""),
        "hybrid_flow_result": diagnostic.get("hybrid_flow_result", ""),
        "hybrid_final_result": diagnostic.get("homography_hybrid_result", ""),
        "hybrid_decision_reason": diagnostic.get("hybrid_decision_reason", ""),
    }


def _summary_lines(samples, base_rows, separation, sweep, malformed):
    lines = [
        "===== TASK 1 MOTION ERROR ANALYSIS =====",
        f"GT MOVING: {sum(row['gt_motion'] == MOVING for row in base_rows)}",
        f"GT STATIONARY: {sum(row['gt_motion'] == STATIONARY for row in base_rows)}",
    ]
    for method in METHODS:
        selected = [row for row in samples if row["method"] == method]
        counts = {group: sum(row["group"] == group for row in selected) for group in (GROUP_FALSE_MOVING, GROUP_FALSE_STATIONARY, GROUP_UNKNOWN)}
        lines.append(
            f"{method}: false_moving={counts[GROUP_FALSE_MOVING]} "
            f"false_stationary={counts[GROUP_FALSE_STATIONARY]} unknown={counts[GROUP_UNKNOWN]}"
        )
        for feature in ("flow_residual", "bbox_center_residual", "local_corrected_residual"):
            false_values = [_number(row.get(feature)) for row in selected if row["group"] == GROUP_FALSE_MOVING]
            true_values = [_number(row.get(feature)) for row in selected if row["group"] == GROUP_TRUE_MOVING]
            false_values = [value for value in false_values if value is not None]
            true_values = [value for value in true_values if value is not None]
            lines.append(
                f"  {feature}: false_moving_median={_median_text(false_values)} "
                f"true_moving_median={_median_text(true_values)}"
            )
    available = [row for row in separation if row["separation_auc"] != ""]
    if available:
        best = max(available, key=lambda row: float(row["separation_auc"]))
        lines.append(
            f"Best separating feature: {best['feature']} auc={float(best['auc']):.6f} "
            f"separation_auc={float(best['separation_auc']):.6f}"
        )
    ranked = sorted(sweep, key=lambda row: (row["macro_f1"], row["balanced_accuracy"], row["coverage"]), reverse=True)
    lines.append("Top 5 local threshold combinations by macro_f1:")
    for row in ranked[:5]:
        lines.append(
            f"  stationary={row['stationary_threshold']:.0f} moving={row['moving_threshold']:.0f} "
            f"macro_f1={row['macro_f1']:.6f} balanced_accuracy={row['balanced_accuracy']:.6f} "
            f"coverage={row['coverage']:.6f}"
        )
    for minimum in (0.70, 0.80):
        candidates = [row for row in ranked if row["coverage"] >= minimum]
        lines.append(
            f"Best macro_f1 with coverage >= {minimum:.2f}: "
            + ("none" if not candidates else f"stationary={candidates[0]['stationary_threshold']:.0f} moving={candidates[0]['moving_threshold']:.0f} macro_f1={candidates[0]['macro_f1']:.6f}")
        )
    if available:
        strongest = max(float(row["separation_auc"]) for row in available)
        conclusion = (
            "Features overlap strongly; threshold tuning alone is unlikely to solve false moving errors."
            if strongest < 0.70
            else "At least one feature has useful separation; offline threshold tuning may improve the trade-off, but production was not changed."
        )
        lines.append(f"Interpretation: {conclusion}")
    lines.append(f"Malformed/skipped rows: {len(malformed)}")
    return lines


def _top_samples(samples, group, *, reverse):
    selected = [row for row in samples if row["group"] == group]
    selected.sort(key=_diagnostic_score, reverse=reverse)
    output = []
    for method in METHODS:
        output.extend([row for row in selected if row["method"] == method][:20])
    return output


def _ambiguous_samples(samples):
    selected = [row for row in samples if row["group"] == GROUP_UNKNOWN]
    return sorted(
        selected,
        key=lambda row: abs((_number(row.get("local_corrected_residual")) or 4.0) - 4.0),
    )[:100]


def _diagnostic_score(row):
    values = [
        _number(row.get("local_corrected_residual")),
        _number(row.get("flow_residual")),
        _number(row.get("bbox_center_residual")),
    ]
    return max((value for value in values if value is not None), default=-math.inf)


def _save_visualizations(images_dir, output_dir, rows):
    import cv2

    target = output_dir / "visualizations" / "top_false_moving"
    target.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        image = cv2.imread(str(images_dir / str(row["current_frame"])), cv2.IMREAD_COLOR)
        bbox = _parse_bbox(row.get("detection_bbox"))
        if image is None or bbox is None:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        margin = 30
        crop = image[max(0, y1 - margin):min(image.shape[0], y2 + margin), max(0, x1 - margin):min(image.shape[1], x2 + margin)].copy()
        lines = (
            f"GT:STATIONARY PRED:MOVING ({row['method']})",
            f"flow={_text(row.get('flow_residual'))} bbox={_text(row.get('bbox_center_residual'))}",
            f"local={_text(row.get('local_corrected_residual'))}",
        )
        for offset, text in enumerate(lines):
            cv2.putText(crop, text, (5, 20 + offset * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(target / f"{index:03d}_{Path(str(row['current_frame'])).stem}_{row['method']}.jpg"), crop)


def _threshold_prediction(value, stationary, moving):
    if value is None:
        return UNKNOWN
    if value <= stationary:
        return STATIONARY
    if value >= moving:
        return MOVING
    return UNKNOWN


def _bbox_prediction(row, stationary, moving, min_iou, min_score):
    residual, iou, score = (
        _number(row.get("bbox_center_residual")),
        _number(row.get("bbox_iou")),
        _number(row.get("association_score")),
    )
    if residual is None or iou is None or score is None or iou < min_iou or score < min_score or _truth(row.get("frame_edge")):
        return UNKNOWN
    if residual <= stationary:
        return STATIONARY
    if residual >= moving:
        return MOVING
    return UNKNOWN


def _load_image_shapes(images_dir):
    import cv2

    shapes = {}
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp"}:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                shapes[path.name] = image.shape[:2]
    return shapes


def _parse_bbox(value):
    try:
        parsed = ast.literal_eval(str(value))
        numbers = tuple(float(item) for item in parsed)
    except (ValueError, TypeError, SyntaxError):
        return None
    if len(numbers) != 4 or not all(math.isfinite(item) for item in numbers) or numbers[2] <= numbers[0] or numbers[3] <= numbers[1]:
        return None
    return numbers


def _read_csv(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"CSV bulunamadı: {resolved}")
    with resolved.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, columns, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _value(mapping, key, fallback=""):
    value = mapping.get(key, "")
    return fallback if value in (None, "") else value


def _truth(value):
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _median_text(values):
    return "n/a" if not values else f"{float(np.median(values)):.6f}"


def _text(value):
    number = _number(value)
    return "n/a" if number is None else f"{number:.2f}px"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = AnalysisOptions(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        evaluation_csv=args.evaluation_csv,
        output_dir=args.output_dir,
        diagnostics_csv=args.diagnostics_csv,
        homography_quality_csv=args.homography_quality_csv,
        save_visualizations=args.save_visualizations,
    )
    try:
        run_analysis(options)
    except Exception as exc:
        print(f"Motion error analysis: FAIL ({type(exc).__name__}: {exc})")
        print("Prediction submission: DISABLED")
        return 10
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
