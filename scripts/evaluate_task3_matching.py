from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings, get_settings
from app.schemas import ImageModality
from app.services.common import FrameContext
from app.services.matching.dinov2_runtime import Dinov2RuntimeRegistry
from app.services.matching.interface import ReferenceImage
from app.services.matching.service import DinoReferenceMatchingService
from app.services.matching.local_features import LocalArtifactUnavailable, LocalFeatureError
from app.services.matching.local_matcher import LocalRefinementPipeline, ReferenceLocalFeatureCache
from scripts.validate_dinov2_artifacts import (
    MatchArtifacts,
    descriptor_metadata,
    evaluate_match,
    inspect_artifacts,
    load_local_image,
)

EXIT_OK = 0
EXIT_CONFIG = 10
EXIT_RUNTIME = 20
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

DETAIL_COLUMNS = (
    "reference_id", "object_id", "reference_path", "frame_name", "frame_path",
    "expected_match", "descriptor_reference_count", "descriptor_frame_count",
    "descriptor_dimension", "correspondence_count", "similarity_min", "similarity_mean",
    "similarity_median", "similarity_max", "spatial_coverage", "homography_valid",
    "homography_inliers", "homography_inlier_ratio", "homography_rms",
    "projected_polygon_valid", "visible_ratio", "bbox", "bbox_valid", "confidence",
    "accepted", "rejection_reason", "matched_reference_object_produced",
    "service_diagnostic_consistent", "coarse_matching_seconds", "homography_seconds",
    "pair_total_seconds",
    "coarse_correspondence_count", "coarse_similarity", "coarse_coverage",
    "aliked_reference_keypoints", "aliked_frame_keypoints", "lightglue_match_count",
    "lightglue_mean_score", "local_homography_inliers", "local_homography_inlier_ratio",
    "local_homography_rms", "dinov2_accepted", "local_accepted", "hybrid_accepted",
    "local_reason", "hybrid_reason", "local_bbox", "local_confidence",
    "hybrid_bbox", "hybrid_confidence",
    "local_device", "aliked_reference_seconds", "aliked_frame_seconds",
    "lightglue_seconds", "local_homography_seconds", "hybrid_total_seconds",
    "model_preloaded", "warmup_completed", "warmup_time_sec",
    "reference_cache_warmed", "reference_prepare_time_sec",
    "frame_local_refinement_time_sec",
)

SUMMARY_COLUMNS = (
    "reference_id", "crop_type", "pair_count", "accepted_count", "accepted_rate",
    "mean_correspondences", "mean_inlier_ratio", "mean_confidence",
)

PERFORMANCE_COLUMNS = (
    "frame_name", "reference_preparation_seconds", "frame_descriptor_seconds",
    "coarse_matching_seconds", "homography_seconds", "service_seconds", "total_seconds",
)

SENSITIVITY_COLUMNS = (
    "parameter", "threshold", "operator", "eligible_count", "pass_count",
    "positive_pass_rate", "negative_pass_rate", "precision", "recall", "f1",
)

CONSISTENCY_COLUMNS = (
    "reference_id", "reference_path", "frame_name", "ground_truth_expected_match",
    "gt_reference_resolved", "gt_frame_resolved", "matching_geometry_method",
    "dinov2_coarse_correspondence_count", "dinov2_coarse_mean_similarity",
    "dinov2_coarse_spatial_coverage", "dinov2_homography_inliers",
    "dinov2_homography_inlier_ratio", "dinov2_accepted", "dinov2_rejection_reason",
    "local_reference_keypoints", "local_frame_keypoints", "local_match_count",
    "local_mean_match_score", "local_homography_inliers", "local_homography_inlier_ratio",
    "local_homography_rms", "local_polygon_valid", "local_bbox_valid", "local_bbox",
    "local_confidence", "local_accepted", "local_rejection_reason",
    "hybrid_coarse_gate_passed", "hybrid_local_refinement_called",
    "hybrid_local_match_count", "hybrid_local_homography_inliers",
    "hybrid_local_homography_inlier_ratio", "hybrid_local_homography_rms",
    "hybrid_polygon_valid", "hybrid_bbox_valid", "hybrid_bbox", "hybrid_confidence",
    "hybrid_accepted", "hybrid_rejection_reason", "reference_active",
    "reference_prepare_timeout_seconds", "actual_reference_prepare_seconds",
    "reference_prepare_timeout_triggered", "local_refinement_timeout_seconds",
    "actual_local_refinement_seconds", "local_refinement_timeout_triggered",
    "service_timeout_stage", "service_timeout_limit_sec", "service_match_elapsed_sec",
    "service_match_outcome", "model_preloaded", "warmup_completed", "warmup_time_sec",
    "reference_cache_warmed", "reference_prepare_time_sec",
    "frame_local_refinement_time_sec",
    "fallback_to_dinov2_after_timeout", "service_result_count",
    "matched_reference_object_produced", "service_final_bbox", "service_final_confidence",
    "service_final_rejection_reason", "root_cause",
)


@dataclass(frozen=True, slots=True)
class EvaluationOptions:
    output_dir: Path
    references_dir: Path | None = None
    frames_dir: Path | None = None
    reference: Path | None = None
    frame: Path | None = None
    ground_truth_csv: Path | None = None
    save_visualizations: bool = False


@dataclass(frozen=True, slots=True)
class ImageAsset:
    path: Path
    content: bytes
    image: object
    sha256: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ReferenceAsset:
    reference_id: str
    object_id: int
    crop_type: str
    asset: ImageAsset


@dataclass(slots=True)
class ExtractionRecord:
    descriptor: object
    metrics: object
    elapsed_seconds: float


class RecordingRuntime:
    """Transparent instrumentation around the production DINOv2 runtime."""

    def __init__(self, runtime: object) -> None:
        self._runtime = runtime
        self.records: dict[str, list[ExtractionRecord]] = defaultdict(list)

    @property
    def model_hash(self):
        return self._runtime.model_hash

    @property
    def device(self):
        return self._runtime.device

    @property
    def is_loaded(self):
        return self._runtime.is_loaded

    @property
    def inference_lock(self):
        return getattr(self._runtime, "inference_lock", None)

    def extract(self, image, source_hash: str):
        started = time.perf_counter()
        descriptor, metrics = self._runtime.extract(image, source_hash)
        self.records[source_hash].append(
            ExtractionRecord(descriptor, metrics, time.perf_counter() - started)
        )
        return descriptor, metrics

    def count(self, source_hash: str) -> int:
        return len(self.records.get(source_hash, ()))

    def latest(self, source_hash: str) -> ExtractionRecord | None:
        values = self.records.get(source_hash)
        return values[-1] if values else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production Task 3 offline matching evaluation")
    parser.add_argument("--references-dir", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--ground-truth-csv", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--save-visualizations", action="store_true")
    return parser


def validate_options(options: EvaluationOptions) -> None:
    directory_mode = options.references_dir is not None or options.frames_dir is not None
    single_mode = options.reference is not None or options.frame is not None
    if directory_mode == single_mode:
        raise ValueError("use either --references-dir/--frames-dir or --reference/--frame")
    if directory_mode and (options.references_dir is None or options.frames_dir is None):
        raise ValueError("--references-dir and --frames-dir must be provided together")
    if single_mode and (options.reference is None or options.frame is None):
        raise ValueError("--reference and --frame must be provided together")


def _natural_key(path: Path):
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name))


def _image_paths(directory: Path) -> list[Path]:
    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"image directory does not exist: {root}")
    paths = sorted(
        (item for item in root.iterdir() if item.is_file() and item.suffix.casefold() in IMAGE_EXTENSIONS),
        key=_natural_key,
    )
    if not paths:
        raise ValueError(f"no supported images found: {root}")
    return paths


def load_asset(path: Path) -> ImageAsset:
    image, metadata = load_local_image(path)
    resolved = path.expanduser().resolve()
    content = resolved.read_bytes()
    return ImageAsset(
        resolved,
        content,
        image,
        str(metadata["source_hash"]),
        int(metadata["width"]),
        int(metadata["height"]),
    )


def _crop_type(stem: str) -> str:
    value = stem.casefold()
    if "tight" in value:
        return "tight"
    if "medium" in value:
        return "medium"
    if "wider" in value or "wide" in value:
        return "wide"
    return "unspecified"


def load_references(paths: Sequence[Path]) -> list[ReferenceAsset]:
    result: list[ReferenceAsset] = []
    used_ids: set[int] = set()
    for order, path in enumerate(paths, start=1):
        stem = path.stem
        match = re.search(r"(\d+)", stem)
        object_id = int(match.group(1)) if match and int(match.group(1)) > 0 else order
        while object_id in used_ids:
            object_id += 1
        used_ids.add(object_id)
        result.append(ReferenceAsset(stem, object_id, _crop_type(stem), load_asset(path)))
    return result


def _canonical_name(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip())


def load_ground_truth(path: Path | None) -> dict[tuple[str, str], str]:
    if path is None:
        return {}
    target = path.expanduser().resolve()
    rows: dict[tuple[str, str], str] = {}
    with target.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"reference_id", "frame_name", "expected_match"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("ground truth CSV requires reference_id,frame_name,expected_match")
        for row in reader:
            expected = str(row["expected_match"]).strip().upper()
            if expected not in {"0", "1", "IGNORE"}:
                raise ValueError(f"invalid expected_match: {expected}")
            key = (
                _canonical_name(str(row["reference_id"])),
                _canonical_name(str(row["frame_name"])),
            )
            if key in rows:
                raise ValueError(f"duplicate normalized ground truth row: {key}")
            rows[key] = expected
    return rows


def ground_truth_resolution(path: Path | None, references, frames) -> list[dict[str, object]]:
    if path is None:
        return []
    reference_names = {_canonical_name(item.reference_id) for item in references}
    frame_names = {_canonical_name(item.path.name) for item in frames}
    result: list[dict[str, object]] = []
    with path.expanduser().resolve().open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_reference = str(row.get("reference_id", ""))
            raw_frame = str(row.get("frame_name", ""))
            normalized_reference = _canonical_name(raw_reference)
            normalized_frame = _canonical_name(raw_frame)
            result.append({
                "raw_reference_id": raw_reference,
                "normalized_reference_id": normalized_reference,
                "raw_frame_name": raw_frame,
                "normalized_frame_name": normalized_frame,
                "expected_match": str(row.get("expected_match", "")).strip().upper(),
                "gt_reference_resolved": normalized_reference in reference_names,
                "gt_frame_resolved": normalized_frame in frame_names,
                "reference_normalization_changed": raw_reference != normalized_reference,
                "frame_normalization_changed": raw_frame != normalized_frame,
            })
    return result


def matching_config_snapshot(settings: Settings) -> dict[str, object]:
    def path_value(value):
        return str(value) if value is not None else None

    return {
        "MATCHING_GEOMETRY_METHOD": settings.matching_geometry_method,
        "MATCHING_LOCAL_REFINEMENT_ENABLED": settings.matching_local_refinement_enabled,
        "MATCHING_LOCAL_FALLBACK_TO_DINOV2": settings.matching_local_fallback_to_dinov2,
        "MATCHING_ALIKED_MODEL_PATH": path_value(settings.matching_aliked_model_path),
        "MATCHING_LIGHTGLUE_MODEL_PATH": path_value(settings.matching_lightglue_model_path),
        "MATCHING_LOCAL_MIN_KEYPOINTS": settings.matching_local_min_keypoints,
        "MATCHING_LOCAL_MIN_MATCHES": settings.matching_local_min_matches,
        "MATCHING_LOCAL_MIN_INLIERS": settings.matching_local_min_inliers,
        "MATCHING_LOCAL_MIN_INLIER_RATIO": settings.matching_local_min_inlier_ratio,
        "MATCHING_LOCAL_MAX_REPROJECTION_ERROR": settings.matching_local_max_reprojection_error,
        "MATCHING_LOCAL_REFINEMENT_TIMEOUT_SEC": settings.matching_local_refinement_timeout_sec,
        "MATCHING_PRELOAD_MODELS": settings.matching_preload_models,
        "MATCHING_WARMUP_ENABLED": settings.matching_warmup_enabled,
        "MATCHING_DINOV2_TIMEOUT_SECONDS": settings.matching_dinov2_timeout_seconds,
        "MATCHING_COARSE_TIMEOUT_SECONDS": settings.matching_coarse_timeout_seconds,
        "MATCHING_REFERENCE_TIMEOUT_SECONDS": settings.matching_reference_timeout_seconds,
    }


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "minimum": None, "mean": None, "p50": None, "p95": None, "maximum": None}
    ordered = sorted(finite)
    return {
        "count": len(ordered),
        "minimum": min(ordered),
        "mean": statistics.fmean(ordered),
        "p50": float(np.percentile(ordered, 50)),
        "p95": float(np.percentile(ordered, 95)),
        "maximum": max(ordered),
    }


def confusion_metrics(rows: Sequence[dict[str, object]]) -> dict[str, object] | None:
    evaluated = [row for row in rows if row.get("expected_match") in {"0", "1"}]
    if not evaluated:
        return None
    tp = sum(row["expected_match"] == "1" and bool(row["matched_reference_object_produced"]) for row in evaluated)
    fp = sum(row["expected_match"] == "0" and bool(row["matched_reference_object_produced"]) for row in evaluated)
    tn = sum(row["expected_match"] == "0" and not bool(row["matched_reference_object_produced"]) for row in evaluated)
    fn = sum(row["expected_match"] == "1" and not bool(row["matched_reference_object_produced"]) for row in evaluated)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (tp + tn) / len(evaluated), "evaluated_count": len(evaluated),
        "ignored_count": sum(row.get("expected_match") == "IGNORE" for row in rows),
    }


def local_timeout_sensitivity(
    rows: Sequence[dict[str, object]], limits: Sequence[float] = (2, 3, 4, 5, 6)
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for limit in limits:
        projected = []
        timeout_count = 0
        successful = 0
        for row in rows:
            elapsed = float(row.get("actual_local_refinement_seconds") or 0.0)
            observed_limit = row.get("service_timeout_limit_sec")
            is_observed_limit = (
                observed_limit is not None
                and math.isclose(float(observed_limit), float(limit), rel_tol=0.0, abs_tol=1e-9)
            )
            timed_out = (
                row.get("service_match_outcome") == "timeout"
                if is_observed_limit else elapsed > float(limit)
            )
            timeout_count += timed_out
            produced = (
                bool(row.get("matched_reference_object_produced"))
                if is_observed_limit
                else bool(row.get("selected_diagnostic_accepted")) and not timed_out
            )
            successful += produced
            projected.append({**row, "matched_reference_object_produced": produced})
        metrics = confusion_metrics(projected) or {}
        result.append({
            "timeout_sec": float(limit),
            "successful_production_matches": successful,
            "timeout_count": timeout_count,
            "true_positive": metrics.get("true_positive"),
            "true_negative": metrics.get("true_negative"),
            "false_positive": metrics.get("false_positive"),
            "false_negative": metrics.get("false_negative"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "accuracy": metrics.get("accuracy"),
            "mode": (
                "observed production run"
                if rows and all(
                    item.get("service_timeout_limit_sec") is not None
                    and math.isclose(
                        float(item["service_timeout_limit_sec"]), float(limit),
                        rel_tol=0.0, abs_tol=1e-9,
                    )
                    for item in rows
                )
                else "offline measured-duration replay"
            ),
        })
    return result


def feature_distributions(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    result = {}
    fields = ("correspondence_count", "homography_inlier_ratio", "homography_rms", "visible_ratio", "confidence")
    for label, expected in (("positive", "1"), ("negative", "0")):
        group = [row for row in rows if row.get("expected_match") == expected]
        result[label] = {
            field: _stats([float(row[field]) for row in group if row.get(field) is not None])
            for field in fields
        }
    result["accepted_confidence"] = _stats([
        float(row["confidence"]) for row in rows if row.get("accepted") and row.get("confidence") is not None
    ])
    result["rejected_confidence"] = _stats([
        float(row["confidence"]) for row in rows if not row.get("accepted") and row.get("confidence") is not None
    ])
    return result


def threshold_sensitivity(settings: Settings, rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    specifications = (
        ("minimum_similarity", "similarity_mean", ">=", settings.matching_coarse_min_similarity, (0.8, 1.0, 1.2)),
        ("minimum_inlier_ratio", "homography_inlier_ratio", ">=", settings.matching_homography_min_inlier_ratio, (0.75, 1.0, 1.25)),
        ("maximum_rms", "homography_rms", "<=", settings.matching_homography_max_rms_reprojection_error, (0.75, 1.0, 1.25)),
        ("minimum_visibility", "visible_ratio", ">=", settings.matching_geometry_min_visible_ratio, (0.75, 1.0, 1.25)),
        ("minimum_confidence", "confidence", ">=", settings.matching_min_confidence, (0.75, 1.0, 1.25)),
    )
    output = []
    for name, field, operator, baseline, factors in specifications:
        for factor in factors:
            threshold = max(0.0, min(1.0, baseline * factor)) if name != "maximum_rms" else baseline * factor
            eligible = [row for row in rows if row.get(field) is not None and row.get("expected_match") != "IGNORE"]
            passed = [row for row in eligible if (float(row[field]) >= threshold if operator == ">=" else float(row[field]) <= threshold)]
            positives = [row for row in eligible if row.get("expected_match") == "1"]
            negatives = [row for row in eligible if row.get("expected_match") == "0"]
            tp = sum(row.get("expected_match") == "1" for row in passed)
            fp = sum(row.get("expected_match") == "0" for row in passed)
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / len(positives) if positives else None
            output.append({
                "parameter": name, "threshold": threshold, "operator": operator,
                "eligible_count": len(eligible), "pass_count": len(passed),
                "positive_pass_rate": sum(row in passed for row in positives) / len(positives) if positives else None,
                "negative_pass_rate": sum(row in passed for row in negatives) / len(negatives) if negatives else None,
                "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None,
            })
    return output


def reference_summary(references: Sequence[ReferenceAsset], rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for reference in references:
        group = [row for row in rows if row["reference_id"] == reference.reference_id]
        accepted = [row for row in group if row["matched_reference_object_produced"]]
        summaries.append({
            "reference_id": reference.reference_id,
            "crop_type": reference.crop_type,
            "pair_count": len(group),
            "accepted_count": len(accepted),
            "accepted_rate": len(accepted) / len(group) if group else 0.0,
            "mean_correspondences": statistics.fmean(float(row["correspondence_count"]) for row in group) if group else None,
            "mean_inlier_ratio": statistics.fmean(float(row["homography_inlier_ratio"]) for row in group if row["homography_inlier_ratio"] is not None) if any(row["homography_inlier_ratio"] is not None for row in group) else None,
            "mean_confidence": statistics.fmean(float(row["confidence"]) for row in accepted if row["confidence"] is not None) if any(row["confidence"] is not None for row in accepted) else None,
        })
    return summaries


def _flatten_pair(reference: ReferenceAsset, frame: ImageAsset, diagnostic, artifacts, timings, produced, expected):
    similarity = diagnostic.get("similarity") or {}
    bbox = diagnostic.get("clipped_bbox")
    service_object = next((item for item in produced if item.object_id == reference.object_id), None)
    accepted = bool(diagnostic.get("accepted"))
    return {
        "reference_id": reference.reference_id,
        "object_id": reference.object_id,
        "reference_path": str(reference.asset.path),
        "frame_name": frame.path.name,
        "frame_path": str(frame.path),
        "expected_match": expected,
        "descriptor_reference_count": None,
        "descriptor_frame_count": None,
        "descriptor_dimension": None,
        "correspondence_count": int(diagnostic.get("correspondence_count") or 0),
        "similarity_min": similarity.get("minimum"),
        "similarity_mean": similarity.get("mean"),
        "similarity_median": similarity.get("median"),
        "similarity_max": similarity.get("maximum"),
        "spatial_coverage": diagnostic.get("spatial_coverage"),
        "homography_valid": bool(diagnostic.get("homography_valid")),
        "homography_inliers": diagnostic.get("inlier_count"),
        "homography_inlier_ratio": diagnostic.get("inlier_ratio"),
        "homography_rms": diagnostic.get("rms_reprojection_error"),
        "projected_polygon_valid": bool(artifacts.polygon and artifacts.polygon.valid),
        "visible_ratio": diagnostic.get("visible_ratio"),
        "bbox": bbox,
        "bbox_valid": bbox is not None,
        "confidence": diagnostic.get("confidence"),
        "accepted": accepted,
        "rejection_reason": diagnostic.get("failure_reason"),
        "dinov2_accepted": accepted,
        "dinov2_rejection_reason": diagnostic.get("failure_reason"),
        "matched_reference_object_produced": service_object is not None,
        "service_result_count": len(produced),
        "service_final_bbox": (
            [service_object.top_left_x, service_object.top_left_y,
             service_object.bottom_right_x, service_object.bottom_right_y]
            if service_object is not None else None
        ),
        "service_final_confidence": service_object.confidence if service_object is not None else None,
        "service_diagnostic_consistent": accepted == (service_object is not None),
        "coarse_matching_seconds": timings.get("coarse_matching_seconds", 0.0),
        "homography_seconds": timings.get("homography_geometry_seconds", 0.0),
        "pair_total_seconds": timings.get("coarse_matching_seconds", 0.0) + timings.get("homography_geometry_seconds", 0.0),
    }


def _local_geometry_flags(result) -> tuple[bool, bool]:
    if result is None:
        return False, False
    reason = result.diagnostics.reason
    polygon_valid = reason in {"bbox_invalid", "confidence_below_threshold", "accepted"}
    bbox_valid = reason in {"confidence_below_threshold", "accepted"}
    return polygon_valid, bbox_valid


def _service_root_cause(row: dict[str, object], settings: Settings) -> str:
    method = settings.matching_geometry_method
    selected_accepted = bool(
        row.get("hybrid_accepted") if method == "hybrid"
        else row.get("local_accepted") if method == "aliked_lightglue"
        else row.get("dinov2_accepted")
    )
    if bool(row.get("matched_reference_object_produced")) == selected_accepted:
        return "consistent"
    if not bool(row.get("reference_active")):
        return "inactive_reference"
    if method == "dinov2" and bool(row.get("hybrid_accepted")):
        return "service_used_dinov2_config"
    if selected_accepted and bool(row.get("local_refinement_timeout_triggered")):
        return "local_refinement_timeout"
    if selected_accepted:
        return "service_result_missing_unknown"
    return "geometry_rejected_after_diagnostic"


def _finalize_service_row(row, settings, selected_result) -> None:
    method = settings.matching_geometry_method
    selected_accepted = bool(
        row.get("hybrid_accepted") if method == "hybrid"
        else row.get("local_accepted") if method == "aliked_lightglue"
        else row.get("dinov2_accepted")
    )
    diagnostic_actual = (
        float(selected_result.diagnostics.total_seconds)
        if selected_result is not None and method != "dinov2" else 0.0
    )
    timeout = (
        settings.matching_local_refinement_timeout_sec
        if method != "dinov2"
        else min(
            settings.matching_coarse_timeout_seconds,
            settings.matching_reference_timeout_seconds,
        )
    )
    service_produced = bool(row.get("matched_reference_object_produced"))
    actual = float(row.get("service_match_elapsed_sec") or diagnostic_actual)
    timeout = float(row.get("service_timeout_limit_sec") or timeout)
    row.update({
        "matching_geometry_method": method,
        "selected_diagnostic_accepted": selected_accepted,
        "local_refinement_timeout_seconds": timeout,
        "actual_local_refinement_seconds": actual,
        "local_refinement_timeout_triggered": (
            method != "dinov2" and row.get("service_match_outcome") == "timeout"
        ),
        # Production catches TimeoutError and continues to the next reference;
        # it does not invoke its configured DINOv2 fallback in this branch.
        "fallback_to_dinov2_after_timeout": False,
        "service_diagnostic_consistent": selected_accepted == service_produced,
    })
    row["root_cause"] = _service_root_cause(row, settings)
    if service_produced:
        row["accepted"] = True
        row["bbox"] = row["service_final_bbox"]
        row["confidence"] = row["service_final_confidence"]
        row["rejection_reason"] = None
        row["service_final_rejection_reason"] = None
    else:
        row["accepted"] = False
        row["bbox"] = None
        row["confidence"] = None
        reason = (
            "timeout" if row["root_cause"] == "local_refinement_timeout"
            else "inactive_reference" if row["root_cause"] == "inactive_reference"
            else (
                row.get("hybrid_reason") if method == "hybrid"
                else row.get("local_reason") if method == "aliked_lightglue"
                else row.get("dinov2_rejection_reason")
            )
        )
        row["rejection_reason"] = reason
        row["service_final_rejection_reason"] = reason


def save_pair_visualization(path: Path, frame: ImageAsset, reference: ReferenceAsset, row, artifacts: MatchArtifacts) -> None:
    canvas = np.asarray(frame.image).copy()
    if artifacts.matches is not None and not row["accepted"]:
        for point in np.asarray(artifacts.matches.frame_points_px)[:: max(1, len(artifacts.matches.frame_points_px) // 100 or 1)]:
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 2, (0, 255, 255), -1)
    if artifacts.polygon is not None and artifacts.polygon.points is not None:
        polygon = np.rint(artifacts.polygon.points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [polygon], True, (255, 0, 0), 2)
    final_bbox = row.get("bbox")
    if final_bbox is not None:
        x1, y1, x2, y2 = final_bbox
        cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
    label = (
        f"{reference.reference_id} method={row.get('matching_geometry_method')} "
        f"conf={row['confidence'] if row['confidence'] is not None else 'n/a'} "
        f"{row['rejection_reason'] or 'accepted'}"
    )
    cv2.putText(canvas, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"visualization could not be saved: {path}")


async def run_evaluation(
    settings: Settings,
    options: EvaluationOptions,
    *,
    runtime_factory: Callable[[Settings], object] = Dinov2RuntimeRegistry.get,
    local_pipeline_factory: Callable[[Settings], object] = LocalRefinementPipeline,
    emit: Callable[[str], None] = print,
) -> dict[str, object]:
    validate_options(options)
    if not settings.matching_enabled or not settings.matching_dinov2_enabled:
        raise ValueError("production matching and DINOv2 must be enabled by configuration")
    if options.reference is not None:
        reference_paths, frame_paths = [options.reference], [options.frame]
    else:
        assert options.references_dir is not None and options.frames_dir is not None
        reference_paths, frame_paths = _image_paths(options.references_dir), _image_paths(options.frames_dir)
    references = load_references([Path(path) for path in reference_paths])
    frames = [load_asset(Path(path)) for path in frame_paths]
    ground_truth = load_ground_truth(options.ground_truth_csv)
    gt_resolution = ground_truth_resolution(options.ground_truth_csv, references, frames)
    gt_resolution_by_key = {
        (item["normalized_reference_id"], item["normalized_frame_name"]): item
        for item in gt_resolution
    }
    output_dir = options.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_started = time.perf_counter()
    runtime = RecordingRuntime(runtime_factory(settings))
    model_hash = runtime.model_hash
    model_load_seconds = time.perf_counter() - runtime_started
    content = {str(asset.path): asset.content for asset in frames}

    async def local_reader(source: str, _timeout: float) -> bytes:
        if source not in content:
            raise ValueError("only preloaded local evaluation frames are accepted")
        return content[source]

    service_local_pipeline = (
        local_pipeline_factory(settings) if settings.matching_geometry_method != "dinov2" else None
    )
    service = DinoReferenceMatchingService(
        settings,
        runtime_factory=lambda _settings: runtime,
        image_reader=local_reader,
        local_pipeline=service_local_pipeline,
    )
    local_pipeline = local_pipeline_factory(settings)
    evaluator_config = matching_config_snapshot(settings)
    service_config = matching_config_snapshot(service._settings)
    local_cache = ReferenceLocalFeatureCache()
    session_id = "task3-offline-evaluation"
    loaded_count = await service.set_references(
        session_id,
        [
            ReferenceImage(
                object_id=item.object_id, content=item.asset.content, modality=ImageModality.RGB,
                order=index,
                official_reference_url=str(item.asset.path),
                image_url=str(item.asset.path),
                video_name="offline-evaluation",
            )
            for index, item in enumerate(references, start=1)
        ],
        frame_modality=ImageModality.RGB,
    )
    if loaded_count != len(references):
        raise RuntimeError(f"ReferenceStore loaded {loaded_count}/{len(references)} references")
    startup_diagnostics = service.get_startup_diagnostics(session_id)

    detailed: list[dict[str, object]] = []
    performance: list[dict[str, object]] = []
    reference_hashes = {item.asset.sha256 for item in references}
    visualization_paths: list[str] = []
    try:
        import torch
        if runtime.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None

    for frame_index, frame in enumerate(frames):
        before_reference_counts = {value: runtime.count(value) for value in reference_hashes}
        before_frame_count = runtime.count(frame.sha256)
        total_started = time.perf_counter()
        service_started = time.perf_counter()
        frame_context = FrameContext(
            frame_id=f"offline-frame-{frame_index}", image_url=str(frame.path),
            video_name="offline-evaluation", session_id=session_id,
            gps_health_status=None, gps_x=None, gps_y=None, gps_z=None,
            frame_index=frame_index, image_modality=ImageModality.RGB,
        )
        produced = await service.process_frame(frame_context)
        service_seconds = time.perf_counter() - service_started
        service_match_diagnostics = service.get_last_match_diagnostics(
            session_id, frame_context.frame_id
        )
        states = {state.object_id: state for state in await service.get_reference_states(session_id)}
        frame_record = runtime.latest(frame.sha256)
        if frame_record is None:
            raise RuntimeError(f"frame descriptor was not produced: {frame.path.name}")
        coarse_total = homography_total = 0.0
        for reference in references:
            state = states.get(reference.object_id)
            if state is None or state.dense_descriptors is None:
                raise RuntimeError(f"reference descriptor unavailable: {reference.reference_id}")
            pair_started = time.perf_counter()
            diagnostic, artifacts, timings = evaluate_match(
                settings, state.dense_descriptors, frame_record.descriptor
            )
            timings["pair_total_seconds"] = time.perf_counter() - pair_started
            expected = ground_truth.get(
                (_canonical_name(reference.reference_id), _canonical_name(frame.path.name))
            )
            row = _flatten_pair(reference, frame, diagnostic, artifacts, timings, produced, expected)
            service_match = service_match_diagnostics.get(reference.object_id, {})
            resolution = gt_resolution_by_key.get(
                (_canonical_name(reference.reference_id), _canonical_name(frame.path.name)), {}
            )
            reference_record = runtime.latest(reference.asset.sha256)
            reference_prepare_seconds = (
                reference_record.elapsed_seconds if reference_record is not None else 0.0
            )
            row.update({
                "ground_truth_expected_match": expected,
                "gt_reference_resolved": resolution.get("gt_reference_resolved", expected is None),
                "gt_frame_resolved": resolution.get("gt_frame_resolved", expected is None),
                "reference_active": bool(state.is_active(frame_index)),
                "reference_prepare_timeout_seconds": settings.matching_dinov2_timeout_seconds,
                "actual_reference_prepare_seconds": reference_prepare_seconds,
                "reference_prepare_timeout_triggered": (
                    reference_prepare_seconds > settings.matching_dinov2_timeout_seconds
                ),
                "service_timeout_stage": service_match.get("timeout_stage"),
                "service_timeout_limit_sec": service_match.get("timeout_limit_sec"),
                "service_match_elapsed_sec": service_match.get("elapsed_sec"),
                "service_match_outcome": service_match.get("outcome", "not_started"),
                "model_preloaded": service_match.get(
                    "model_preloaded", startup_diagnostics["model_preloaded"]
                ),
                "warmup_completed": service_match.get(
                    "warmup_completed", startup_diagnostics["warmup_completed"]
                ),
                "warmup_time_sec": service_match.get(
                    "warmup_time_sec", startup_diagnostics["warmup_time_sec"]
                ),
                "reference_cache_warmed": service_match.get(
                    "reference_cache_warmed", startup_diagnostics["reference_cache_warmed"]
                ),
                "reference_prepare_time_sec": service_match.get(
                    "reference_prepare_time_sec", startup_diagnostics["reference_prepare_time_sec"]
                ),
                "frame_local_refinement_time_sec": service_match.get(
                    "frame_local_refinement_time_sec"
                ),
                "dinov2_coarse_correspondence_count": diagnostic.get("correspondence_count", 0),
                "dinov2_coarse_mean_similarity": (diagnostic.get("similarity") or {}).get("mean"),
                "dinov2_coarse_spatial_coverage": diagnostic.get("spatial_coverage"),
                "dinov2_homography_inliers": diagnostic.get("inlier_count"),
                "dinov2_homography_inlier_ratio": diagnostic.get("inlier_ratio"),
            })
            row["descriptor_reference_count"] = state.dense_descriptors.shape[0]
            row["descriptor_frame_count"] = frame_record.descriptor.shape[0]
            row["descriptor_dimension"] = frame_record.descriptor.descriptor_dim
            row["pair_total_seconds"] = timings["pair_total_seconds"]
            row["dinov2_accepted"] = bool(row["accepted"])
            comparison = {}
            for geometry_method, prefix in (
                ("aliked_lightglue", "local"),
                ("hybrid", "hybrid"),
            ):
                try:
                    local_result = local_pipeline.match_reference(
                        method=geometry_method,
                        object_id=reference.object_id,
                        reference_descriptor=state.dense_descriptors,
                        frame_descriptor=frame_record.descriptor,
                        reference_image=reference.asset.image,
                        frame_image=frame.image,
                        reference_hash=reference.asset.sha256,
                        frame_hash=frame.sha256,
                        cache=local_cache,
                    )
                    comparison[prefix] = local_result
                except (LocalArtifactUnavailable, LocalFeatureError) as exc:
                    comparison[prefix] = None
                    row[f"{prefix}_reason"] = f"local_refinement_unavailable:{type(exc).__name__}"
            diagnostic_source = next(
                (value.diagnostics for value in (comparison.get("hybrid"), comparison.get("local")) if value is not None),
                None,
            )
            if diagnostic_source is not None:
                for field in (
                    "coarse_correspondence_count", "coarse_similarity", "coarse_coverage",
                    "aliked_reference_keypoints", "aliked_frame_keypoints", "lightglue_match_count",
                    "lightglue_mean_score", "local_homography_inliers",
                    "local_homography_inlier_ratio", "local_homography_rms",
                    "aliked_reference_seconds", "aliked_frame_seconds", "lightglue_seconds",
                ):
                    row[field] = getattr(diagnostic_source, field)
                row["local_homography_seconds"] = diagnostic_source.homography_seconds
                row["hybrid_total_seconds"] = diagnostic_source.total_seconds
                row["local_device"] = diagnostic_source.device
            for prefix in ("local", "hybrid"):
                value = comparison.get(prefix)
                row[f"{prefix}_accepted"] = bool(value and value.matched is not None)
                if value is not None:
                    row[f"{prefix}_reason"] = value.diagnostics.reason
                    row[f"{prefix}_bbox"] = (
                        [
                            value.matched.top_left_x, value.matched.top_left_y,
                            value.matched.bottom_right_x, value.matched.bottom_right_y,
                        ] if value.matched is not None else None
                    )
                    row[f"{prefix}_confidence"] = (
                        value.matched.confidence if value.matched is not None else None
                    )
                    polygon_valid, bbox_valid = _local_geometry_flags(value)
                    row[f"{prefix}_polygon_valid"] = polygon_valid
                    row[f"{prefix}_bbox_valid"] = bbox_valid
            local_value = comparison.get("local")
            hybrid_value = comparison.get("hybrid")
            local_diagnostic = local_value.diagnostics if local_value is not None else None
            hybrid_diagnostic = hybrid_value.diagnostics if hybrid_value is not None else None
            row.update({
                "local_reference_keypoints": getattr(local_diagnostic, "aliked_reference_keypoints", 0),
                "local_frame_keypoints": getattr(local_diagnostic, "aliked_frame_keypoints", 0),
                "local_match_count": getattr(local_diagnostic, "lightglue_match_count", 0),
                "local_mean_match_score": getattr(local_diagnostic, "lightglue_mean_score", 0.0),
                "local_cache_status": getattr(local_diagnostic, "cache_status", "unavailable"),
                "local_rejection_reason": getattr(local_diagnostic, "reason", "unavailable"),
                "hybrid_coarse_gate_passed": bool(
                    hybrid_diagnostic is not None
                    and hybrid_diagnostic.reason not in {
                        "no_mutual_match", "below_minimum_correspondences",
                        "low_mean_similarity", "low_spatial_coverage",
                    }
                ),
                "hybrid_local_refinement_called": bool(
                    hybrid_diagnostic is not None
                    and hybrid_diagnostic.aliked_frame_keypoints > 0
                ),
                "hybrid_local_match_count": getattr(hybrid_diagnostic, "lightglue_match_count", 0),
                "hybrid_local_homography_inliers": getattr(hybrid_diagnostic, "local_homography_inliers", 0),
                "hybrid_local_homography_inlier_ratio": getattr(hybrid_diagnostic, "local_homography_inlier_ratio", 0.0),
                "hybrid_local_homography_rms": getattr(hybrid_diagnostic, "local_homography_rms", None),
                "hybrid_cache_status": getattr(hybrid_diagnostic, "cache_status", "unavailable"),
                "hybrid_rejection_reason": getattr(hybrid_diagnostic, "reason", "unavailable"),
            })
            selected_result = (
                hybrid_value if settings.matching_geometry_method == "hybrid"
                else local_value if settings.matching_geometry_method == "aliked_lightglue"
                else None
            )
            _finalize_service_row(row, settings, selected_result)
            detailed.append(row)
            coarse_total += float(row["coarse_matching_seconds"])
            homography_total += float(row["homography_seconds"])
            if options.save_visualizations:
                status = "accepted" if row["matched_reference_object_produced"] else "rejected"
                target = output_dir / "visualizations" / f"{frame.path.stem}__{reference.reference_id}__{status}.jpg"
                save_pair_visualization(target, frame, reference, row, artifacts)
                visualization_paths.append(str(target))
        reference_preparation = sum(
            sum(record.elapsed_seconds for record in runtime.records[value][before_reference_counts[value] :])
            for value in reference_hashes
        )
        frame_events = runtime.records[frame.sha256][before_frame_count:]
        performance.append({
            "frame_name": frame.path.name,
            "reference_preparation_seconds": reference_preparation,
            "frame_descriptor_seconds": sum(record.elapsed_seconds for record in frame_events),
            "coarse_matching_seconds": coarse_total,
            "homography_seconds": homography_total,
            "service_seconds": service_seconds,
            "total_seconds": time.perf_counter() - total_started,
        })

    cache_before = {value: runtime.count(value) for value in reference_hashes}
    frame_cache_before = runtime.count(frames[0].sha256)
    await service.process_frame(
        FrameContext(
            frame_id="offline-cache-check", image_url=str(frames[0].path),
            video_name="offline-evaluation", session_id=session_id,
            gps_health_status=None, gps_x=None, gps_y=None, gps_z=None,
            frame_index=0, image_modality=ImageModality.RGB,
        )
    )
    cache_after = {value: runtime.count(value) for value in reference_hashes}
    frame_cache_after = runtime.count(frames[0].sha256)
    cache_report = {
        "first_request": "MISS",
        "second_request": "HIT" if cache_before == cache_after else "MISS",
        "reference_forward_repeated": cache_before != cache_after,
        "frame_descriptor_persistently_cached": frame_cache_after == frame_cache_before,
        "frame_forward_repeated": frame_cache_after == frame_cache_before + 1,
        "reference_extract_counts": cache_after,
    }

    summaries = reference_summary(references, detailed)
    metrics = confusion_metrics(detailed)
    sensitivity = threshold_sensitivity(settings, detailed)
    performance_summary = {
        field: _stats([float(row[field]) for row in performance])
        for field in PERFORMANCE_COLUMNS[1:]
    }
    crop_summary = {}
    for crop in ("tight", "medium", "wide", "unspecified"):
        members = [row for row in summaries if row["crop_type"] == crop]
        if members:
            crop_summary[crop] = {
                "reference_count": len(members),
                "accepted_rate": statistics.fmean(float(row["accepted_rate"]) for row in members),
                "mean_correspondences": statistics.fmean(float(row["mean_correspondences"]) for row in members),
                "mean_inlier_ratio": statistics.fmean(float(row["mean_inlier_ratio"]) for row in members if row["mean_inlier_ratio"] is not None) if any(row["mean_inlier_ratio"] is not None for row in members) else None,
            }
    gpu_memory = None
    if runtime.device == "cuda" and torch is not None:
        gpu_memory = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    timeout_sensitivity = local_timeout_sensitivity(detailed)
    smallest_safe_timeout_candidate = next(
        (
            row["timeout_sec"] for row in timeout_sensitivity
            if row["timeout_count"] == 0
            and row["false_positive"] == 0
            and row["false_negative"] == 0
        ),
        None,
    )
    smallest_safe_timeout_observed = next(
        (
            row["timeout_sec"] for row in timeout_sensitivity
            if row["mode"] == "observed production run"
            and row["timeout_count"] == 0
            and row["false_positive"] == 0
            and row["false_negative"] == 0
        ),
        None,
    )
    report = {
        "mode": "single" if options.reference is not None else "directory",
        "production_components": [
            "Dinov2RuntimeRegistry", "DinoReferenceMatchingService", "ReferenceStore",
            "CoarseMatcher", "HomographyEstimator", "ProjectedPolygonValidator",
            "ProjectedBoundingBoxValidator", "ConfidenceScorer", "AlikedRuntimeRegistry",
            "LightGlueRuntimeRegistry", "LocalRefinementPipeline",
        ],
        "artifacts": inspect_artifacts(settings),
        "model": {
            "loaded": bool(runtime.is_loaded), "device": runtime.device,
            "model_hash_short": str(model_hash)[:12], "load_seconds": model_load_seconds,
            "gpu_peak_memory": gpu_memory,
        },
        "references": [
            {
                "reference_id": item.reference_id, "object_id": item.object_id,
                "crop_type": item.crop_type, "reference_path": str(item.asset.path),
                "width": item.asset.width, "height": item.asset.height, "sha256": item.asset.sha256,
            }
            for item in references
        ],
        "frames": [
            {"frame_name": item.path.name, "frame_path": str(item.path), "width": item.width, "height": item.height, "sha256": item.sha256}
            for item in frames
        ],
        "detailed": detailed,
        "reference_summary": summaries,
        "crop_sensitivity": crop_summary,
        "ground_truth_metrics": metrics,
        "feature_distributions": feature_distributions(detailed),
        "threshold_sensitivity": sensitivity,
        "cache": cache_report,
        "performance": performance,
        "performance_summary": performance_summary,
        "startup_warmup": startup_diagnostics,
        "method_comparison": {
            "dinov2_accepted": sum(bool(row.get("dinov2_accepted")) for row in detailed),
            "aliked_lightglue_accepted": sum(bool(row.get("local_accepted")) for row in detailed),
            "hybrid_accepted": sum(bool(row.get("hybrid_accepted")) for row in detailed),
        },
        "service_consistency": {
            "evaluator_service_config_equal": evaluator_config == service_config,
            "registry_stale_config": evaluator_config != service_config,
            "diagnostic_selected_accepted": sum(
                bool(row.get("selected_diagnostic_accepted")) for row in detailed
            ),
            "production_matched_reference_objects": sum(
                bool(row.get("matched_reference_object_produced")) for row in detailed
            ),
            "timeout_pairs": sum(
                bool(row.get("local_refinement_timeout_triggered")) for row in detailed
            ),
            "root_causes": dict(Counter(str(row.get("root_cause")) for row in detailed)),
        },
        "local_timeout_sensitivity": timeout_sensitivity,
        "smallest_safe_timeout_sec": smallest_safe_timeout_observed,
        "smallest_safe_timeout_candidate_sec": smallest_safe_timeout_candidate,
        "config_snapshot": {
            "evaluator": evaluator_config,
            "production_service": service_config,
        },
        "ground_truth_resolution": gt_resolution,
        "local_performance_summary": {
            field: _stats([float(row[field]) for row in detailed if row.get(field) is not None])
            for field in (
                "aliked_reference_seconds", "aliked_frame_seconds", "lightglue_seconds",
                "local_homography_seconds", "hybrid_total_seconds",
            )
        },
        "reference_cache_hit_performance": {
            field: _stats([
                float(row[field]) for row in detailed
                if row.get("hybrid_cache_status") == "HIT" and row.get(field) is not None
            ])
            for field in (
                "aliked_reference_seconds", "aliked_frame_seconds", "lightglue_seconds",
                "local_homography_seconds", "hybrid_total_seconds",
            )
        },
        "visualizations": visualization_paths,
        "prediction_submission": "DISABLED",
    }
    _write_outputs(output_dir, detailed, summaries, performance, sensitivity, metrics, report)
    _emit_summary(report, emit)
    await service.clear_session(session_id)
    return report


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            if isinstance(payload.get("bbox"), list):
                payload["bbox"] = json.dumps(payload["bbox"])
            writer.writerow(payload)


def _write_outputs(output_dir, detailed, summaries, performance, sensitivity, metrics, report):
    _write_csv(output_dir / "task3_matching_detailed.csv", DETAIL_COLUMNS, detailed)
    _write_csv(output_dir / "task3_matching_summary.csv", SUMMARY_COLUMNS, summaries)
    _write_csv(output_dir / "task3_matching_performance.csv", PERFORMANCE_COLUMNS, performance)
    _write_csv(output_dir / "task3_matching_threshold_sensitivity.csv", SENSITIVITY_COLUMNS, sensitivity)
    if metrics is not None:
        _write_csv(output_dir / "task3_matching_confusion.csv", tuple(metrics), [metrics])
    (output_dir / "task3_matching_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_consistency_outputs(output_dir, detailed, report)


def _write_consistency_outputs(output_dir: Path, detailed, report) -> None:
    _write_csv(
        output_dir / "task3_service_consistency_detailed.csv",
        CONSISTENCY_COLUMNS,
        detailed,
    )
    snapshot = {
        **report["config_snapshot"],
        "evaluator_service_config_equal": report["service_consistency"][
            "evaluator_service_config_equal"
        ],
        "registry_stale_config": report["service_consistency"]["registry_stale_config"],
        "ground_truth_resolution": report["ground_truth_resolution"],
    }
    (output_dir / "task3_service_config_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    consistency = report["service_consistency"]
    lines = [
        "TASK 3 SERVICE CONSISTENCY DEBUG",
        f"Diagnostic selected accepted: {consistency['diagnostic_selected_accepted']}",
        f"Production MatchedReferenceObject: {consistency['production_matched_reference_objects']}",
        f"Evaluator/service config equal: {consistency['evaluator_service_config_equal']}",
        f"Registry stale config: {consistency['registry_stale_config']}",
        f"Timeout pairs: {consistency['timeout_pairs']}",
        f"Root causes: {json.dumps(consistency['root_causes'], ensure_ascii=False, sort_keys=True)}",
        f"Smallest safe timeout (observed): {report['smallest_safe_timeout_sec']}",
        f"Smallest safe timeout candidate (offline replay): {report['smallest_safe_timeout_candidate_sec']}",
        "",
    ]
    lines.append("TIMEOUT SENSITIVITY:")
    for item in report["local_timeout_sensitivity"]:
        lines.append(
            f"{item['timeout_sec']:.1f}s matches={item['successful_production_matches']} "
            f"timeouts={item['timeout_count']} precision={item['precision']} "
            f"recall={item['recall']} f1={item['f1']} accuracy={item['accuracy']}"
        )
    lines.append("")
    for row in detailed:
        if not row.get("selected_diagnostic_accepted"):
            continue
        lines.extend([
            "PAIR:",
            f"reference={row['reference_id']}",
            f"frame={row['frame_name']}",
            f"diagnostic_{row['matching_geometry_method']}_accepted=True",
            f"production_service_match_count={1 if row['matched_reference_object_produced'] else 0}",
            "ROOT_CAUSE:",
            str(row["root_cause"]),
            "",
        ])
    lines.append("Prediction submission: DISABLED")
    (output_dir / "task3_service_consistency_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _emit_summary(report, emit):
    model = report["model"]
    emit("===== TASK 3 MATCHING OFFLINE EVALUATION =====")
    emit(f"Model load: {'OK' if model['loaded'] else 'FAIL'}; device={model['device']}")
    emit(f"References: {len(report['references'])}; Frames: {len(report['frames'])}; Pairs: {len(report['detailed'])}")
    accepted = sum(bool(row["matched_reference_object_produced"]) for row in report["detailed"])
    emit(f"MatchedReferenceObject produced: {accepted}/{len(report['detailed'])}")
    emit(f"Reference cache: first={report['cache']['first_request']}; second={report['cache']['second_request']}; reference_forward_repeated={report['cache']['reference_forward_repeated']}")
    emit(f"Frame descriptor persistent cache: {report['cache']['frame_descriptor_persistently_cached']}")
    startup = report["startup_warmup"]
    emit(
        "Startup warmup: "
        f"model_preloaded={startup['model_preloaded']} "
        f"completed={startup['warmup_completed']} "
        f"warmup={startup['warmup_time_sec']:.6f}s "
        f"reference_cache={startup['reference_cache_warmed']} "
        f"reference_prepare={startup['reference_prepare_time_sec']:.6f}s"
    )
    comparison = report["method_comparison"]
    emit(
        "A/B/C accepted: "
        f"DINOv2={comparison['dinov2_accepted']}; "
        f"ALIKED+LightGlue={comparison['aliked_lightglue_accepted']}; "
        f"Hybrid={comparison['hybrid_accepted']}"
    )
    devices = sorted({str(row.get("local_device")) for row in report["detailed"] if row.get("local_device")})
    emit(f"Local refinement devices: {', '.join(devices) if devices else 'unavailable'}")
    if report["ground_truth_metrics"]:
        metrics = report["ground_truth_metrics"]
        emit(f"GT: precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} accuracy={metrics['accuracy']:.4f}")
    total = report["performance_summary"]["total_seconds"]
    emit(f"Performance total: min={total['minimum']:.6f}s mean={total['mean']:.6f}s p50={total['p50']:.6f}s p95={total['p95']:.6f}s")
    emit("Prediction submission: DISABLED")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = EvaluationOptions(
        output_dir=args.output_dir,
        references_dir=args.references_dir,
        frames_dir=args.frames_dir,
        reference=args.reference,
        frame=args.frame,
        ground_truth_csv=args.ground_truth_csv,
        save_visualizations=args.save_visualizations,
    )
    try:
        asyncio.run(run_evaluation(get_settings(), options))
    except (ValueError, OSError) as exc:
        print(f"Task 3 matching evaluation: FAIL ({type(exc).__name__}: {exc})")
        print("Prediction submission: DISABLED")
        return EXIT_CONFIG
    except Exception as exc:
        print(f"Task 3 matching evaluation: FAIL ({type(exc).__name__}: {exc})")
        print("Prediction submission: DISABLED")
        return EXIT_RUNTIME
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
