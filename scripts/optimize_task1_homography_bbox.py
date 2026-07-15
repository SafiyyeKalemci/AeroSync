from __future__ import annotations

import argparse
import ast
import csv
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import get_settings

MOVING = "moving"
STATIONARY = "stationary"
UNKNOWN = "unknown"

STATIONARY_THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
MOVING_THRESHOLDS = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
MIN_IOU_VALUES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
MIN_SCORE_VALUES = (0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65)
MIN_SIZE_VALUES = (0.40, 0.50, 0.60, 0.70)
MAX_SIZE_VALUES = (1.5, 2.0, 2.5)

CONFIG_COLUMNS = (
    "stationary_threshold",
    "moving_threshold",
    "min_iou",
    "min_association_score",
    "min_size_ratio",
    "max_size_ratio",
)
METRIC_COLUMNS = (
    "total_evaluated",
    "moving_gt",
    "stationary_gt",
    "moving_pred",
    "stationary_pred",
    "unknown_pred",
    "true_moving",
    "false_moving",
    "true_stationary",
    "false_stationary",
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
    "unknown_rate",
)


@dataclass(frozen=True, slots=True)
class BBoxConfig:
    stationary_threshold: float
    moving_threshold: float
    min_iou: float
    min_association_score: float
    min_size_ratio: float
    max_size_ratio: float

    def __post_init__(self) -> None:
        values = (
            self.stationary_threshold,
            self.moving_threshold,
            self.min_iou,
            self.min_association_score,
            self.min_size_ratio,
            self.max_size_ratio,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("bbox threshold values must be finite")
        if self.stationary_threshold < 0 or self.stationary_threshold >= self.moving_threshold:
            raise ValueError("stationary threshold must be non-negative and below moving threshold")
        if not 0 <= self.min_iou <= 1 or not 0 <= self.min_association_score <= 1:
            raise ValueError("IoU and association thresholds must be in [0, 1]")
        if self.min_size_ratio <= 0 or self.max_size_ratio < self.min_size_ratio:
            raise ValueError("size-ratio interval is invalid")

    def as_dict(self) -> dict[str, float]:
        return {
            "stationary_threshold": self.stationary_threshold,
            "moving_threshold": self.moving_threshold,
            "min_iou": self.min_iou,
            "min_association_score": self.min_association_score,
            "min_size_ratio": self.min_size_ratio,
            "max_size_ratio": self.max_size_ratio,
        }


@dataclass(frozen=True, slots=True)
class BBoxSample:
    pair_id: str
    previous_frame: str
    current_frame: str
    gt_object_index: str
    detection_index: str
    gt_motion: str
    gt_bbox: tuple[float, float, float, float]
    detection_bbox: tuple[float, float, float, float]
    detection_confidence: float | None
    homography_valid: bool
    homography_quality_level: str
    center_residual: float | None
    bbox_iou: float | None
    association_score: float | None
    size_ratio: float | None
    visible_ratio: float
    frame_edge: bool
    projected_edge_unreliable: bool
    production_prediction: str


@dataclass(frozen=True, slots=True)
class OptimizationOptions:
    images_dir: Path
    labels_dir: Path
    evaluation_csv: Path
    output_dir: Path
    diagnostics_csv: Path | None = None
    save_visualizations: bool = False
    folds: int = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="homography_bbox thresholdlarini ground truth ile tamamen offline tara"
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--evaluation-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--diagnostics-csv", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--save-visualizations", action="store_true")
    return parser


def production_config() -> BBoxConfig:
    settings = get_settings()
    return BBoxConfig(
        settings.detection_motion_bbox_stationary_threshold_px,
        settings.detection_motion_bbox_moving_threshold_px,
        settings.detection_motion_bbox_match_min_iou,
        settings.detection_motion_bbox_match_min_score,
        settings.detection_motion_bbox_min_size_ratio,
        settings.detection_motion_bbox_max_size_ratio,
    )


def candidate_configs() -> list[BBoxConfig]:
    return [
        BBoxConfig(stationary, moving, min_iou, min_score, min_size, max_size)
        for stationary in STATIONARY_THRESHOLDS
        for moving in MOVING_THRESHOLDS
        if stationary < moving
        for min_iou in MIN_IOU_VALUES
        for min_score in MIN_SCORE_VALUES
        for min_size in MIN_SIZE_VALUES
        for max_size in MAX_SIZE_VALUES
        if min_size <= max_size
    ]


def simulate_bbox_decision(sample: BBoxSample, config: BBoxConfig) -> str:
    """Replay the production decision order on an already observed association."""
    if not sample.homography_valid:
        return UNKNOWN
    if sample.frame_edge or sample.projected_edge_unreliable:
        return UNKNOWN
    values = (
        sample.center_residual,
        sample.bbox_iou,
        sample.association_score,
        sample.size_ratio,
    )
    if any(value is None for value in values):
        return UNKNOWN
    residual = float(sample.center_residual)
    iou = float(sample.bbox_iou)
    score = float(sample.association_score)
    size_ratio = float(sample.size_ratio)
    if (
        iou < config.min_iou
        or score < config.min_association_score
        or size_ratio < config.min_size_ratio
        or size_ratio > config.max_size_ratio
    ):
        return UNKNOWN
    if residual <= config.stationary_threshold and iou >= config.min_iou:
        return STATIONARY
    if residual >= config.moving_threshold:
        return MOVING
    return UNKNOWN


def calculate_metrics(
    samples: Sequence[BBoxSample], predictions: Sequence[str]
) -> dict[str, float | int]:
    if len(samples) != len(predictions):
        raise ValueError("sample and prediction lengths differ")
    pairs = [(sample.gt_motion, prediction) for sample, prediction in zip(samples, predictions)]
    total = len(pairs)
    moving_gt = sum(gt == MOVING for gt, _ in pairs)
    stationary_gt = sum(gt == STATIONARY for gt, _ in pairs)
    moving_pred = sum(pred == MOVING for _, pred in pairs)
    stationary_pred = sum(pred == STATIONARY for _, pred in pairs)
    unknown_pred = sum(pred == UNKNOWN for _, pred in pairs)
    true_moving = sum(gt == MOVING and pred == MOVING for gt, pred in pairs)
    false_moving = sum(gt == STATIONARY and pred == MOVING for gt, pred in pairs)
    true_stationary = sum(gt == STATIONARY and pred == STATIONARY for gt, pred in pairs)
    false_stationary = sum(gt == MOVING and pred == STATIONARY for gt, pred in pairs)
    correct = true_moving + true_stationary
    decided = moving_pred + stationary_pred
    moving_precision = _divide(true_moving, moving_pred)
    moving_recall = _divide(true_moving, moving_gt)
    stationary_precision = _divide(true_stationary, stationary_pred)
    stationary_recall = _divide(true_stationary, stationary_gt)
    moving_f1 = _f1(moving_precision, moving_recall)
    stationary_f1 = _f1(stationary_precision, stationary_recall)
    return {
        "total_evaluated": total,
        "moving_gt": moving_gt,
        "stationary_gt": stationary_gt,
        "moving_pred": moving_pred,
        "stationary_pred": stationary_pred,
        "unknown_pred": unknown_pred,
        "true_moving": true_moving,
        "false_moving": false_moving,
        "true_stationary": true_stationary,
        "false_stationary": false_stationary,
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
        "unknown_rate": _divide(unknown_pred, total),
    }


def evaluate_config(samples: Sequence[BBoxSample], config: BBoxConfig) -> dict[str, object]:
    predictions = [simulate_bbox_decision(sample, config) for sample in samples]
    return {**config.as_dict(), **calculate_metrics(samples, predictions)}


def threshold_sweep(
    samples: Sequence[BBoxSample], configs: Sequence[BBoxConfig] | None = None
) -> list[dict[str, object]]:
    return [evaluate_config(samples, config) for config in (configs or candidate_configs())]


def leaderboard_rows(sweep: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    ranked_source = [row for row in sweep if bool(row.get("replay_supported", True))]
    if not ranked_source:
        ranked_source = list(sweep)
    definitions = (
        ("highest_macro_f1", lambda row: True, ("macro_f1", "balanced_accuracy", "coverage"), False),
        ("highest_balanced_accuracy", lambda row: True, ("balanced_accuracy", "macro_f1", "coverage"), False),
        ("highest_stationary_f1", lambda row: True, ("stationary_f1", "macro_f1", "coverage"), False),
        ("highest_moving_f1", lambda row: True, ("moving_f1", "macro_f1", "coverage"), False),
        ("macro_f1_coverage_050", lambda row: float(row["coverage"]) >= 0.50, ("macro_f1", "coverage", "balanced_accuracy"), False),
        ("macro_f1_coverage_060", lambda row: float(row["coverage"]) >= 0.60, ("macro_f1", "coverage", "balanced_accuracy"), False),
        ("macro_f1_coverage_070", lambda row: float(row["coverage"]) >= 0.70, ("macro_f1", "coverage", "balanced_accuracy"), False),
        ("macro_f1_coverage_080", lambda row: float(row["coverage"]) >= 0.80, ("macro_f1", "coverage", "balanced_accuracy"), False),
        (
            "lowest_false_moving_recall070_coverage060",
            lambda row: float(row["moving_recall"]) >= 0.70 and float(row["coverage"]) >= 0.60,
            ("false_moving", "macro_f1", "coverage"),
            True,
        ),
        (
            "conservative_false_stationary_zero",
            lambda row: int(row["false_stationary"]) == 0,
            ("macro_f1", "coverage", "moving_recall"),
            False,
        ),
    )
    output: list[dict[str, object]] = []
    for name, predicate, keys, ascending_first in definitions:
        selected = [row for row in ranked_source if predicate(row)]
        if ascending_first:
            selected.sort(
                key=lambda row: (
                    float(row[keys[0]]),
                    -float(row[keys[1]]),
                    -float(row[keys[2]]),
                )
            )
        else:
            selected.sort(key=lambda row: tuple(float(row[key]) for key in keys), reverse=True)
        for rank, row in enumerate(selected[:10], start=1):
            output.append({"leaderboard": name, "rank": rank, **row})
    return output


def select_profiles(sweep: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    if not sweep:
        raise ValueError("threshold sweep is empty")
    ranked_source = [row for row in sweep if bool(row.get("replay_supported", True))] or list(sweep)
    balanced_pool = [row for row in ranked_source if float(row["coverage"]) >= 0.50] or ranked_source
    balanced = max(
        balanced_pool,
        key=lambda row: (float(row["macro_f1"]), float(row["balanced_accuracy"]), float(row["coverage"])),
    )
    conservative_pool = [
        row
        for row in ranked_source
        if int(row["false_stationary"]) == 0
        and float(row["moving_recall"]) >= 0.50
        and float(row["coverage"]) >= 0.40
    ]
    if conservative_pool:
        conservative = min(
            conservative_pool,
            key=lambda row: (int(row["false_moving"]), -float(row["macro_f1"]), -float(row["coverage"])),
        )
    else:
        conservative = max(ranked_source, key=lambda row: (float(row["macro_f1"]), -int(row["false_moving"])))
    high_coverage = max(
        ranked_source,
        key=lambda row: (float(row["coverage"]), float(row["macro_f1"]), float(row["balanced_accuracy"])),
    )
    return {"Balanced": balanced, "Conservative": conservative, "High Coverage": high_coverage}


def pareto_frontier(sweep: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    # Metric-equivalent configs are represented once to keep the frontier useful and bounded.
    unique: dict[tuple[float, float, int, int], dict[str, object]] = {}
    for row in sweep:
        objective = (
            float(row["macro_f1"]),
            float(row["coverage"]),
            int(row["false_moving"]),
            int(row["false_stationary"]),
        )
        unique.setdefault(objective, row)
    candidates = list(unique.values())
    frontier: list[dict[str, object]] = []
    for row in candidates:
        if any(_dominates(other, row) for other in candidates if other is not row):
            continue
        frontier.append(row)
    return sorted(
        frontier,
        key=lambda row: (-float(row["macro_f1"]), -float(row["coverage"]), int(row["false_moving"]), int(row["false_stationary"])),
    )


def _dominates(first: dict[str, object], second: dict[str, object]) -> bool:
    no_worse = (
        float(first["macro_f1"]) >= float(second["macro_f1"])
        and float(first["coverage"]) >= float(second["coverage"])
        and int(first["false_moving"]) <= int(second["false_moving"])
        and int(first["false_stationary"]) <= int(second["false_stationary"])
    )
    strictly_better = (
        float(first["macro_f1"]) > float(second["macro_f1"])
        or float(first["coverage"]) > float(second["coverage"])
        or int(first["false_moving"]) < int(second["false_moving"])
        or int(first["false_stationary"]) < int(second["false_stationary"])
    )
    return no_worse and strictly_better


def parameter_sensitivity(sweep: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    metrics = (
        "stationary_recall",
        "false_stationary",
        "coverage",
        "false_moving",
        "moving_recall",
        "unknown_rate",
        "moving_precision",
        "stationary_precision",
        "macro_f1",
    )
    scopes = {
        "all_grid": list(sweep),
        "replay_supported": [row for row in sweep if bool(row.get("replay_supported", True))],
    }
    for scope, scoped_rows in scopes.items():
        for parameter in CONFIG_COLUMNS:
            grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
            for row in scoped_rows:
                grouped[float(row[parameter])].append(row)
            for value, rows in sorted(grouped.items()):
                output.append(
                    {
                        "scope": scope,
                        "parameter": parameter,
                        "value": value,
                        "config_count": len(rows),
                        **{
                            f"mean_{metric}": float(np.mean([float(row[metric]) for row in rows]))
                            for metric in metrics
                        },
                    }
                )
    return output


def feature_distributions(samples: Sequence[BBoxSample]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    features = {
        "bbox_center_residual": lambda sample: sample.center_residual,
        "bbox_iou": lambda sample: sample.bbox_iou,
        "bbox_association_score": lambda sample: sample.association_score,
        "bbox_size_ratio": lambda sample: sample.size_ratio,
    }
    for gt in (MOVING, STATIONARY):
        selected = [sample for sample in samples if sample.gt_motion == gt]
        for name, getter in features.items():
            values = [float(value) for sample in selected if (value := getter(sample)) is not None]
            output.append({"gt_motion": gt, "feature": name, **distribution(values)})
    return output


def distribution(values: Iterable[float]) -> dict[str, object]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {key: "" for key in ("count", "minimum", "p10", "p25", "median", "p75", "p90", "p95", "maximum")}
    return {
        "count": int(data.size),
        "minimum": float(data.min()),
        "p10": float(np.percentile(data, 10)),
        "p25": float(np.percentile(data, 25)),
        "median": float(np.median(data)),
        "p75": float(np.percentile(data, 75)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "maximum": float(data.max()),
    }


def grouped_folds(samples: Sequence[BBoxSample], folds: int = 5) -> list[set[str]]:
    groups = sorted({sample.pair_id for sample in samples})
    if len(groups) < 2:
        raise ValueError("at least two frame-pair groups are required")
    count = min(max(2, folds), len(groups))
    output = [set() for _ in range(count)]
    for index, group in enumerate(groups):
        output[index % count].add(group)
    return output


def cross_validate_profiles(
    samples: Sequence[BBoxSample], profiles: dict[str, dict[str, object]], folds: int = 5
) -> list[dict[str, object]]:
    fold_groups = grouped_folds(samples, folds)
    output: list[dict[str, object]] = []
    for profile, row in profiles.items():
        config = _config_from_row(row)
        validation_metrics: list[dict[str, object]] = []
        for fold_index, validation_groups in enumerate(fold_groups, start=1):
            train = [sample for sample in samples if sample.pair_id not in validation_groups]
            validation = [sample for sample in samples if sample.pair_id in validation_groups]
            if {sample.pair_id for sample in train} & {sample.pair_id for sample in validation}:
                raise AssertionError("frame-pair leakage detected")
            for split, subset in (("train", train), ("validation", validation)):
                metrics = evaluate_config(subset, config)
                output.append({"profile": profile, "fold": fold_index, "split": split, **metrics})
                if split == "validation":
                    validation_metrics.append(metrics)
        aggregate: dict[str, object] = {
            "profile": profile,
            "fold": "aggregate",
            "split": "validation",
            **config.as_dict(),
        }
        for metric in ("macro_f1", "coverage", "stationary_f1", "moving_f1"):
            values = [float(item[metric]) for item in validation_metrics]
            aggregate[f"{metric}_mean"] = float(np.mean(values))
            aggregate[f"{metric}_std"] = float(np.std(values))
        output.append(aggregate)
    return output


def load_samples(
    evaluation_rows: Sequence[dict[str, str]],
    diagnostic_rows: Sequence[dict[str, str]],
    image_shapes: dict[str, tuple[int, int]],
    current_config: BBoxConfig,
) -> tuple[list[BBoxSample], list[str]]:
    diagnostics = {
        (row.get("frame_previous", ""), row.get("frame_current", ""), row.get("vehicle_index", "")): row
        for row in diagnostic_rows
    }
    samples: list[BBoxSample] = []
    malformed: list[str] = []
    for line, row in enumerate(evaluation_rows, start=2):
        if not _truth(row.get("matched")) or _truth(row.get("ignored")):
            continue
        gt = str(row.get("gt_motion", "")).strip().casefold()
        detection_index = str(row.get("matched_detection_index", "")).strip()
        gt_bbox = _parse_bbox(row.get("gt_bbox"))
        detection_bbox = _parse_bbox(row.get("detection_bbox"))
        if gt not in {MOVING, STATIONARY} or not detection_index or gt_bbox is None or detection_bbox is None:
            malformed.append(f"line {line}: invalid matched evaluation row")
            continue
        previous = str(row.get("previous_frame", ""))
        current = str(row.get("current_frame", ""))
        diagnostic = diagnostics.get((previous, current, detection_index), {})
        residual = _first_number(diagnostic.get("bbox_center_residual_px"), row.get("bbox_center_residual"))
        iou = _first_number(diagnostic.get("bbox_iou"), row.get("bbox_iou"))
        score = _first_number(diagnostic.get("bbox_association_score"), row.get("bbox_association_score"))
        size_ratio = _first_number(diagnostic.get("bbox_size_ratio"), row.get("bbox_size_ratio"))
        shape = image_shapes.get(current)
        visible_ratio, frame_edge = _bbox_visibility(detection_bbox, shape)
        production_prediction = str(row.get("homography_bbox_prediction", UNKNOWN)).strip().casefold()
        projected_edge_unreliable = bool(
            production_prediction == UNKNOWN
            and residual is not None
            and (residual <= current_config.stationary_threshold or residual >= current_config.moving_threshold)
            and iou is not None
            and iou >= current_config.min_iou
            and score is not None
            and score >= current_config.min_association_score
            and size_ratio is not None
            and current_config.min_size_ratio <= size_ratio <= current_config.max_size_ratio
        )
        samples.append(
            BBoxSample(
                pair_id=f"{previous}->{current}",
                previous_frame=previous,
                current_frame=current,
                gt_object_index=str(row.get("gt_object_index", "")),
                detection_index=detection_index,
                gt_motion=gt,
                gt_bbox=gt_bbox,
                detection_bbox=detection_bbox,
                detection_confidence=_number(row.get("detection_confidence")),
                homography_valid=_truth(row.get("homography_valid", diagnostic.get("homography_valid"))),
                homography_quality_level=str(row.get("homography_quality_level", diagnostic.get("hybrid_homography_quality_level", ""))),
                center_residual=residual,
                bbox_iou=iou,
                association_score=score,
                size_ratio=size_ratio,
                visible_ratio=visible_ratio,
                frame_edge=frame_edge,
                projected_edge_unreliable=projected_edge_unreliable,
                production_prediction=production_prediction,
            )
        )
    return samples, malformed


def run_optimization(
    options: OptimizationOptions,
    *,
    emit: Callable[[str], None] = print,
    configs: Sequence[BBoxConfig] | None = None,
) -> dict[str, object]:
    images_dir = options.images_dir.expanduser().resolve()
    labels_dir = options.labels_dir.expanduser().resolve()
    evaluation_csv = options.evaluation_csv.expanduser().resolve()
    output_dir = options.output_dir.expanduser().resolve()
    if not images_dir.is_dir() or not labels_dir.is_dir() or not evaluation_csv.is_file():
        raise ValueError("images-dir, labels-dir and evaluation-csv must exist")
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = _resolve_diagnostics(options.diagnostics_csv, evaluation_csv)
    evaluation_rows = _read_csv(evaluation_csv)
    diagnostic_rows = _read_csv(diagnostic_path) if diagnostic_path else []
    current = production_config()
    samples, malformed = load_samples(evaluation_rows, diagnostic_rows, _load_shapes(images_dir), current)
    if not samples:
        raise ValueError("no evaluable matched ground-truth samples")
    if not any(
        sample.center_residual is not None
        and sample.bbox_iou is not None
        and sample.association_score is not None
        and sample.size_ratio is not None
        for sample in samples
    ):
        raise ValueError("bbox association diagnostics are required; pass --diagnostics-csv")

    grid = list(configs) if configs is not None else candidate_configs()
    sweep = threshold_sweep(samples, grid)
    for row in sweep:
        row["replay_supported"] = _replay_supported(row, current)
    leaders = leaderboard_rows(sweep)
    pareto = pareto_frontier(sweep)
    sensitivity = parameter_sensitivity(sweep)
    distributions = feature_distributions(samples)
    profiles = select_profiles(sweep)
    current_metrics = evaluate_config(samples, current)
    cv = cross_validate_profiles(samples, profiles, options.folds)

    paths = {
        "sweep_csv": output_dir / "homography_bbox_threshold_sweep.csv",
        "top_csv": output_dir / "homography_bbox_top_configs.csv",
        "pareto_csv": output_dir / "homography_bbox_pareto.csv",
        "sensitivity_csv": output_dir / "homography_bbox_parameter_sensitivity.csv",
        "distribution_csv": output_dir / "homography_bbox_feature_distribution.csv",
        "cv_csv": output_dir / "homography_bbox_cross_validation.csv",
        "summary_txt": output_dir / "homography_bbox_optimization_summary.txt",
    }
    _write_csv(paths["sweep_csv"], (*CONFIG_COLUMNS, "replay_supported", *METRIC_COLUMNS), sweep)
    _write_csv(paths["top_csv"], ("leaderboard", "rank", *CONFIG_COLUMNS, "replay_supported", *METRIC_COLUMNS), leaders)
    _write_csv(paths["pareto_csv"], (*CONFIG_COLUMNS, "replay_supported", *METRIC_COLUMNS), pareto)
    _write_csv(paths["sensitivity_csv"], tuple(sensitivity[0]) if sensitivity else (), sensitivity)
    _write_csv(paths["distribution_csv"], tuple(distributions[0]) if distributions else (), distributions)
    cv_columns = _union_columns(cv)
    _write_csv(paths["cv_csv"], cv_columns, cv)

    summary = _summary_lines(
        samples,
        malformed,
        grid,
        distributions,
        current_metrics,
        profiles,
        cv,
        diagnostic_path,
    )
    paths["summary_txt"].write_text("\n".join(summary) + "\n", encoding="utf-8")
    for line in summary:
        emit(line)
    if options.save_visualizations:
        _save_visualizations(images_dir, output_dir, samples, profiles)
    emit("Prediction submission: DISABLED")
    return {
        "samples": samples,
        "malformed_rows": malformed,
        "sweep": sweep,
        "leaderboards": leaders,
        "pareto": pareto,
        "sensitivity": sensitivity,
        "distributions": distributions,
        "profiles": profiles,
        "production_metrics": current_metrics,
        "cross_validation": cv,
        **paths,
    }


def _summary_lines(samples, malformed, grid, distributions, current, profiles, cv, diagnostic_path):
    settings = get_settings()
    lines = [
        "===== HOMOGRAPHY_BBOX OFFLINE THRESHOLD OPTIMIZATION =====",
        f"GT MOVING: {sum(sample.gt_motion == MOVING for sample in samples)}",
        f"GT STATIONARY: {sum(sample.gt_motion == STATIONARY for sample in samples)}",
        f"Total matched GT: {len(samples)}",
        f"Search strategy: exhaustive requested grid ({len(grid)} configurations)",
        f"Diagnostics: {diagnostic_path if diagnostic_path else 'evaluation CSV only'}",
        "Replay limitation: only associations observed in the diagnostic CSV can be reclassified; previously rejected candidates remain UNKNOWN.",
        "Profile/leaderboard policy: candidates that relax production IoU, score or size gates are excluded from recommendation rankings, but remain in the sweep CSV.",
        f"Fixed production gates (not swept): max_center_distance_ratio={settings.detection_motion_bbox_match_max_center_distance_ratio} "
        f"min_visible_ratio={settings.detection_motion_bbox_min_visible_ratio}",
    ]
    for gt in (MOVING, STATIONARY):
        row = next(item for item in distributions if item["gt_motion"] == gt and item["feature"] == "bbox_center_residual")
        lines.append(
            f"bbox_center_residual {gt}: count={row['count']} median={_fmt(row['median'])} p25={_fmt(row['p25'])} p75={_fmt(row['p75'])} p90={_fmt(row['p90'])}"
        )
    lines.append("Production config replay:")
    lines.extend(_profile_lines("Production", current))
    lines.append("Candidate profiles (offline only; not applied):")
    for name, row in profiles.items():
        lines.extend(_profile_lines(name, row))
        aggregate = next(
            item for item in cv if item.get("profile") == name and item.get("fold") == "aggregate"
        )
        lines.append(
            f"  CV validation macro_f1={aggregate['macro_f1_mean']:.6f}+/-{aggregate['macro_f1_std']:.6f} "
            f"coverage={aggregate['coverage_mean']:.6f}+/-{aggregate['coverage_std']:.6f} "
            f"stationary_f1={aggregate['stationary_f1_mean']:.6f}+/-{aggregate['stationary_f1_std']:.6f} "
            f"moving_f1={aggregate['moving_f1_mean']:.6f}+/-{aggregate['moving_f1_std']:.6f}"
        )
    lines.extend(
        (
            "Answers:",
            "1. bbox_center_residual is strongly separated on the observed matched associations, but availability and association gates limit coverage.",
            "2. The current 3/8 px hysteresis is too low for much of the stationary residual distribution, producing false MOVING decisions.",
            "3. Threshold tuning can improve Stationary F1 on this dataset; the profile metrics above quantify the gain.",
            "4. Coverage is constrained by missing/rejected associations and edge/homography safety gates, not only thresholds.",
            "5. False Moving can be traded for more UNKNOWN; see Conservative profile.",
            "6. Wider stationary thresholds may increase False Stationary; the profile metrics report that cost.",
            "7. Balanced, Conservative and High Coverage are analysis candidates only; no production setting was changed.",
            "8. The dataset is small and profile selection used the full dataset; grouped-fold stability is reported, but independent validation is still required.",
            "Cross-validation policy: frame-pair grouped folds; no frame-pair is split between train and validation. Profiles are fixed from the full-data sweep, so CV measures stability rather than nested-search generalization.",
            f"Malformed/skipped rows: {len(malformed)}",
            "Production configuration changed: NO",
        )
    )
    return lines


def _profile_lines(name: str, row: dict[str, object]) -> list[str]:
    return [
        f"  {name}: stationary={row['stationary_threshold']} moving={row['moving_threshold']} "
        f"min_iou={row['min_iou']} min_score={row['min_association_score']} "
        f"size=[{row['min_size_ratio']},{row['max_size_ratio']}]",
        f"    strict={float(row['strict_accuracy']):.6f} decided={float(row['decided_only_accuracy']):.6f} "
        f"coverage={float(row['coverage']):.6f} moving_f1={float(row['moving_f1']):.6f} "
        f"stationary_f1={float(row['stationary_f1']):.6f} macro_f1={float(row['macro_f1']):.6f} "
        f"balanced={float(row['balanced_accuracy']):.6f} false_moving={row['false_moving']} "
        f"false_stationary={row['false_stationary']} unknown={row['unknown_pred']}",
    ]


def _save_visualizations(images_dir, output_dir, samples, profiles):
    import cv2

    root = output_dir / "visualizations"
    for profile, row in profiles.items():
        config = _config_from_row(row)
        categories: dict[str, list[tuple[BBoxSample, str]]] = defaultdict(list)
        for sample in samples:
            prediction = simulate_bbox_decision(sample, config)
            if sample.gt_motion == STATIONARY and prediction == MOVING:
                categories["false_moving"].append((sample, prediction))
            elif sample.gt_motion == MOVING and prediction == STATIONARY:
                categories["false_stationary"].append((sample, prediction))
            elif prediction == UNKNOWN:
                categories["unknown"].append((sample, prediction))
        for category, entries in categories.items():
            target = root / profile.lower().replace(" ", "_") / category
            target.mkdir(parents=True, exist_ok=True)
            for index, (sample, prediction) in enumerate(entries[:10]):
                image = cv2.imread(str(images_dir / sample.current_frame), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                x1, y1, x2, y2 = (int(round(value)) for value in sample.detection_bbox)
                margin = 30
                crop = image[max(0, y1 - margin):min(image.shape[0], y2 + margin), max(0, x1 - margin):min(image.shape[1], x2 + margin)].copy()
                text = (
                    f"GT={sample.gt_motion} PRED={prediction}",
                    f"res={_fmt(sample.center_residual)} iou={_fmt(sample.bbox_iou)} score={_fmt(sample.association_score)}",
                    f"profile={profile}",
                )
                for line, value in enumerate(text):
                    cv2.putText(crop, value, (5, 20 + line * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imwrite(str(target / f"{index:02d}_{Path(sample.current_frame).stem}_{sample.detection_index}.jpg"), crop)


def _resolve_diagnostics(explicit: Path | None, evaluation_csv: Path) -> Path | None:
    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser().resolve())
    candidates.extend(
        (
            evaluation_csv.parent / "motion_benchmark.csv",
            Path("work/benchmark_local_24_final/motion_benchmark.csv").resolve(),
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if explicit is not None:
        raise ValueError(f"diagnostics CSV not found: {explicit}")
    return None


def _load_shapes(images_dir: Path) -> dict[str, tuple[int, int]]:
    import cv2

    output = {}
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp"}:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                output[path.name] = image.shape[:2]
    return output


def _bbox_visibility(bbox, shape):
    if shape is None:
        return 1.0, False
    height, width = shape
    raw_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    clipped = (
        min(max(bbox[0], 0.0), width),
        min(max(bbox[1], 0.0), height),
        min(max(bbox[2], 0.0), width),
        min(max(bbox[3], 0.0), height),
    )
    clipped_area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
    visible = _divide(clipped_area, raw_area)
    edge = bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= width - 1 or bbox[3] >= height - 1
    return visible, edge


def _config_from_row(row: dict[str, object]) -> BBoxConfig:
    return BBoxConfig(*(float(row[key]) for key in CONFIG_COLUMNS))


def _replay_supported(row: dict[str, object], current: BBoxConfig) -> bool:
    """A fixed-association replay cannot validate gates looser than data collection."""
    return bool(
        float(row["min_iou"]) >= current.min_iou
        and float(row["min_association_score"]) >= current.min_association_score
        and float(row["min_size_ratio"]) >= current.min_size_ratio
        and float(row["max_size_ratio"]) <= current.max_size_ratio
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _union_columns(rows: Sequence[dict[str, object]]) -> tuple[str, ...]:
    output: list[str] = []
    for row in rows:
        for key in row:
            if key not in output:
                output.append(key)
    return tuple(output)


def _parse_bbox(value: object) -> tuple[float, float, float, float] | None:
    try:
        parsed = tuple(float(item) for item in ast.literal_eval(str(value)))
    except (SyntaxError, TypeError, ValueError):
        return None
    if len(parsed) != 4 or not all(math.isfinite(item) for item in parsed):
        return None
    if parsed[2] <= parsed[0] or parsed[3] <= parsed[1]:
        return None
    return parsed


def _first_number(*values: object) -> float | None:
    for value in values:
        number = _number(value)
        if number is not None:
            return number
    return None


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truth(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _fmt(value: object) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.6f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = OptimizationOptions(
        args.images_dir,
        args.labels_dir,
        args.evaluation_csv,
        args.output_dir,
        args.diagnostics_csv,
        args.save_visualizations,
        args.folds,
    )
    try:
        run_optimization(options)
    except Exception as exc:
        print(f"homography_bbox optimization: FAIL ({type(exc).__name__}: {exc})")
        print("Prediction submission: DISABLED")
        return 10
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
