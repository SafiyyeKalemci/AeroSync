from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings, get_settings
from scripts.validate_task1_detection import LoadedImage, load_local_image
from scripts.validate_task2_localization import process_loaded_sequence

DEFAULT_RATIOS = (0.50, 0.45, 0.40, 0.35, 0.30, 0.25)

DETAIL_COLUMNS = (
    "threshold",
    "frame_index",
    "frame_name",
    "vo_valid",
    "tracked_points",
    "inliers",
    "inlier_ratio",
    "rms_residual",
    "delta_x_px",
    "delta_y_px",
    "delta_yaw_rad",
    "translation_magnitude_px",
    "abs_yaw_rad",
    "cumulative_x_px",
    "cumulative_y_px",
    "cumulative_yaw_rad",
    "translation_jump",
    "yaw_jump",
    "failure_reason",
)

SUMMARY_COLUMNS = (
    "threshold",
    "total_frames",
    "initialization_reset_count",
    "valid_count",
    "invalid_count",
    "valid_ratio",
    "low_quality_count",
    "insufficient_features_count",
    "other_failure_reasons",
    "min_translation_px",
    "max_translation_px",
    "mean_translation_px",
    "median_translation_px",
    "p90_translation_px",
    "p95_translation_px",
    "min_abs_yaw_rad",
    "max_abs_yaw_rad",
    "mean_abs_yaw_rad",
    "median_abs_yaw_rad",
    "p90_abs_yaw_rad",
    "p95_abs_yaw_rad",
    "largest_translation_jump_frame_pair",
    "largest_yaw_jump_frame_pair",
    "translation_jump_count",
    "yaw_jump_count",
    "trajectory_length_px",
    "net_displacement_px",
    "cumulative_yaw_rad",
)


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    images_dir: Path
    output_dir: Path
    ratios: tuple[float, ...] = DEFAULT_RATIOS
    save_visualizations: bool = False


SequenceProcessor = Callable[..., Awaitable[tuple[list[dict[str, object]], list[object], list[object]]]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production AffineVO min-inlier-ratio quality sweep (offline)"
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--save-visualizations", action="store_true")
    return parser


def validate_ratios(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("at least one ratio is required")
    ratios = tuple(float(value) for value in values)
    if not all(math.isfinite(value) and 0 < value <= 1 for value in ratios):
        raise ValueError("all inlier ratios must be finite and in (0, 1]")
    if len(set(ratios)) != len(ratios):
        raise ValueError("duplicate inlier ratios are not allowed")
    return ratios


def discover_and_validate_frames(
    images_dir: Path,
    settings: Settings,
) -> tuple[list[LoadedImage], list[dict[str, object]]]:
    root = images_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("images directory does not exist")
    paths = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp"}
        ),
        key=_natural_key,
    )
    loaded: list[LoadedImage] = []
    rejected: list[dict[str, object]] = []
    expected = (settings.localization_camera_width, settings.localization_camera_height)
    for path in paths:
        image = load_local_image(path)
        actual = (int(image.metadata["width"]), int(image.metadata["height"]))
        if actual != expected:
            rejected.append(
                {
                    "frame_name": path.name,
                    "reason": "camera_resolution_mismatch",
                    "actual_width": actual[0],
                    "actual_height": actual[1],
                    "expected_width": expected[0],
                    "expected_height": expected[1],
                }
            )
            continue
        loaded.append(image)
    if not loaded:
        raise ValueError("no frames match configured camera resolution")
    return loaded, rejected


async def quality_sweep(
    settings: Settings,
    loaded_images: Sequence[LoadedImage],
    ratios: Sequence[float],
    *,
    sequence_processor: SequenceProcessor = process_loaded_sequence,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[float, list[dict[str, object]]]]:
    detailed: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    frames_by_ratio: dict[float, list[dict[str, object]]] = {}
    for ratio in validate_ratios(ratios):
        candidate_settings = replace(settings, localization_min_inlier_ratio=ratio)
        candidate_settings.validate_localization_vo()
        frames, _, _ = await sequence_processor(
            candidate_settings,
            loaded_images,
            session_id=f"task2-vo-quality-{_ratio_suffix(ratio)}",
            video_name="task2-vo-quality",
            respect_enabled=False,
        )
        frames_by_ratio[ratio] = frames
        rows = detailed_rows(ratio, frames)
        mark_mad_jumps(rows)
        detailed.extend(rows)
        summaries.append(summarize_threshold(ratio, rows))
    return detailed, summaries, frames_by_ratio


def detailed_rows(threshold: float, frames: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in frames:
        diagnostic = frame["vo_diagnostics"]
        pose = frame["pose"]
        dx = _number(diagnostic.get("translation_x_px"))
        dy = _number(diagnostic.get("translation_y_px"))
        yaw = _number(diagnostic.get("rotation_yaw_rad"))
        translation = math.hypot(dx, dy) if dx is not None and dy is not None else None
        rows.append(
            {
                "threshold": threshold,
                "frame_index": frame["frame_index"],
                "frame_name": frame["frame_name"],
                "vo_valid": bool(diagnostic["transform_valid"]),
                "tracked_points": diagnostic["tracked_points"],
                "inliers": diagnostic["inlier_count"],
                "inlier_ratio": diagnostic["inlier_ratio"],
                "rms_residual": diagnostic["reprojection_error"],
                "delta_x_px": dx,
                "delta_y_px": dy,
                "delta_yaw_rad": yaw,
                "translation_magnitude_px": translation,
                "abs_yaw_rad": abs(yaw) if yaw is not None else None,
                "cumulative_x_px": pose["cumulative_dx_px"],
                "cumulative_y_px": pose["cumulative_dy_px"],
                "cumulative_yaw_rad": pose["cumulative_yaw_rad"],
                "translation_jump": False,
                "yaw_jump": False,
                "failure_reason": diagnostic["failure_reason"],
            }
        )
    return rows


def mark_mad_jumps(rows: Sequence[dict[str, object]]) -> None:
    valid = [row for row in rows if bool(row["vo_valid"])]
    for field, output in (
        ("translation_magnitude_px", "translation_jump"),
        ("abs_yaw_rad", "yaw_jump"),
    ):
        values = [float(row[field]) for row in valid if row[field] is not None]
        if not values:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(np.asarray(values) - median)))
        limit = median + 3.0 * mad
        for row in valid:
            value = row[field]
            row[output] = bool(value is not None and float(value) > limit)


def numeric_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "max", "mean", "median", "p90", "p95")}
    data = np.asarray(values, dtype=np.float64)
    return {
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
    }


def summarize_threshold(threshold: float, rows: Sequence[dict[str, object]]) -> dict[str, object]:
    valid = [row for row in rows if bool(row["vo_valid"])]
    failures = Counter(str(row["failure_reason"]) for row in rows if not bool(row["vo_valid"]))
    translation = numeric_stats(
        [float(row["translation_magnitude_px"]) for row in valid if row["translation_magnitude_px"] is not None]
    )
    yaw = numeric_stats(
        [float(row["abs_yaw_rad"]) for row in valid if row["abs_yaw_rad"] is not None]
    )
    largest_translation = max(valid, key=lambda row: float(row["translation_magnitude_px"] or 0), default=None)
    largest_yaw = max(valid, key=lambda row: float(row["abs_yaw_rad"] or 0), default=None)
    other = {
        reason: count
        for reason, count in failures.items()
        if reason not in {"initialization_or_reset", "low_quality", "insufficient_features"}
    }
    final = rows[-1] if rows else {}
    return {
        "threshold": threshold,
        "total_frames": len(rows),
        "initialization_reset_count": failures["initialization_or_reset"],
        "valid_count": len(valid),
        "invalid_count": len(rows) - len(valid),
        "valid_ratio": _divide(len(valid), len(rows)),
        "low_quality_count": failures["low_quality"],
        "insufficient_features_count": failures["insufficient_features"],
        "other_failure_reasons": json.dumps(other, sort_keys=True),
        "min_translation_px": translation["min"],
        "max_translation_px": translation["max"],
        "mean_translation_px": translation["mean"],
        "median_translation_px": translation["median"],
        "p90_translation_px": translation["p90"],
        "p95_translation_px": translation["p95"],
        "min_abs_yaw_rad": yaw["min"],
        "max_abs_yaw_rad": yaw["max"],
        "mean_abs_yaw_rad": yaw["mean"],
        "median_abs_yaw_rad": yaw["median"],
        "p90_abs_yaw_rad": yaw["p90"],
        "p95_abs_yaw_rad": yaw["p95"],
        "largest_translation_jump_frame_pair": _pair_label(rows, largest_translation),
        "largest_yaw_jump_frame_pair": _pair_label(rows, largest_yaw),
        "translation_jump_count": sum(bool(row["translation_jump"]) for row in valid),
        "yaw_jump_count": sum(bool(row["yaw_jump"]) for row in valid),
        "trajectory_length_px": sum(float(row["translation_magnitude_px"] or 0) for row in valid),
        "net_displacement_px": math.hypot(
            float(final.get("cumulative_x_px", 0) or 0),
            float(final.get("cumulative_y_px", 0) or 0),
        ),
        "cumulative_yaw_rad": float(final.get("cumulative_yaw_rad", 0) or 0),
    }


def candidate_assessment(summaries: Sequence[dict[str, object]]) -> dict[str, object]:
    if not summaries:
        return {}
    highest = sorted(
        summaries,
        key=lambda row: (float(row["valid_ratio"]), -int(row["translation_jump_count"]) - int(row["yaw_jump_count"])),
        reverse=True,
    )
    def lowest_jump(minimum):
        candidates = [row for row in summaries if float(row["valid_ratio"]) >= minimum]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda row: (
                int(row["translation_jump_count"]) + int(row["yaw_jump_count"]),
                -float(row["valid_ratio"]),
            ),
        )
    scored = []
    for row in summaries:
        valid_count = max(1, int(row["valid_count"]))
        penalty = 0.5 * _divide(int(row["translation_jump_count"]), valid_count) + 0.5 * _divide(int(row["yaw_jump_count"]), valid_count)
        scored.append({**row, "stability_coverage_score": float(row["valid_ratio"]) - penalty})
    compromise = max(scored, key=lambda row: (float(row["stability_coverage_score"]), float(row["valid_ratio"])))
    return {
        "highest_valid_ratio": highest[:3],
        "lowest_jump_valid_ratio_ge_050": lowest_jump(0.50),
        "lowest_jump_valid_ratio_ge_070": lowest_jump(0.70),
        "stability_coverage_compromise": compromise,
        "heuristic": "valid_ratio - 0.5*(translation_jump_count/valid_count) - 0.5*(yaw_jump_count/valid_count)",
        "note": "diagnostic recommendation only; no production setting was changed",
    }


async def run_benchmark(
    settings: Settings,
    options: BenchmarkOptions,
    *,
    emit: Callable[[str], None] = print,
    sequence_processor: SequenceProcessor = process_loaded_sequence,
) -> dict[str, object]:
    ratios = validate_ratios(options.ratios)
    output_dir = options.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded, rejected = discover_and_validate_frames(options.images_dir, settings)
    detailed, summaries, frames_by_ratio = await quality_sweep(
        settings, loaded, ratios, sequence_processor=sequence_processor
    )
    assessment = candidate_assessment(summaries)
    detailed_path = output_dir / "task2_vo_quality_detailed.csv"
    summary_path = output_dir / "task2_vo_quality_summary.csv"
    report_path = output_dir / "task2_vo_quality_report.json"
    _write_csv(detailed_path, DETAIL_COLUMNS, detailed)
    _write_csv(summary_path, SUMMARY_COLUMNS, summaries)
    report = {
        "production_threshold": settings.localization_min_inlier_ratio,
        "production_threshold_changed": False,
        "ratios": list(ratios),
        "camera": {
            "width": settings.localization_camera_width,
            "height": settings.localization_camera_height,
            "fx": settings.localization_camera_fx,
            "fy": settings.localization_camera_fy,
            "cx": settings.localization_camera_cx,
            "cy": settings.localization_camera_cy,
            "compatible_frame_count": len(loaded),
            "rejected_frames": rejected,
        },
        "summaries": summaries,
        "candidate_assessment": assessment,
        "gps_scale": "NOT EVALUATED",
        "accuracy_claim": "NONE; pixel-space stability diagnostics only",
        "prediction_submission": "DISABLED",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if options.save_visualizations:
        for ratio, frames in frames_by_ratio.items():
            _save_trajectory(output_dir / f"trajectory_threshold_{_ratio_suffix(ratio)}.png", ratio, frames)
    _emit_report(settings, summaries, assessment, rejected, detailed_path, summary_path, emit)
    emit("GPS scale: NOT EVALUATED")
    emit("Prediction submission: DISABLED")
    return {
        "detailed": detailed,
        "summaries": summaries,
        "assessment": assessment,
        "rejected_frames": rejected,
        "detailed_csv": detailed_path,
        "summary_csv": summary_path,
        "report_json": report_path,
    }


def _emit_report(settings, summaries, assessment, rejected, detailed_path, summary_path, emit):
    emit(
        f"Camera: configured={settings.localization_camera_width}x{settings.localization_camera_height} "
        f"fx={settings.localization_camera_fx} fy={settings.localization_camera_fy} "
        f"cx={settings.localization_camera_cx} cy={settings.localization_camera_cy}"
    )
    emit(f"Resolution mismatches excluded: {len(rejected)}")
    emit("threshold valid valid_ratio low_quality translation_jumps yaw_jumps median_translation p95_translation")
    for row in summaries:
        emit(
            f"{float(row['threshold']):.2f} {row['valid_count']} {float(row['valid_ratio']):.4f} "
            f"{row['low_quality_count']} {row['translation_jump_count']} {row['yaw_jump_count']} "
            f"{_fmt(row['median_translation_px'])} {_fmt(row['p95_translation_px'])}"
        )
    compromise = assessment.get("stability_coverage_compromise")
    if compromise:
        emit(
            f"Stability/coverage compromise: threshold={compromise['threshold']} "
            f"score={float(compromise['stability_coverage_score']):.6f}"
        )
    emit(f"Heuristic: {assessment.get('heuristic')}")
    emit(f"Detailed CSV: {detailed_path}")
    emit(f"Summary CSV: {summary_path}")
    emit("Production threshold changed: NO")


def _save_trajectory(path, ratio, frames):
    canvas = np.full((700, 900, 3), 255, np.uint8)
    points = [(0.0, 0.0)]
    for frame in frames:
        points.append(
            (
                float(frame["pose"]["cumulative_dx_px"]),
                float(frame["pose"]["cumulative_dy_px"]),
            )
        )
    array = np.asarray(points, np.float64)
    span = np.ptp(array, axis=0)
    scale = min(700 / max(span[0], 1.0), 500 / max(span[1], 1.0))
    center = np.asarray([450.0, 350.0]) - (array.min(0) + span / 2) * scale
    pixels = np.round(array * scale + center).astype(int)
    for first, second in zip(pixels, pixels[1:]):
        cv2.line(canvas, tuple(first), tuple(second), (255, 0, 0), 2)
    cv2.putText(canvas, f"Pixel-space VO trajectory; min inlier ratio={ratio:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(canvas, "Not real-world coordinates; not an accuracy measurement", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.imwrite(str(path), canvas)


def _pair_label(rows, row):
    if row is None:
        return None
    position = rows.index(row)
    previous = rows[position - 1]["frame_name"] if position > 0 else "initialization"
    return f"{previous}->{row['frame_name']}"


def _write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _natural_key(item):
    return tuple(int(part) if part.isdigit() else part.casefold() for part in __import__("re").split(r"(\d+)", item.name))


def _ratio_suffix(ratio):
    return f"{round(float(ratio) * 100):03d}"


def _fmt(value):
    return "n/a" if value is None else f"{float(value):.4f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = BenchmarkOptions(
        args.images_dir,
        args.output_dir,
        tuple(args.ratios),
        args.save_visualizations,
    )
    try:
        asyncio.run(run_benchmark(get_settings(), options))
    except Exception as exc:
        print(f"Task 2 VO quality benchmark: FAIL ({type(exc).__name__}: {exc})")
        print("GPS scale: NOT EVALUATED")
        print("Prediction submission: DISABLED")
        return 20
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
