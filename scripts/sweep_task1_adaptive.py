from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings, get_settings
from scripts.benchmark_task1_motion import (
    OfflineMotionPairProcessor,
    build_frame_pairs,
    discover_frames,
)
from scripts.evaluate_task1_motion import (
    IGNORE,
    MOVING,
    STATIONARY,
    UNKNOWN,
    associate_objects,
    validate_dataset,
)
from scripts.validate_task1_detection import _configure_local_runtime_state


@dataclass(frozen=True, slots=True)
class SweepOptions:
    images_dir: Path
    labels_dir: Path
    output_dir: Path
    min_iou: float = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ground-truth üzerinde adaptive scene selector eşiklerini offline tara."
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-iou", type=float, default=0.5)
    return parser


async def run_sweep(settings: Settings, options: SweepOptions, *, emit=print):
    images = options.images_dir.expanduser().resolve()
    labels = options.labels_dir.expanduser().resolve()
    output = options.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    annotations, issues = validate_dataset(images, labels)
    _configure_local_runtime_state()
    processor = OfflineMotionPairProcessor(settings, images)
    samples: list[dict[str, object]] = []
    for pair in build_frame_pairs(discover_frames(images)):
        annotation = annotations.get(pair.current.path.stem.casefold())
        if annotation is None:
            continue
        analysis = await processor(pair)
        association = associate_objects(annotation.objects, analysis.vehicles, options.min_iou)
        rows = {int(row["vehicle_index"]): row for row in analysis.rows}
        for position, ground_truth in enumerate(annotation.objects):
            match = association.gt_to_detection.get(position)
            if match is None or ground_truth.motion == IGNORE:
                continue
            sample = dict(rows[match[0]])
            sample["gt_motion"] = ground_truth.motion
            samples.append(sample)

    defaults = (
        settings.detection_motion_adaptive_background_median_max,
        settings.detection_motion_adaptive_background_p90_max,
        settings.detection_motion_adaptive_grid_spread_max,
        settings.detection_motion_adaptive_min_valid_background_ratio,
    )
    candidates = [
        sorted({max(0.0, value * factor) for factor in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)})
        for value in defaults[:3]
    ]
    candidates.append(sorted({max(0.01, min(1.0, defaults[3] * factor))
                              for factor in (0.25, 0.5, 1.0, 1.5)}))
    results = []
    for median_max, p90_max, spread_max, min_ratio in itertools.product(*candidates):
        predictions = []
        selected = {"homography": 0, "homography_bbox": 0, "unknown": 0}
        for sample in samples:
            reliable = (sample.get("homography_valid") is True and
                        float(sample.get("adaptive_valid_background_ratio") or 0) >= min_ratio and
                        sample.get("adaptive_homography_quality") not in {None, "", "low"})
            if not reliable:
                method, prediction = "unknown", UNKNOWN
            else:
                low = (
                    float(sample.get("adaptive_background_residual_median") or 0) <= median_max
                    and float(sample.get("adaptive_background_residual_p90") or 0) <= p90_max
                    and float(sample.get("adaptive_background_grid_spread") or 0) <= spread_max
                )
                method = "homography" if low else "homography_bbox"
                prediction = str(sample[f"{method}_result"])
            selected[method] += 1
            predictions.append((str(sample["gt_motion"]), prediction))
        metrics = _metrics(predictions)
        results.append({
            "background_median_max": median_max,
            "background_p90_max": p90_max,
            "grid_spread_max": spread_max,
            "min_valid_background_ratio": min_ratio,
            **metrics,
            "homography_selected": selected["homography"],
            "homography_bbox_selected": selected["homography_bbox"],
            "unknown_selected": selected["unknown"],
        })
    results.sort(key=lambda row: (-row["macro_f1"], -row["stationary_f1"], -row["coverage"]))
    path = output / "adaptive_threshold_sweep.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(results[0]))
        writer.writeheader()
        writer.writerows(results)
    emit(f"Samples: {len(samples)}; XML validation issues: {len(issues)}")
    emit(f"Best offline candidate: {results[0]}")
    emit("Production config changed: NO")
    emit("Prediction submission: DISABLED")
    return {"samples": len(samples), "best": results[0], "csv": path}


def _metrics(rows: list[tuple[str, str]]) -> dict[str, float | int]:
    def count(gt, pred):
        return sum(item == (gt, pred) for item in rows)
    tp_m, fp_m, fn_m = count(MOVING, MOVING), count(STATIONARY, MOVING), sum(
        gt == MOVING and pred != MOVING for gt, pred in rows)
    tp_s, fp_s, fn_s = count(STATIONARY, STATIONARY), count(MOVING, STATIONARY), sum(
        gt == STATIONARY and pred != STATIONARY for gt, pred in rows)
    def f1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    moving_f1, stationary_f1 = f1(tp_m, fp_m, fn_m), f1(tp_s, fp_s, fn_s)
    decided = sum(pred != UNKNOWN for _, pred in rows)
    correct = sum(gt == pred for gt, pred in rows)
    return {
        "moving_f1": moving_f1,
        "stationary_f1": stationary_f1,
        "macro_f1": (moving_f1 + stationary_f1) / 2,
        "strict_accuracy": correct / len(rows) if rows else 0.0,
        "coverage": decided / len(rows) if rows else 0.0,
        "unknown": len(rows) - decided,
        "false_moving": fp_m,
        "false_stationary": fp_s,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(run_sweep(get_settings(), SweepOptions(
            args.images_dir, args.labels_dir, args.output_dir, args.min_iou)))
    except Exception as exc:
        print(f"Adaptive sweep: FAIL ({type(exc).__name__}: {exc})")
        print("Prediction submission: DISABLED")
        return 10
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
