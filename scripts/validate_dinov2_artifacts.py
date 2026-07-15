from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings, get_settings
from app.services.matching.bbox_validator import ProjectedBoundingBoxValidator
from app.services.matching.coarse_matcher import CoarseMatcher
from app.services.matching.dinov2_runtime import Dinov2RuntimeRegistry
from app.services.matching.geometry import ConfidenceScorer, HomographyEstimator, ProjectedPolygonValidator
from app.utils.images import detect_image_format

EXIT_OK = 0
EXIT_ARTIFACT = 10
EXIT_MODEL = 20
EXIT_IMAGE = 30
EXIT_MATCH = 40

RuntimeFactory = Callable[[Settings], object]
ImageLoader = Callable[[Path], tuple[object, dict[str, object]]]


@dataclass(frozen=True, slots=True)
class ValidationOptions:
    image: Path | None = None
    reference: Path | None = None
    frame: Path | None = None
    benchmark_runs: int = 0
    save_visualization: Path | None = None
    json_output: Path | None = None


@dataclass(slots=True)
class MatchArtifacts:
    matches: object | None = None
    homography: object | None = None
    polygon: object | None = None
    bbox: object | None = None
    quality: object | None = None


class ReferenceDescriptorCache:
    """In-process validation cache; frame descriptors are never stored."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], tuple[object, object]] = {}

    def get(self, runtime, image, image_hash: str):
        model_hash = runtime.model_hash
        key = (image_hash, model_hash)
        started = time.perf_counter()
        cached = self._entries.get(key)
        if cached is not None:
            return cached[0], cached[1], True, time.perf_counter() - started
        descriptor, metrics = runtime.extract(image, image_hash)
        self._entries[key] = (descriptor, metrics)
        return descriptor, metrics, False, time.perf_counter() - started


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Yerel DINOv2 artifact, descriptor ve coarse matching dogrulamasi."
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--benchmark-runs", type=int, default=0)
    parser.add_argument("--save-visualization", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_artifacts(settings: Settings) -> dict[str, object]:
    repo = settings.matching_dinov2_repo_path
    weights = settings.matching_dinov2_weights_path
    repo_exists = bool(repo and repo.is_dir())
    hubconf_exists = bool(repo_exists and repo and (repo / "hubconf.py").is_file())
    weight_exists = bool(weights and weights.is_file())
    extension_valid = bool(weights and weights.suffix.lower() in {".pt", ".pth", ".ckpt"})
    weight_metadata: dict[str, object] = {
        "path": str(weights) if weights else None,
        "exists": weight_exists,
        "extension": weights.suffix.lower() if weights else None,
        "extension_valid": extension_valid,
        "size_bytes": weights.stat().st_size if weight_exists and weights else None,
        "sha256_short": _sha256(weights)[:12] if weight_exists and weights else None,
        "trusted_artifact_required": True,
    }
    valid = repo_exists and hubconf_exists and weight_exists and extension_valid
    return {
        "valid": valid,
        "repository": {
            "path": str(repo) if repo else None,
            "exists": repo_exists,
            "hubconf_exists": hubconf_exists,
        },
        "weights": weight_metadata,
        "model_name": settings.matching_dinov2_model_name,
        "requested_device": settings.matching_dinov2_device,
        "cpu_fallback_allowed": settings.matching_dinov2_allow_cpu_fallback,
        "failure_reason": None if valid else _artifact_failure_reason(
            repo_exists, hubconf_exists, weight_exists, extension_valid
        ),
    }


def _artifact_failure_reason(
    repo_exists: bool,
    hubconf_exists: bool,
    weight_exists: bool,
    extension_valid: bool,
) -> str:
    if not repo_exists:
        return "dinov2_repository_missing"
    if not hubconf_exists:
        return "hubconf_missing"
    if not weight_exists:
        return "dinov2_weights_missing"
    if not extension_valid:
        return "weight_extension_invalid"
    return "artifact_invalid"


def _environment() -> dict[str, object]:
    try:
        import torch

        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
        }
    except ImportError:
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
        }


def load_local_image(path: Path) -> tuple[object, dict[str, object]]:
    import cv2
    import numpy as np

    resolved = path.expanduser().resolve()
    content = resolved.read_bytes()
    image_format = detect_image_format(content)
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Goruntu decode edilemedi.")
    height, width = image.shape[:2]
    channels = int(image.shape[2]) if image.ndim == 3 else 1
    return image, {
        "path": str(resolved),
        "format": image_format,
        "width": int(width),
        "height": int(height),
        "channels": channels,
        "decode": "OK",
        "sha256_short": hashlib.sha256(content).hexdigest()[:12],
        "source_hash": hashlib.sha256(content).hexdigest(),
    }


def descriptor_metadata(descriptor, metrics) -> dict[str, object]:
    import torch

    values = descriptor.descriptors.detach().float()
    finite = bool(torch.isfinite(values).all())
    norms = torch.linalg.vector_norm(values, dim=1)
    norms_finite = bool(torch.isfinite(norms).all())
    return {
        "valid": finite and norms_finite,
        "resized_width": descriptor.resized_width,
        "resized_height": descriptor.resized_height,
        "grid_width": descriptor.grid_width,
        "grid_height": descriptor.grid_height,
        "descriptor_count": descriptor.shape[0],
        "descriptor_dimension": descriptor.descriptor_dim,
        "dtype": descriptor.dtype,
        "device": descriptor.device,
        "nan_or_inf": not finite,
        "l2_norm": {
            "minimum": float(norms.min()) if len(norms) and norms_finite else None,
            "maximum": float(norms.max()) if len(norms) and norms_finite else None,
            "mean": float(norms.mean()) if len(norms) and norms_finite else None,
        },
        "preprocessing_seconds": metrics.preprocessing_seconds,
        "forward_seconds": metrics.forward_seconds,
        "descriptor_bytes": metrics.descriptor_bytes,
        "failure_reason": None if finite and norms_finite else "descriptor_non_finite",
    }


def evaluate_match(settings: Settings, reference_descriptor, frame_descriptor) -> tuple[dict[str, object], MatchArtifacts, dict[str, float]]:
    timings: dict[str, float] = {}
    artifacts = MatchArtifacts()
    coarse = CoarseMatcher(settings)
    started = time.perf_counter()
    matches = coarse.match(reference_descriptor, frame_descriptor)
    timings["coarse_matching_seconds"] = time.perf_counter() - started
    artifacts.matches = matches
    report: dict[str, object] = {
        "correspondence_count": matches.correspondence_count,
        "similarity": {
            "minimum": matches.min_similarity,
            "mean": matches.mean_similarity,
            "median": matches.median_similarity,
            "maximum": matches.max_similarity,
        },
        "spatial_coverage": matches.spatial_coverage,
        "homography_valid": False,
        "accepted": False,
        "failure_reason": matches.failure_reason,
    }
    if matches.failure_reason:
        report["thresholds"] = _threshold_observations(settings, report)
        return report, artifacts, timings

    geometry_started = time.perf_counter()
    homography = HomographyEstimator(settings).estimate(matches)
    artifacts.homography = homography
    report.update({
        "homography_valid": homography.valid,
        "inlier_count": homography.inlier_count,
        "inlier_ratio": homography.inlier_ratio,
        "rms_reprojection_error": _finite_or_none(homography.rms_reprojection_error),
        "failure_reason": homography.failure_reason,
    })
    if not homography.valid:
        timings["homography_geometry_seconds"] = time.perf_counter() - geometry_started
        report["thresholds"] = _threshold_observations(settings, report)
        return report, artifacts, timings

    polygon = ProjectedPolygonValidator(settings).project_and_validate(
        homography.matrix,
        reference_width=reference_descriptor.image_width,
        reference_height=reference_descriptor.image_height,
        frame_width=frame_descriptor.image_width,
        frame_height=frame_descriptor.image_height,
    )
    artifacts.polygon = polygon
    report.update({
        "projected_polygon": _points_list(polygon.points),
        "projected_area": polygon.raw_area,
        "visible_area": polygon.visible_area,
        "visible_ratio": polygon.visible_ratio,
        "failure_reason": polygon.failure_reason,
    })
    if not polygon.valid:
        timings["homography_geometry_seconds"] = time.perf_counter() - geometry_started
        report["thresholds"] = _threshold_observations(settings, report)
        return report, artifacts, timings

    bbox = ProjectedBoundingBoxValidator(settings).validate(
        polygon,
        frame_width=frame_descriptor.image_width,
        frame_height=frame_descriptor.image_height,
    )
    artifacts.bbox = bbox
    if bbox is None:
        report["failure_reason"] = "bbox_invalid"
        timings["homography_geometry_seconds"] = time.perf_counter() - geometry_started
        report["thresholds"] = _threshold_observations(settings, report)
        return report, artifacts, timings
    report["raw_bbox"] = list(bbox.raw_box)
    report["clipped_bbox"] = list(bbox.clipped_box)

    quality = ConfidenceScorer(settings).score(matches, homography, polygon)
    artifacts.quality = quality
    timings["homography_geometry_seconds"] = time.perf_counter() - geometry_started
    if quality is None:
        report["failure_reason"] = "confidence_below_threshold"
        report["thresholds"] = _threshold_observations(settings, report)
        return report, artifacts, timings
    report.update({
        "confidence": quality.confidence,
        "confidence_components": {
            "inlier_ratio": quality.inlier_ratio,
            "similarity_score": quality.similarity_score,
            "reprojection_score": quality.reprojection_score,
            "visibility_score": quality.visibility_score,
            "coverage_score": quality.coverage_score,
        },
        "accepted": True,
        "failure_reason": None,
    })
    report["thresholds"] = _threshold_observations(settings, report)
    return report, artifacts, timings


def _threshold_observations(settings: Settings, report: dict[str, object]) -> dict[str, object]:
    similarity = report.get("similarity") or {}
    values = {
        "minimum_similarity": (similarity.get("minimum", 0.0), settings.matching_coarse_min_similarity, ">="),
        "correspondence_count": (report.get("correspondence_count", 0), settings.matching_coarse_min_correspondences, ">="),
        "inlier_count": (report.get("inlier_count", 0), settings.matching_homography_min_inliers, ">="),
        "inlier_ratio": (report.get("inlier_ratio", 0.0), settings.matching_homography_min_inlier_ratio, ">="),
        "rms_reprojection_error": (
            report.get("rms_reprojection_error"),
            settings.matching_homography_max_rms_reprojection_error,
            "<=",
        ),
        "visible_ratio": (report.get("visible_ratio", 0.0), settings.matching_geometry_min_visible_ratio, ">="),
        "confidence": (report.get("confidence"), settings.matching_min_confidence, ">="),
    }
    observations = {}
    for name, (measured, threshold, operator) in values.items():
        passed = False
        if measured is not None and isinstance(measured, (int, float)) and math.isfinite(float(measured)):
            passed = measured >= threshold if operator == ">=" else measured <= threshold
        observations[name] = {
            "measured": measured,
            "threshold": threshold,
            "operator": operator,
            "pass": passed,
        }
    return observations


def _finite_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _points_list(points) -> list[list[float]] | None:
    if points is None:
        return None
    return [[float(x), float(y)] for x, y in points]


def _statistics(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
    }


def run_benchmark(
    settings: Settings,
    runtime,
    cache: ReferenceDescriptorCache,
    reference_image,
    reference_hash: str,
    frame_image,
    frame_hash: str,
    runs: int,
) -> dict[str, object]:
    if runs < 1:
        return {}
    try:
        import torch
    except ImportError:
        torch = None
    if runtime.device == "cuda" and torch is not None:
        torch.cuda.reset_peak_memory_stats()

    reference_descriptor, _, _, _ = cache.get(runtime, reference_image, reference_hash)
    warmup_descriptor, _ = runtime.extract(frame_image, frame_hash)
    evaluate_match(settings, reference_descriptor, warmup_descriptor)

    samples: dict[str, list[float]] = {
        "preprocessing_seconds": [],
        "descriptor_forward_seconds": [],
        "reference_cache_hit_seconds": [],
        "coarse_matching_seconds": [],
        "homography_geometry_seconds": [],
        "total_seconds": [],
    }
    for _ in range(runs):
        total_started = time.perf_counter()
        _, _, hit, cache_seconds = cache.get(runtime, reference_image, reference_hash)
        if not hit:
            raise RuntimeError("Reference cache benchmark sirasinda beklenmedik miss verdi.")
        frame_descriptor, metrics = runtime.extract(frame_image, frame_hash)
        _, _, timings = evaluate_match(settings, reference_descriptor, frame_descriptor)
        samples["preprocessing_seconds"].append(metrics.preprocessing_seconds)
        samples["descriptor_forward_seconds"].append(metrics.forward_seconds)
        samples["reference_cache_hit_seconds"].append(cache_seconds)
        samples["coarse_matching_seconds"].append(timings.get("coarse_matching_seconds", 0.0))
        samples["homography_geometry_seconds"].append(timings.get("homography_geometry_seconds", 0.0))
        samples["total_seconds"].append(time.perf_counter() - total_started)
    report: dict[str, object] = {
        "runs": runs,
        "warmup_runs": 1,
        "timings": {name: _statistics(values) for name, values in samples.items()},
    }
    if runtime.device == "cuda" and torch is not None:
        report["gpu_memory"] = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    else:
        report["gpu_memory"] = None
    return report


def save_visualization(path: Path, frame_image, artifacts: MatchArtifacts) -> None:
    import cv2
    import numpy as np

    canvas = frame_image.copy()
    if artifacts.matches is not None and artifacts.homography is not None:
        mask = artifacts.homography.inlier_mask
        if mask is not None:
            for point in np.asarray(artifacts.matches.frame_points_px)[np.asarray(mask, dtype=bool)]:
                cv2.circle(canvas, tuple(np.rint(point).astype(int)), 3, (0, 255, 255), -1)
    if artifacts.polygon is not None and artifacts.polygon.points is not None:
        polygon = np.rint(artifacts.polygon.points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [polygon], True, (255, 0, 0), 2)
    if artifacts.bbox is not None:
        x1, y1, x2, y2 = artifacts.bbox.clipped_box
        cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), canvas):
        raise OSError("Gorsellestirme kaydedilemedi.")


def run_validation(
    settings: Settings,
    options: ValidationOptions,
    *,
    runtime_factory: RuntimeFactory = Dinov2RuntimeRegistry.get,
    image_loader: ImageLoader = load_local_image,
    emit: Callable[[str], None] = print,
) -> tuple[int, dict[str, object]]:
    report: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "artifact_metadata": inspect_artifacts(settings),
        "environment": _environment(),
        "image_metadata": {},
        "descriptor_metadata": {},
        "match_metrics": None,
        "benchmark_metrics": None,
        "reference_cache": None,
        "final_result": "FAIL",
        "failure_reason": None,
        "prediction_submission": "DISABLED",
    }
    artifact = report["artifact_metadata"]
    assert isinstance(artifact, dict)
    emit("DINOv2 artifact validation (offline/local-only)")
    emit("WARNING: Yalniz guvenilir yerel weight artifact kullanin.")
    repository = artifact["repository"]
    weights = artifact["weights"]
    emit(
        f"Repository: {'OK' if repository['exists'] else 'FAIL'}; "
        f"hubconf.py={'OK' if repository['hubconf_exists'] else 'FAIL'}"
    )
    emit(
        f"Weights: {'OK' if weights['exists'] and weights['extension_valid'] else 'FAIL'}; "
        f"extension={weights['extension']}; size={weights['size_bytes']}; "
        f"sha256={weights['sha256_short']}"
    )
    emit(
        f"Model config: name={artifact['model_name']}; device={artifact['requested_device']}; "
        f"cpu_fallback={'enabled' if artifact['cpu_fallback_allowed'] else 'disabled'}"
    )
    if not artifact["valid"]:
        report["failure_reason"] = artifact["failure_reason"]
        emit(f"Artifact validation: FAIL ({report['failure_reason']})")
        _write_json_if_requested(report, options.json_output)
        return EXIT_ARTIFACT, report

    try:
        load_started = time.perf_counter()
        runtime = runtime_factory(settings)
        model_hash = runtime.model_hash
        load_seconds = time.perf_counter() - load_started
        report["model"] = {
            "load": "OK",
            "load_seconds": load_seconds,
            "device": runtime.device,
            "model_hash_short": model_hash[:12],
            "missing_keys": 0,
            "unexpected_keys": 0,
            "descriptor_dimension": None,
        }
        emit(f"Model load: OK; device={runtime.device}; hash={model_hash[:12]}")
    except Exception as exc:
        report["model"] = {"load": "FAIL", "failure_type": type(exc).__name__}
        report["failure_reason"] = "model_load_failed"
        emit(f"Model load: FAIL ({type(exc).__name__})")
        _write_json_if_requested(report, options.json_output)
        return EXIT_MODEL, report

    try:
        if options.image is not None:
            image, metadata = image_loader(options.image)
            descriptor, metrics = runtime.extract(image, metadata["source_hash"])
            descriptor_report = descriptor_metadata(descriptor, metrics)
            report["image_metadata"] = {"image": metadata}
            report["descriptor_metadata"] = {"image": descriptor_report}
            report["model"]["descriptor_dimension"] = descriptor.descriptor_dim
            if not descriptor_report["valid"]:
                raise ValueError("descriptor_non_finite")

        if options.reference is not None and options.frame is not None:
            reference_image, reference_metadata = image_loader(options.reference)
            frame_image, frame_metadata = image_loader(options.frame)
            cache = ReferenceDescriptorCache()
            reference_descriptor, reference_metrics, first_hit, _ = cache.get(
                runtime, reference_image, reference_metadata["source_hash"]
            )
            _, _, second_hit, second_seconds = cache.get(
                runtime, reference_image, reference_metadata["source_hash"]
            )
            frame_descriptor, frame_metrics = runtime.extract(
                frame_image, frame_metadata["source_hash"]
            )
            reference_report = descriptor_metadata(reference_descriptor, reference_metrics)
            frame_report = descriptor_metadata(frame_descriptor, frame_metrics)
            if not reference_report["valid"] or not frame_report["valid"]:
                raise ValueError("descriptor_non_finite")
            report["image_metadata"] = {
                "reference": reference_metadata,
                "frame": frame_metadata,
            }
            report["descriptor_metadata"] = {
                "reference": reference_report,
                "frame": frame_report,
            }
            report["model"]["descriptor_dimension"] = frame_descriptor.descriptor_dim
            report["reference_cache"] = {
                "first_request": "HIT" if first_hit else "MISS",
                "second_request": "HIT" if second_hit else "MISS",
                "second_request_seconds": second_seconds,
                "reference_forward_repeated": not second_hit,
            }
            match_report, match_artifacts, _ = evaluate_match(
                settings, reference_descriptor, frame_descriptor
            )
            report["match_metrics"] = match_report
            if options.save_visualization is not None:
                save_visualization(options.save_visualization, frame_image, match_artifacts)
                report["visualization"] = str(options.save_visualization.expanduser().resolve())
            if options.benchmark_runs:
                report["benchmark_metrics"] = run_benchmark(
                    settings,
                    runtime,
                    cache,
                    reference_image,
                    reference_metadata["source_hash"],
                    frame_image,
                    frame_metadata["source_hash"],
                    options.benchmark_runs,
                )
    except (OSError, ValueError) as exc:
        report["failure_reason"] = str(exc) or type(exc).__name__
        emit(f"Validation: FAIL ({report['failure_reason']})")
        _write_json_if_requested(report, options.json_output)
        return EXIT_IMAGE, report
    except Exception as exc:
        report["failure_reason"] = type(exc).__name__
        emit(f"Validation: FAIL ({type(exc).__name__})")
        _write_json_if_requested(report, options.json_output)
        return EXIT_MATCH, report

    accepted = not isinstance(report.get("match_metrics"), dict) or bool(report["match_metrics"].get("accepted"))
    report["final_result"] = "PASS" if accepted else "REJECTED"
    report["failure_reason"] = None if accepted else report["match_metrics"].get("failure_reason")
    _emit_measurements(report, emit)
    emit(f"Final result: {report['final_result']}")
    emit("Prediction submission: DISABLED")
    _write_json_if_requested(report, options.json_output)
    return EXIT_OK, report


def _emit_measurements(report: dict[str, object], emit: Callable[[str], None]) -> None:
    descriptors = report.get("descriptor_metadata") or {}
    for label, metadata in descriptors.items():
        emit(
            f"Descriptor[{label}]: resized={metadata['resized_width']}x{metadata['resized_height']}; "
            f"grid={metadata['grid_width']}x{metadata['grid_height']}; "
            f"count={metadata['descriptor_count']}; dim={metadata['descriptor_dimension']}; "
            f"dtype={metadata['dtype']}; nan_or_inf={metadata['nan_or_inf']}; "
            f"l2_mean={metadata['l2_norm']['mean']}"
        )
    cache = report.get("reference_cache")
    if cache:
        emit(
            f"Reference cache: first={cache['first_request']}; second={cache['second_request']}; "
            f"forward_repeated={cache['reference_forward_repeated']}"
        )
    match = report.get("match_metrics")
    if match:
        similarity = match.get("similarity", {})
        emit(
            f"Match: correspondences={match.get('correspondence_count')}; "
            f"similarity={similarity.get('minimum')}/{similarity.get('mean')}/"
            f"{similarity.get('median')}/{similarity.get('maximum')}; "
            f"coverage={match.get('spatial_coverage')}; accepted={match.get('accepted')}; "
            f"reason={match.get('failure_reason')}"
        )
        emit(
            f"Geometry: homography={match.get('homography_valid')}; "
            f"inliers={match.get('inlier_count')}; ratio={match.get('inlier_ratio')}; "
            f"rms={match.get('rms_reprojection_error')}; visible={match.get('visible_ratio')}; "
            f"bbox={match.get('clipped_bbox')}; confidence={match.get('confidence')}"
        )
        for name, observation in match.get("thresholds", {}).items():
            emit(
                f"Threshold[{name}]: measured={observation['measured']}; "
                f"required={observation['operator']}{observation['threshold']}; "
                f"pass={observation['pass']}"
            )
    benchmark = report.get("benchmark_metrics")
    if benchmark:
        emit(f"Benchmark: runs={benchmark['runs']}; warmup={benchmark['warmup_runs']}")
        for name, values in benchmark["timings"].items():
            emit(
                f"Benchmark[{name}]: min={values['minimum']:.6f}; p50={values['p50']:.6f}; "
                f"p95={values['p95']:.6f}; max={values['maximum']:.6f}; mean={values['mean']:.6f}"
            )
        if benchmark.get("gpu_memory"):
            emit(
                f"GPU memory: allocated={benchmark['gpu_memory']['peak_allocated_bytes']}; "
                f"reserved={benchmark['gpu_memory']['peak_reserved_bytes']}"
            )


def _write_json_if_requested(report: dict[str, object], path: Path | None) -> None:
    if path is None:
        return
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_options(options: ValidationOptions) -> str | None:
    if (options.reference is None) != (options.frame is None):
        return "--reference ve --frame birlikte verilmelidir."
    if options.image is not None and options.reference is not None:
        return "--image ile --reference/--frame ayni anda kullanilamaz."
    if options.benchmark_runs < 0:
        return "--benchmark-runs negatif olamaz."
    if options.benchmark_runs and options.reference is None:
        return "Benchmark icin --reference ve --frame gereklidir."
    if options.save_visualization is not None and options.reference is None:
        return "Gorsellestirme icin --reference ve --frame gereklidir."
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = ValidationOptions(
        image=args.image,
        reference=args.reference,
        frame=args.frame,
        benchmark_runs=args.benchmark_runs,
        save_visualization=args.save_visualization,
        json_output=args.json_output,
    )
    error = _validate_options(options)
    if error:
        print(f"Configuration: FAIL ({error})")
        return EXIT_ARTIFACT
    code, _ = run_validation(get_settings(), options)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
