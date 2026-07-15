from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from app.core.config import Settings, get_settings
from app.schemas import DetectedObject, ImageModality, MotionStatus, ObjectClass
from app.services.common import FrameContext
from app.services.detection.service import YoloDetectionService
from app.services.detection.homography_adaptive_motion import HomographyAdaptiveMotionAnalyzer
from app.services.detection.homography_quality import quality_gate_from_settings
from app.services.detection.yolo_runtime import YoloRuntime
from scripts.compare_task1_motion import (
    _bbox,
    _bbox_analyzer,
    _global_analyzer,
    _homography_analyzer,
    _hybrid_analyzer,
    _local_analyzer,
    _save_local_comparison,
    _save_comparison,
)
from scripts.validate_task1_detection import (
    LoadedImage,
    _configure_local_runtime_state,
    load_local_image,
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
METHODS = (
    "global_median",
    "homography",
    "homography_bbox",
    "homography_hybrid",
    "homography_local",
    "homography_adaptive",
)
CSV_COLUMNS = (
    "frame_previous",
    "frame_current",
    "vehicle_index",
    "bbox",
    "global_median_result",
    "homography_result",
    "homography_bbox_result",
    "homography_hybrid_result",
    "homography_local_result",
    "homography_adaptive_result",
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
    "homography_reason",
    "homography_matches",
    "homography_inliers",
    "homography_inlier_ratio",
    "homography_residual_px",
    "bbox_previous_index",
    "bbox_projected_bbox",
    "bbox_iou",
    "bbox_center_residual_px",
    "bbox_size_ratio",
    "bbox_association_score",
    "hybrid_bbox_result",
    "hybrid_flow_result",
    "hybrid_flow_residual_px",
    "hybrid_homography_quality_level",
    "hybrid_decision_reason",
    "local_homography_quality_level",
    "local_vehicle_residual_x",
    "local_vehicle_residual_y",
    "local_vehicle_residual_magnitude",
    "local_vehicle_magnitude_p50",
    "local_vehicle_magnitude_p75",
    "local_vehicle_magnitude_p90",
    "local_vehicle_valid_pixels",
    "local_background_residual_x",
    "local_background_residual_y",
    "local_background_residual_magnitude",
    "local_background_magnitude_p50",
    "local_background_magnitude_p75",
    "local_background_magnitude_p90",
    "local_background_valid_pixels",
    "local_background_valid_ratio",
    "local_corrected_residual_x",
    "local_corrected_residual_y",
    "local_corrected_residual_magnitude",
    "local_stationary_threshold",
    "local_moving_threshold",
    "local_decision_reason",
)
SUMMARY_COLUMNS = (
    "method",
    "total_detections",
    "moving_count",
    "stationary_count",
    "unknown_count",
    "moving_percentage",
    "stationary_percentage",
    "unknown_percentage",
)
QUALITY_CSV_COLUMNS = (
    "previous_frame",
    "current_frame",
    "matches",
    "inliers",
    "inlier_ratio",
    "fixed_050_accepted",
    "fixed_045_accepted",
    "fixed_040_accepted",
    "adaptive_accepted",
    "adaptive_quality_level",
    "adaptive_reason",
    "condition_number",
    "reprojection_error",
    "spatial_coverage",
    "projected_overlap",
)


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    images_dir: Path
    output_dir: Path
    save_visualizations: bool = False
    max_pairs: int | None = None


@dataclass(frozen=True, slots=True)
class FrameFile:
    path: Path
    sequence: str
    frame_number: int


@dataclass(frozen=True, slots=True)
class FramePair:
    previous: FrameFile
    current: FrameFile


@dataclass(frozen=True, slots=True)
class PairAnalysis:
    rows: tuple[dict[str, object], ...]
    homography_failed: bool
    current_image: object | None = None
    vehicles: tuple[DetectedObject, ...] = ()
    global_statuses: tuple[MotionStatus, ...] = ()
    homography_statuses: tuple[MotionStatus, ...] = ()
    bbox_statuses: tuple[MotionStatus, ...] = ()
    hybrid_statuses: tuple[MotionStatus, ...] = ()
    local_statuses: tuple[MotionStatus, ...] = ()
    adaptive_statuses: tuple[MotionStatus, ...] = ()
    adaptive_scene: object | None = None
    local_measurements: tuple[object, ...] = ()
    projected_boxes: tuple[tuple[float, float, float, float] | None, ...] = ()
    debug: PairDebug | None = None
    quality_row: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class PairDebug:
    previous_path: Path
    current_path: Path
    previous_sha256: str
    current_sha256: str
    images_equal: bool
    previous_shape: tuple[int, ...]
    current_shape: tuple[int, ...]
    homography_valid: bool
    homography_matches: int
    homography_inliers: int
    homography_inlier_ratio: float
    failure_reason: str


PairProcessor = Callable[[FramePair], Awaitable[PairAnalysis]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Üç production Task 1 motion yöntemini tamamen offline benchmark et."
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--max-pairs", type=int)
    return parser


def natural_sort_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )


def discover_frames(images_dir: Path) -> list[FrameFile]:
    root = images_dir.expanduser().resolve()
    frames: list[FrameFile] = []
    for path in sorted(
        (item for item in root.iterdir() if item.is_file()), key=natural_sort_key
    ):
        if path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
            continue
        match = re.fullmatch(r"(.*?)(\d+)", path.stem)
        if match is None:
            continue
        frames.append(FrameFile(path.resolve(), match.group(1).casefold(), int(match.group(2))))
    return frames


def build_frame_pairs(
    frames: Sequence[FrameFile], max_pairs: int | None = None
) -> list[FramePair]:
    grouped: dict[str, list[FrameFile]] = {}
    for frame in frames:
        grouped.setdefault(frame.sequence, []).append(frame)
    pairs: list[FramePair] = []
    for sequence_frames in grouped.values():
        ordered = sorted(sequence_frames, key=lambda item: item.frame_number)
        pairs.extend(
            FramePair(previous, current)
            for previous, current in zip(ordered, ordered[1:])
            if current.frame_number == previous.frame_number + 1
        )
    return pairs if max_pairs is None else pairs[:max_pairs]


def pairing_diagnostics(frames: Sequence[FrameFile]) -> list[str]:
    grouped: dict[str, list[FrameFile]] = {}
    for frame in frames:
        grouped.setdefault(frame.sequence, []).append(frame)
    messages: list[str] = []
    for sequence, sequence_frames in grouped.items():
        ordered = sorted(sequence_frames, key=lambda item: item.frame_number)
        valid_pairs = sum(
            current.frame_number == previous.frame_number + 1
            for previous, current in zip(ordered, ordered[1:])
        )
        first = ordered[0].frame_number
        last = ordered[-1].frame_number
        messages.append(
            f"SEQUENCE {sequence or '<empty>'}: frames={len(ordered)} "
            f"valid_pairs={valid_pairs} range={first}->{last}"
        )
        for previous, current in zip(ordered, ordered[1:]):
            if current.frame_number != previous.frame_number + 1:
                messages.append(
                    f"PAIR SKIPPED {previous.path.name} -> {current.path.name}: "
                    f"non-consecutive frame gap={current.frame_number - previous.frame_number}"
                )
    return messages


class OfflineMotionPairProcessor:
    """Runs local files through one production detector and the four analyzers."""

    def __init__(
        self,
        settings: Settings,
        images_dir: Path,
        *,
        detector: object | None = None,
    ) -> None:
        self.settings = settings
        self.root = images_dir.expanduser().resolve()
        self.loaded: dict[Path, LoadedImage] = {}
        self.detections: dict[Path, list[DetectedObject]] = {}
        detector_settings = replace(settings, detection_motion_enabled=False)
        runtime = YoloRuntime(
            settings.detection_model_path,
            settings.detection_confidence,
            settings.detection_iou,
        )

        async def local_reader(source: str, _timeout: float) -> bytes:
            path = Path(source).expanduser().resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Benchmark yalnız images-dir altındaki dosyaları okuyabilir.")
            loaded = self.loaded.get(path)
            return loaded.content if loaded is not None else path.read_bytes()

        self.detector = detector or YoloDetectionService(
            detector_settings, runtime=runtime, image_reader=local_reader
        )
        self.global_analyzer = _global_analyzer(settings)
        self.homography_analyzer = _homography_analyzer(settings)
        self.bbox_analyzer = _bbox_analyzer(settings, self.homography_analyzer)
        self.hybrid_analyzer = _hybrid_analyzer(settings, self.bbox_analyzer)
        self.local_analyzer = _local_analyzer(settings, self.homography_analyzer)
        self.adaptive_analyzer = HomographyAdaptiveMotionAnalyzer(
            self.homography_analyzer,
            self.bbox_analyzer,
            background_median_max=settings.detection_motion_adaptive_background_median_max,
            background_p90_max=settings.detection_motion_adaptive_background_p90_max,
            grid_spread_max=settings.detection_motion_adaptive_grid_spread_max,
            min_valid_background_ratio=settings.detection_motion_adaptive_min_valid_background_ratio,
        )
        self.quality_gates = {
            "fixed_050": quality_gate_from_settings(
                settings, mode="fixed", fixed_min_inlier_ratio=0.50
            ),
            "fixed_045": quality_gate_from_settings(
                settings, mode="fixed", fixed_min_inlier_ratio=0.45
            ),
            "fixed_040": quality_gate_from_settings(
                settings, mode="fixed", fixed_min_inlier_ratio=0.40
            ),
            "adaptive": quality_gate_from_settings(settings, mode="adaptive"),
        }

    async def __call__(self, pair: FramePair) -> PairAnalysis:
        previous_image, previous_detections = await self._load_and_detect(pair.previous)
        current_image, current_detections = await self._load_and_detect(pair.current)
        if previous_image.image.shape[:2] != current_image.image.shape[:2]:
            raise ValueError("frame resolution changed")
        previous_gray = self.global_analyzer.to_grayscale(previous_image.image)
        current_gray = self.global_analyzer.to_grayscale(current_image.image)
        exclusions = [
            _bbox(item)
            for item in current_detections
            if item.cls in {ObjectClass.TASIT, ObjectClass.INSAN}
        ]
        quality_decisions = self.homography_analyzer.evaluate_quality_gates(
            previous_gray, current_gray, exclusions, self.quality_gates
        )
        global_field = self.global_analyzer.compute_flow(
            previous_gray, current_gray, exclusions
        )
        computation = self.homography_analyzer.analyze_pair(
            previous_gray, current_gray, exclusions
        )
        bbox_analysis = self.bbox_analyzer.analyze(
            previous_gray,
            current_gray,
            previous_detections,
            current_detections,
            exclusions,
            homography_computation=computation,
        )
        hybrid_analysis = self.hybrid_analyzer.analyze(
            previous_gray,
            current_gray,
            previous_detections,
            current_detections,
            exclusions,
            homography_computation=computation,
        )
        local_analysis = self.local_analyzer.analyze(
            previous_gray,
            current_gray,
            current_detections,
            exclusions,
            homography_computation=computation,
        )
        adaptive_analysis = self.adaptive_analyzer.analyze(
            previous_gray,
            current_gray,
            previous_detections,
            current_detections,
            exclusions,
            homography_computation=computation,
        )
        vehicles = [item for item in current_detections if item.cls is ObjectClass.TASIT]
        current_vehicle_indices = [
            index for index, item in enumerate(current_detections)
            if item.cls is ObjectClass.TASIT
        ]
        global_statuses = [
            self.global_analyzer.classify_vehicle(global_field, _bbox(item))
            for item in vehicles
        ]
        homography_measurements = [
            self.homography_analyzer.measure_vehicle(computation.field, _bbox(item))
            for item in vehicles
        ]
        bbox_measurements = list(bbox_analysis.measurements)
        hybrid_measurements = list(hybrid_analysis.measurements)
        local_measurements = list(local_analysis.measurements)
        diagnostics = computation.diagnostics
        rows: list[dict[str, object]] = []
        adaptive_by_index = {item.current_index: item.status for item in adaptive_analysis.measurements}
        for index, (vehicle, global_status, homography, bbox, hybrid, local) in enumerate(
            zip(
                vehicles,
                global_statuses,
                homography_measurements,
                bbox_measurements,
                hybrid_measurements,
                local_measurements,
                strict=True,
            )
        ):
            rows.append(
                {
                    "frame_previous": pair.previous.path.name,
                    "frame_current": pair.current.path.name,
                    "vehicle_index": index,
                    "bbox": _json_bbox(_bbox(vehicle)),
                    "global_median_result": global_status.value,
                    "homography_result": homography.status.value,
                    "homography_bbox_result": bbox.status.value,
                    "homography_hybrid_result": hybrid.final_result.value,
                    "homography_local_result": local.final_result.value,
                    "homography_adaptive_result": adaptive_by_index.get(
                        current_vehicle_indices[index], MotionStatus.UNKNOWN
                    ).value,
                    "adaptive_selected_method": adaptive_analysis.scene.selected_method,
                    "adaptive_scene_quality": adaptive_analysis.scene.scene_quality.value,
                    "adaptive_selection_reason": adaptive_analysis.scene.selection_reason,
                    "adaptive_homography_quality": adaptive_analysis.scene.homography_quality,
                    "adaptive_background_residual_median": _optional(adaptive_analysis.scene.background_residual_median),
                    "adaptive_background_residual_p90": _optional(adaptive_analysis.scene.background_residual_p90),
                    "adaptive_background_residual_p95": _optional(adaptive_analysis.scene.background_residual_p95),
                    "adaptive_background_grid_spread": _optional(adaptive_analysis.scene.background_grid_spread),
                    "adaptive_background_spatial_variance": _optional(adaptive_analysis.scene.background_spatial_variance),
                    "adaptive_valid_background_ratio": adaptive_analysis.scene.valid_background_ratio,
                    "homography_valid": diagnostics.valid,
                    "homography_reason": diagnostics.reason,
                    "homography_matches": diagnostics.match_count,
                    "homography_inliers": diagnostics.inlier_count,
                    "homography_inlier_ratio": diagnostics.inlier_ratio,
                    "homography_residual_px": _optional(homography.residual_motion_magnitude),
                    "bbox_previous_index": _optional(bbox.previous_index),
                    "bbox_projected_bbox": (
                        _json_bbox(bbox.projected_bbox) if bbox.projected_bbox else ""
                    ),
                    "bbox_iou": _optional(bbox.iou),
                    "bbox_center_residual_px": _optional(bbox.center_residual_px),
                    "bbox_size_ratio": _optional(bbox.size_ratio),
                    "bbox_association_score": _optional(bbox.association_score),
                    "hybrid_bbox_result": hybrid.bbox_result.value,
                    "hybrid_flow_result": hybrid.flow_result.value,
                    "hybrid_flow_residual_px": _optional(hybrid.flow_residual_px),
                    "hybrid_homography_quality_level": _optional(
                        hybrid.homography_quality_level
                    ),
                    "hybrid_decision_reason": hybrid.decision_reason,
                    "local_homography_quality_level": _optional(
                        local.homography_quality_level
                    ),
                    **_local_statistics_columns("vehicle", local.vehicle_statistics),
                    **_local_statistics_columns(
                        "background", local.background_statistics
                    ),
                    "local_background_valid_ratio": local.background_valid_ratio,
                    "local_corrected_residual_x": _optional(
                        local.corrected_residual_x
                    ),
                    "local_corrected_residual_y": _optional(
                        local.corrected_residual_y
                    ),
                    "local_corrected_residual_magnitude": _optional(
                        local.corrected_residual_magnitude
                    ),
                    "local_stationary_threshold": local.stationary_threshold,
                    "local_moving_threshold": local.moving_threshold,
                    "local_decision_reason": local.decision_reason,
                }
            )
        return PairAnalysis(
            rows=tuple(rows),
            homography_failed=not diagnostics.valid,
            current_image=current_image.image,
            vehicles=tuple(vehicles),
            global_statuses=tuple(global_statuses),
            homography_statuses=tuple(item.status for item in homography_measurements),
            bbox_statuses=tuple(item.status for item in bbox_measurements),
            hybrid_statuses=tuple(item.final_result for item in hybrid_measurements),
            local_statuses=tuple(item.final_result for item in local_measurements),
            adaptive_statuses=tuple(
                adaptive_by_index.get(current_index, MotionStatus.UNKNOWN)
                for current_index in current_vehicle_indices
            ),
            adaptive_scene=adaptive_analysis.scene,
            local_measurements=tuple(local_measurements),
            projected_boxes=tuple(item.projected_bbox for item in bbox_measurements),
            debug=PairDebug(
                previous_path=pair.previous.path,
                current_path=pair.current.path,
                previous_sha256=hashlib.sha256(previous_image.content).hexdigest(),
                current_sha256=hashlib.sha256(current_image.content).hexdigest(),
                images_equal=(previous_image.content == current_image.content),
                previous_shape=tuple(int(value) for value in previous_image.image.shape),
                current_shape=tuple(int(value) for value in current_image.image.shape),
                homography_valid=diagnostics.valid,
                homography_matches=diagnostics.match_count,
                homography_inliers=diagnostics.inlier_count,
                homography_inlier_ratio=diagnostics.inlier_ratio,
                failure_reason=diagnostics.reason,
            ),
            quality_row=self._quality_row(pair, quality_decisions),
        )

    @staticmethod
    def _quality_row(pair: FramePair, decisions: dict[str, object]) -> dict[str, object]:
        fixed_050 = decisions["fixed_050"]
        fixed_045 = decisions["fixed_045"]
        fixed_040 = decisions["fixed_040"]
        adaptive = decisions["adaptive"]
        return {
            "previous_frame": pair.previous.path.name,
            "current_frame": pair.current.path.name,
            "matches": adaptive.matches,
            "inliers": adaptive.inliers,
            "inlier_ratio": adaptive.inlier_ratio,
            "fixed_050_accepted": fixed_050.accepted,
            "fixed_045_accepted": fixed_045.accepted,
            "fixed_040_accepted": fixed_040.accepted,
            "adaptive_accepted": adaptive.accepted,
            "adaptive_quality_level": adaptive.quality_level,
            "adaptive_reason": adaptive.reason,
            "condition_number": _optional(adaptive.condition_number),
            "reprojection_error": _optional(adaptive.reprojection_error),
            "spatial_coverage": _optional(adaptive.spatial_coverage),
            "projected_overlap": _optional(adaptive.projected_overlap),
        }

    async def _load_and_detect(
        self, frame: FrameFile
    ) -> tuple[LoadedImage, list[DetectedObject]]:
        path = frame.path.resolve()
        loaded = self.loaded.get(path)
        if loaded is None:
            loaded = load_local_image(path)
            self.loaded[path] = loaded
        detections = self.detections.get(path)
        if detections is None:
            context = FrameContext(
                frame_id=f"offline:{frame.sequence}:{frame.frame_number}",
                image_url=str(path),
                video_name=frame.sequence,
                session_id=f"benchmark:{frame.sequence}",
                gps_health_status=None,
                gps_x=None,
                gps_y=None,
                gps_z=None,
                frame_index=frame.frame_number,
                image_modality=ImageModality.RGB,
            )
            detections = await self.detector.process_frame(context)
            self.detections[path] = detections
        return loaded, detections


def calculate_summary(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for method in METHODS:
        key = f"{method}_result"
        counts = {
            status.value: sum(row.get(key, MotionStatus.UNKNOWN.value) == status.value for row in rows)
            for status in (MotionStatus.MOVING, MotionStatus.STATIONARY, MotionStatus.UNKNOWN)
        }
        total = sum(counts.values())
        summary.append(
            {
                "method": method,
                "total_detections": total,
                "moving_count": counts[MotionStatus.MOVING.value],
                "stationary_count": counts[MotionStatus.STATIONARY.value],
                "unknown_count": counts[MotionStatus.UNKNOWN.value],
                "moving_percentage": _percentage(counts[MotionStatus.MOVING.value], total),
                "stationary_percentage": _percentage(counts[MotionStatus.STATIONARY.value], total),
                "unknown_percentage": _percentage(counts[MotionStatus.UNKNOWN.value], total),
            }
        )
    return summary


def calculate_disagreements(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    comparisons = {
        "global_median_vs_homography": ("global_median_result", "homography_result"),
        "global_median_vs_homography_bbox": (
            "global_median_result",
            "homography_bbox_result",
        ),
        "homography_vs_homography_bbox": ("homography_result", "homography_bbox_result"),
        "global_median_vs_homography_hybrid": (
            "global_median_result",
            "homography_hybrid_result",
        ),
        "homography_vs_homography_hybrid": (
            "homography_result",
            "homography_hybrid_result",
        ),
        "homography_bbox_vs_homography_hybrid": (
            "homography_bbox_result",
            "homography_hybrid_result",
        ),
        "global_median_vs_homography_local": (
            "global_median_result",
            "homography_local_result",
        ),
        "homography_vs_homography_local": (
            "homography_result",
            "homography_local_result",
        ),
        "homography_bbox_vs_homography_local": (
            "homography_bbox_result",
            "homography_local_result",
        ),
        "homography_hybrid_vs_homography_local": (
            "homography_hybrid_result",
            "homography_local_result",
        ),
    }
    return {
        name: sum(row.get(first) != row.get(second) for row in rows)
        for name, (first, second) in comparisons.items()
    }


def calculate_transitions(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    transitions = {
        "global_median_moving_to_homography_stationary": (
            "global_median_result", MotionStatus.MOVING.value,
            "homography_result", MotionStatus.STATIONARY.value,
        ),
        "global_median_moving_to_homography_unknown": (
            "global_median_result", MotionStatus.MOVING.value,
            "homography_result", MotionStatus.UNKNOWN.value,
        ),
        "homography_stationary_to_bbox_unknown": (
            "homography_result", MotionStatus.STATIONARY.value,
            "homography_bbox_result", MotionStatus.UNKNOWN.value,
        ),
        "homography_moving_to_bbox_unknown": (
            "homography_result", MotionStatus.MOVING.value,
            "homography_bbox_result", MotionStatus.UNKNOWN.value,
        ),
    }
    return {
        name: sum(
            row.get(first_key) == first_value and row.get(second_key) == second_value
            for row in rows
        )
        for name, (first_key, first_value, second_key, second_value) in transitions.items()
    }


async def run_benchmark(
    settings: Settings,
    options: BenchmarkOptions,
    *,
    processor: PairProcessor | None = None,
    emit: Callable[[str], None] = print,
) -> dict[str, object]:
    if options.max_pairs is not None and options.max_pairs < 1:
        raise ValueError("--max-pairs en az 1 olmalıdır")
    root = options.images_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("--images-dir mevcut bir klasör olmalıdır")
    output = options.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = discover_frames(root)
    pairs = build_frame_pairs(frames, options.max_pairs)
    emit(f"Frames discovered: {len(frames)}")
    for message in pairing_diagnostics(frames):
        emit(message)
    emit(f"Frame pairs selected: {len(pairs)}")
    if processor is None:
        _configure_local_runtime_state()
        processor = OfflineMotionPairProcessor(settings, root)
    rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    errors: list[str] = []
    tested = 0
    homography_failures = 0
    visualizations = output / "visualizations"
    first_pair_debug_emitted = False
    for pair in pairs:
        try:
            analysis = await processor(pair)
            tested += 1
            homography_failures += int(analysis.homography_failed)
            rows.extend(analysis.rows)
            if analysis.quality_row is not None:
                quality_rows.append(analysis.quality_row)
                _emit_quality_pair(emit, analysis.quality_row)
            if not first_pair_debug_emitted and analysis.debug is not None:
                _emit_pair_debug(emit, analysis.debug)
                first_pair_debug_emitted = True
            if options.save_visualizations and analysis.current_image is not None:
                visualizations.mkdir(parents=True, exist_ok=True)
                stem = f"{pair.previous.path.stem}_to_{pair.current.path.stem}"
                _save_comparison(
                    visualizations / f"{stem}_global_median.jpg",
                    analysis.current_image,
                    list(analysis.vehicles),
                    list(analysis.global_statuses),
                    "global_median",
                )
                _save_comparison(
                    visualizations / f"{stem}_homography.jpg",
                    analysis.current_image,
                    list(analysis.vehicles),
                    list(analysis.homography_statuses),
                    "homography",
                )
                _save_comparison(
                    visualizations / f"{stem}_homography_bbox.jpg",
                    analysis.current_image,
                    list(analysis.vehicles),
                    list(analysis.bbox_statuses),
                    "homography_bbox",
                    list(analysis.projected_boxes),
                )
                _save_comparison(
                    visualizations / f"{stem}_homography_adaptive.jpg",
                    analysis.current_image,
                    list(analysis.vehicles),
                    list(analysis.adaptive_statuses),
                    "homography_adaptive",
                )
                _save_local_comparison(
                    visualizations / f"{stem}_homography_local.jpg",
                    analysis.current_image,
                    list(analysis.vehicles),
                    list(analysis.local_measurements),
                )
                _save_comparison(
                    visualizations / f"{stem}_homography_hybrid.jpg",
                    analysis.current_image,
                    list(analysis.vehicles),
                    list(analysis.hybrid_statuses),
                    "homography_hybrid",
                    list(analysis.projected_boxes),
                )
        except Exception as exc:
            message = (
                f"PAIR ERROR {pair.previous.path.name} -> {pair.current.path.name}: "
                f"{type(exc).__name__}: {exc}"
            )
            errors.append(message)
            emit(message)
    summary = calculate_summary(rows)
    disagreements = calculate_disagreements(rows)
    transitions = calculate_transitions(rows)
    detail_path = output / "motion_benchmark.csv"
    summary_path = output / "motion_benchmark_summary.csv"
    quality_path = output / "homography_quality_benchmark.csv"
    _write_csv(detail_path, CSV_COLUMNS, rows)
    _write_csv(summary_path, SUMMARY_COLUMNS, summary)
    _write_csv(quality_path, QUALITY_CSV_COLUMNS, quality_rows)
    _emit_summary(
        emit,
        tested,
        len(rows),
        summary,
        disagreements,
        transitions,
        homography_failures,
        detail_path,
        summary_path,
    )
    quality_summary = calculate_quality_summary(quality_rows)
    _emit_quality_summary(emit, quality_summary, quality_rows, quality_path)
    emit("Prediction submission: DISABLED")
    selector_pairs = {
        method: sorted({
            f"{row['frame_previous']}->{row['frame_current']}"
            for row in rows if row.get("adaptive_selected_method") == method
        })
        for method in ("homography", "homography_bbox", "unknown")
    }
    for method, selected_pairs in selector_pairs.items():
        emit(f"Adaptive selected {method}: {len(selected_pairs)} frame pair(s)")
        for selected_pair in selected_pairs:
            emit(f"  {selected_pair}")
    return {
        "frame_pairs_discovered": len(pairs),
        "frame_pairs_tested": tested,
        "total_vehicle_detections": len(rows),
        "errors": errors,
        "summary": summary,
        "disagreements": disagreements,
        "transitions": transitions,
        "homography_failures": homography_failures,
        "detail_csv": detail_path,
        "summary_csv": summary_path,
        "quality_csv": quality_path,
        "quality_summary": quality_summary,
        "adaptive_selector_pairs": selector_pairs,
    }


def _emit_summary(
    emit: Callable[[str], None],
    tested: int,
    total: int,
    summary: Sequence[dict[str, object]],
    disagreements: dict[str, int],
    transitions: dict[str, int],
    failures: int,
    detail_path: Path,
    summary_path: Path,
) -> None:
    emit("===== TASK 1 MOTION BENCHMARK =====")
    emit(f"Frame pairs tested: {tested}")
    emit(f"Total vehicle detections: {total}")
    for item in summary:
        emit(f"{item['method']}:")
        emit(
            f"  MOVING: {item['moving_count']}  STATIONARY: {item['stationary_count']} "
            f" UNKNOWN: {item['unknown_count']}"
        )
    emit("Method disagreements:")
    emit(f"  global_median vs homography: {disagreements['global_median_vs_homography']}")
    emit(f"  global_median vs homography_bbox: {disagreements['global_median_vs_homography_bbox']}")
    emit(f"  homography vs homography_bbox: {disagreements['homography_vs_homography_bbox']}")
    emit(
        "  global_median vs homography_hybrid: "
        f"{disagreements['global_median_vs_homography_hybrid']}"
    )
    emit(
        "  homography vs homography_hybrid: "
        f"{disagreements['homography_vs_homography_hybrid']}"
    )
    emit(
        "  homography_bbox vs homography_hybrid: "
        f"{disagreements['homography_bbox_vs_homography_hybrid']}"
    )
    emit(
        "  global_median vs homography_local: "
        f"{disagreements['global_median_vs_homography_local']}"
    )
    emit(
        "  homography vs homography_local: "
        f"{disagreements['homography_vs_homography_local']}"
    )
    emit(
        "  homography_bbox vs homography_local: "
        f"{disagreements['homography_bbox_vs_homography_local']}"
    )
    emit(
        "  homography_hybrid vs homography_local: "
        f"{disagreements['homography_hybrid_vs_homography_local']}"
    )
    emit("Important transitions:")
    emit(
        "  global_median MOVING -> homography STATIONARY: "
        f"{transitions['global_median_moving_to_homography_stationary']}"
    )
    emit(
        "  global_median MOVING -> homography UNKNOWN: "
        f"{transitions['global_median_moving_to_homography_unknown']}"
    )
    emit(
        "  homography STATIONARY -> homography_bbox UNKNOWN: "
        f"{transitions['homography_stationary_to_bbox_unknown']}"
    )
    emit(
        "  homography MOVING -> homography_bbox UNKNOWN: "
        f"{transitions['homography_moving_to_bbox_unknown']}"
    )
    emit(f"Homography failures: {failures}")
    emit(f"Results: {detail_path}")
    emit(f"         {summary_path}")


def _emit_pair_debug(emit: Callable[[str], None], debug: PairDebug) -> None:
    emit("===== FIRST VALID PAIR DEBUG =====")
    emit(f"previous image path: {debug.previous_path}")
    emit(f"current image path: {debug.current_path}")
    emit(f"previous image SHA-256: {debug.previous_sha256}")
    emit(f"current image SHA-256: {debug.current_sha256}")
    emit(f"images_equal: {debug.images_equal}")
    emit(f"previous shape: {debug.previous_shape}")
    emit(f"current shape: {debug.current_shape}")
    emit(f"homography valid: {debug.homography_valid}")
    emit(f"homography matches: {debug.homography_matches}")
    emit(f"homography inliers: {debug.homography_inliers}")
    emit(f"homography inlier_ratio: {debug.homography_inlier_ratio:.6f}")
    emit(f"failure_reason: {debug.failure_reason}")


def calculate_quality_summary(
    rows: Sequence[dict[str, object]],
) -> dict[str, int]:
    total = len(rows)
    result = {"total_frame_pairs": total}
    for name in ("fixed_050", "fixed_045", "fixed_040", "adaptive"):
        accepted = sum(row[f"{name}_accepted"] is True for row in rows)
        result[f"{name}_accepted"] = accepted
        result[f"{name}_rejected"] = total - accepted
    result["adaptive_high_accepted"] = sum(
        row["adaptive_accepted"] is True
        and row["adaptive_quality_level"] == "high"
        for row in rows
    )
    result["adaptive_intermediate_accepted"] = sum(
        row["adaptive_accepted"] is True
        and row["adaptive_quality_level"] == "intermediate"
        for row in rows
    )
    result["adaptive_intermediate_rejected"] = sum(
        row["adaptive_accepted"] is False
        and row["adaptive_quality_level"] == "intermediate"
        for row in rows
    )
    result["adaptive_low_rejected"] = sum(
        row["adaptive_accepted"] is False
        and row["adaptive_quality_level"] == "low"
        for row in rows
    )
    return result


def _emit_quality_pair(emit: Callable[[str], None], row: dict[str, object]) -> None:
    emit(
        f"QUALITY {row['previous_frame']} -> {row['current_frame']}: "
        f"matches={row['matches']} inliers={row['inliers']} "
        f"ratio={float(row['inlier_ratio']):.6f} "
        f"fixed_050={'accepted' if row['fixed_050_accepted'] else 'rejected'} "
        f"fixed_045={'accepted' if row['fixed_045_accepted'] else 'rejected'} "
        f"fixed_040={'accepted' if row['fixed_040_accepted'] else 'rejected'} "
        f"adaptive={'accepted' if row['adaptive_accepted'] else 'rejected'} "
        f"level={row['adaptive_quality_level']} reason={row['adaptive_reason']}"
    )


def _emit_quality_summary(
    emit: Callable[[str], None],
    summary: dict[str, int],
    rows: Sequence[dict[str, object]],
    quality_path: Path,
) -> None:
    emit("===== HOMOGRAPHY QUALITY BENCHMARK =====")
    emit(f"Total frame pairs: {summary['total_frame_pairs']}")
    for name, label in (
        ("fixed_050", "Fixed 0.50"),
        ("fixed_045", "Fixed 0.45"),
        ("fixed_040", "Fixed 0.40"),
        ("adaptive", "Adaptive"),
    ):
        emit(
            f"{label}: accepted={summary[f'{name}_accepted']} "
            f"rejected={summary[f'{name}_rejected']}"
        )
    emit(f"Adaptive HIGH accepted: {summary['adaptive_high_accepted']}")
    emit(
        "Adaptive INTERMEDIATE accepted: "
        f"{summary['adaptive_intermediate_accepted']}"
    )
    emit(
        "Adaptive INTERMEDIATE rejected: "
        f"{summary['adaptive_intermediate_rejected']}"
    )
    emit(f"Adaptive LOW rejected: {summary['adaptive_low_rejected']}")
    adaptive_extra = [
        f"{row['previous_frame']}->{row['current_frame']}"
        for row in rows
        if row["adaptive_accepted"] is True and row["fixed_050_accepted"] is False
    ]
    fixed_040_extra = [
        f"{row['previous_frame']}->{row['current_frame']}"
        for row in rows
        if row["adaptive_accepted"] is False and row["fixed_040_accepted"] is True
    ]
    emit(
        "Adaptive accepted / Fixed 0.50 rejected: "
        + (", ".join(adaptive_extra) if adaptive_extra else "none")
    )
    emit(
        "Adaptive rejected / Fixed 0.40 accepted: "
        + (", ".join(fixed_040_extra) if fixed_040_extra else "none")
    )
    emit(f"Quality results: {quality_path}")


def _write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_bbox(bbox: tuple[float, float, float, float]) -> str:
    return json.dumps([float(value) for value in bbox], separators=(",", ":"))


def _optional(value: object | None) -> object:
    return "" if value is None else value


def _local_statistics_columns(prefix: str, statistics: object | None) -> dict[str, object]:
    def value(name: str) -> object:
        return _optional(getattr(statistics, name, None))

    return {
        f"local_{prefix}_residual_x": value("median_x"),
        f"local_{prefix}_residual_y": value("median_y"),
        f"local_{prefix}_residual_magnitude": value("vector_magnitude"),
        f"local_{prefix}_magnitude_p50": value("magnitude_p50"),
        f"local_{prefix}_magnitude_p75": value("magnitude_p75"),
        f"local_{prefix}_magnitude_p90": value("magnitude_p90"),
        f"local_{prefix}_valid_pixels": value("valid_pixel_count"),
    }


def _percentage(count: int, total: int) -> float:
    return round((100.0 * count / total), 3) if total else 0.0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = BenchmarkOptions(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        save_visualizations=args.save_visualizations,
        max_pairs=args.max_pairs,
    )
    try:
        asyncio.run(run_benchmark(get_settings(), options))
    except Exception as exc:
        print(f"Benchmark: FAIL ({type(exc).__name__}: {exc})")
        print("Prediction submission: DISABLED")
        return 10
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
