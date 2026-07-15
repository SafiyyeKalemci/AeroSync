from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings, get_settings
from app.schemas import CompetitionResponse, DetectedTranslation, GPSHealthStatus, ImageModality
from app.services.common import FrameContext
from app.services.localization.camera_model import CameraModel
from app.services.localization.interface import LocalizationSessionState
from app.services.localization.service import AffineLocalizationService
from app.services.localization.state import AffineMotionResult
from scripts.benchmark_task2_vo_quality import discover_and_validate_frames
from scripts.validate_task1_detection import LoadedImage
from scripts.validate_task2_localization import process_loaded_sequence, task2_config_report

EXIT_OK = 0
EXIT_INPUT = 20

CSV_COLUMNS = (
    "frame_index",
    "frame_name",
    "gps_healthy",
    "vo_valid",
    "camera_dx_px",
    "camera_dy_px",
    "gt_x",
    "gt_y",
    "gt_z",
    "calibration_ready",
    "calibration_sample_count",
    "scale_estimate",
    "pred_x",
    "pred_y",
    "pred_z",
    "position_error",
    "status",
    "failure_reason",
)


@dataclass(frozen=True, slots=True)
class ValidationOptions:
    images_dir: Path
    output_dir: Path
    synthetic_scale: float = 0.01
    synthetic_rotation_deg: float = 15.0
    synthetic_z: float = 100.0
    healthy_frames: int | None = None
    loss_frames: int | None = None
    recovery_frames: int | None = None


@dataclass(frozen=True, slots=True)
class ScenarioPartition:
    healthy_count: int
    loss_count: int
    recovery_count: int

    @property
    def loss_start(self) -> int:
        return self.healthy_count

    @property
    def recovery_start(self) -> int:
        return self.healthy_count + self.loss_count


@dataclass(frozen=True, slots=True)
class OfflineFrame:
    name: str
    source: str
    content: bytes


class ReplayAffineVO:
    """Replays results produced by production AffineVO; it does not estimate motion."""

    def __init__(self, motions: Sequence[AffineMotionResult]) -> None:
        self._motions = list(motions)
        self.calls = 0

    def estimate(self, *_args: object) -> AffineMotionResult:
        if self.calls >= len(self._motions):
            return AffineMotionResult(failure_reason="replay_exhausted")
        result = self._motions[self.calls]
        self.calls += 1
        return result


class FixedCameraProvider:
    def __init__(self, width: int, height: int) -> None:
        self._camera = CameraModel(
            width=width,
            height=height,
            fx=max(width, 1),
            fy=max(height, 1),
            cx=width / 2,
            cy=height / 2,
            distortion=(),
        )

    def for_resolution(self, width: int, height: int) -> CameraModel | None:
        return self._camera if (width, height) == (self._camera.width, self._camera.height) else None


PrepassProcessor = Callable[..., Awaitable[tuple[list[dict[str, object]], list[object], list[object]]]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production Localization GPS loss/recovery offline validator"
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--synthetic-scale", type=float, default=0.01)
    parser.add_argument("--synthetic-rotation-deg", type=float, default=15.0)
    parser.add_argument("--synthetic-z", type=float, default=100.0)
    parser.add_argument("--healthy-frames", type=int)
    parser.add_argument("--loss-frames", type=int)
    parser.add_argument("--recovery-frames", type=int)
    return parser


def validate_options(options: ValidationOptions) -> None:
    values = (options.synthetic_scale, options.synthetic_rotation_deg, options.synthetic_z)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("synthetic transform values must be finite")
    if options.synthetic_scale <= 0:
        raise ValueError("synthetic scale must be positive")
    split = (options.healthy_frames, options.loss_frames, options.recovery_frames)
    configured = sum(value is not None for value in split)
    if configured not in {0, 3}:
        raise ValueError(
            "--healthy-frames, --loss-frames and --recovery-frames must be provided together"
        )
    if configured:
        assert all(value is not None for value in split)
        if options.healthy_frames <= 0:
            raise ValueError("--healthy-frames must be greater than zero")
        if options.loss_frames <= 0:
            raise ValueError("--loss-frames must be greater than zero")
        if options.recovery_frames < 0:
            raise ValueError("--recovery-frames cannot be negative")


def make_partition(
    frame_count: int,
    healthy_frames: int | None = None,
    loss_frames: int | None = None,
    recovery_frames: int | None = None,
) -> ScenarioPartition:
    custom = (healthy_frames, loss_frames, recovery_frames)
    if any(value is not None for value in custom):
        if not all(value is not None for value in custom):
            raise ValueError("custom frame split requires healthy, loss and recovery counts")
        assert healthy_frames is not None and loss_frames is not None and recovery_frames is not None
        if healthy_frames <= 0 or loss_frames <= 0 or recovery_frames < 0:
            raise ValueError("custom frame split requires healthy>0, loss>0 and recovery>=0")
        requested = healthy_frames + loss_frames + recovery_frames
        if requested > frame_count:
            raise ValueError(
                f"requested frame split ({requested}) exceeds available frame count ({frame_count})"
            )
        if requested < frame_count:
            raise ValueError(
                f"requested frame split ({requested}) must equal available frame count ({frame_count})"
            )
        return ScenarioPartition(healthy_frames, loss_frames, recovery_frames)
    if frame_count < 14:
        raise ValueError("at least 14 frames are required (9 healthy, 3 loss, 2 recovery)")
    if frame_count >= 24:
        return ScenarioPartition(healthy_count=12, loss_count=6, recovery_count=frame_count - 18)
    recovery = max(2, frame_count // 4)
    loss = max(3, frame_count // 4)
    healthy = frame_count - loss - recovery
    if healthy < 9:
        recovery, loss, healthy = 2, 3, frame_count - 5
    return ScenarioPartition(healthy, loss, recovery)


def _rotation_matrix(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )


def _motion_from_frame_report(frame: dict[str, object]) -> AffineMotionResult:
    diagnostic = frame["vo_diagnostics"]
    assert isinstance(diagnostic, dict)
    return AffineMotionResult(
        delta_x_px=_finite_or_none(diagnostic.get("translation_x_px")),
        delta_y_px=_finite_or_none(diagnostic.get("translation_y_px")),
        delta_yaw_rad=_finite_or_none(diagnostic.get("rotation_yaw_rad")),
        tracked_points=int(diagnostic.get("tracked_points") or 0),
        inlier_count=int(diagnostic.get("inlier_count") or 0),
        inlier_ratio=float(diagnostic.get("inlier_ratio") or 0.0),
        rms_residual=_finite_or_none(diagnostic.get("reprojection_error")),
        quality_valid=bool(diagnostic.get("transform_valid")),
        failure_reason=diagnostic.get("failure_reason"),
    )


def truth_from_prepass(
    frames: Sequence[dict[str, object]],
    *,
    scale: float,
    rotation_deg: float,
    z: float,
    origin: tuple[float, float] = (1000.0, 2000.0),
) -> list[tuple[float, float, float]]:
    rotation = _rotation_matrix(rotation_deg)
    truth: list[tuple[float, float, float]] = []
    for frame in frames:
        pose = frame["pose"]
        assert isinstance(pose, dict)
        camera = np.asarray(
            [float(pose["cumulative_dx_px"]), float(pose["cumulative_dy_px"])],
            dtype=np.float64,
        )
        xy = np.asarray(origin) + scale * (rotation @ camera)
        truth.append((float(xy[0]), float(xy[1]), float(z)))
    return truth


async def run_service_scenario(
    settings: Settings,
    frames: Sequence[OfflineFrame],
    motions: Sequence[AffineMotionResult],
    truth: Sequence[tuple[float, float, float]],
    partition: ScenarioPartition,
    *,
    session_id: str,
    camera_provider: object | None = None,
    image_decoder: Callable[[bytes], object] | None = None,
) -> dict[str, object]:
    if len(frames) != len(truth) or len(motions) != max(0, len(frames) - 1):
        raise ValueError("frame, truth and motion sequence lengths are inconsistent")
    content = {item.source: item.content for item in frames}

    async def local_reader(source: str, _timeout: float) -> bytes:
        if source not in content:
            raise ValueError("only preloaded offline frames are accepted")
        return content[source]

    service = AffineLocalizationService(
        settings,
        vo=ReplayAffineVO(motions),
        camera_provider=camera_provider,
        image_reader=local_reader,
        **({"image_decoder": image_decoder} if image_decoder is not None else {}),
    )
    state = LocalizationSessionState(session_id)
    frame_rows: list[dict[str, object]] = []
    calibration_history: list[dict[str, object]] = []
    outputs: list[DetectedTranslation | None] = []

    for index, offline in enumerate(frames):
        healthy = index < partition.loss_start or index >= partition.recovery_start
        gps = truth[index] if healthy else None
        before = state.vo_state
        before_sample_count = len(before.calibration_samples) if before else 0
        before_last_estimate = before.last_estimate if before else None
        frame = FrameContext(
            frame_id=f"{session_id}-{index}",
            image_url=offline.source,
            video_name=session_id,
            session_id=session_id,
            gps_health_status=GPSHealthStatus.HEALTHY if healthy else GPSHealthStatus.UNHEALTHY,
            gps_x=gps[0] if gps else None,
            gps_y=gps[1] if gps else None,
            gps_z=gps[2] if gps else None,
            frame_index=index,
            image_modality=ImageModality.RGB,
        )
        result = await service.process_frame(frame, state)
        outputs.append(result)
        current = state.vo_state
        assert current is not None
        motion = current.last_motion_result if index else None
        sample_count = len(current.calibration_samples)
        accepted = sample_count > before_sample_count
        reject_reason = _sample_reject_reason(
            service, index, healthy, state.last_gps_health_status, before, current, motion, truth
        ) if not accepted else None
        calibration = _serialize_calibration(current.calibration_result)
        calibration_history.append({"frame_index": index, **calibration})
        expected = truth[index]
        position_error = None
        if not healthy and result is not None:
            position_error = math.hypot(
                result.translation_x - expected[0], result.translation_y - expected[1]
            )
        finite_output = result is None or all(
            math.isfinite(value)
            for value in (result.translation_x, result.translation_y, result.translation_z)
        )
        stale = bool(not healthy and motion is not None and not motion.quality_valid and result is not None)
        clamp_applied = _clamp_applied(service, current, before_last_estimate, result) if not healthy else False
        status = (
            "gps_ground_truth"
            if healthy and result is not None
            else "gps_loss_prediction"
            if result is not None
            else "gps_loss_unavailable"
            if not healthy
            else "ground_truth_unavailable"
        )
        frame_rows.append(
            {
                "frame_index": index,
                "frame_name": offline.name,
                "gps_healthy": healthy,
                "ground_truth_x": expected[0],
                "ground_truth_y": expected[1],
                "ground_truth_z": expected[2],
                "vo_valid": bool(motion and motion.quality_valid),
                "camera_dx_px": motion.delta_x_px if motion else None,
                "camera_dy_px": motion.delta_y_px if motion else None,
                "camera_yaw": motion.delta_yaw_rad if motion else None,
                "tracked_points": motion.tracked_points if motion else 0,
                "inliers": motion.inlier_count if motion else 0,
                "inlier_ratio": motion.inlier_ratio if motion else 0.0,
                "rms_residual": motion.rms_residual if motion else None,
                "calibration_sample_accepted": accepted,
                "calibration_sample_rejected": bool(index and healthy and not accepted),
                "calibration_reject_reason": reject_reason,
                "calibration_sample_count": sample_count,
                "calibration_inlier_count": calibration["inlier_count"],
                "calibration_inlier_ratio": calibration["inlier_ratio"],
                "calibration_scale": calibration["scale"],
                "calibration_rotation_deg": calibration["rotation_deg"],
                "calibration_rms": calibration["rms_residual"],
                "calibration_directional_diversity": calibration["directional_diversity"],
                "calibration_ready": calibration["ready"],
                "cumulative_camera_x_px": current.cumulative_dx_px,
                "cumulative_camera_y_px": current.cumulative_dy_px,
                "gps_anchor": list(current.gps_anchor) if current.gps_anchor else None,
                "camera_anchor": list(current.camera_anchor) if current.camera_anchor else None,
                "calibration_frozen": current.calibration_frozen,
                "predicted_translation_x": result.translation_x if result else None,
                "predicted_translation_y": result.translation_y if result else None,
                "predicted_translation_z": result.translation_z if result else None,
                "absolute_error_x": abs(result.translation_x - expected[0]) if result and not healthy else None,
                "absolute_error_y": abs(result.translation_y - expected[1]) if result and not healthy else None,
                "position_error": position_error,
                "clamp_applied": clamp_applied,
                "non_finite_output": not finite_output,
                "stale_result": stale,
                "status": status,
                "failure_reason": motion.failure_reason if motion and not motion.quality_valid else None,
                "gps_health_transition": current.gps_health_transition,
                "recovery_healthy_count": current.recovery_healthy_count,
            }
        )
    return _scenario_report(
        settings, frame_rows, calibration_history, outputs, truth, partition
    )


def _sample_reject_reason(service, index, healthy, _current_health, previous, current, motion, truth):
    if not healthy or index == 0:
        return None
    if motion is None or not motion.quality_valid:
        return "motion_quality_invalid"
    if previous is None or previous.previous_valid_gps is None:
        return "unmatched_gps_step"
    if index > 0 and index - 1 >= 0:
        from app.services.localization.calibration import make_calibration_sample

        _, reason = make_calibration_sample(
            frame_index=index,
            sequence=current.frame_count,
            camera_delta=(
                current.cumulative_dx_px - previous.cumulative_dx_px,
                current.cumulative_dy_px - previous.cumulative_dy_px,
            ),
            gps_delta=(truth[index][0] - truth[index - 1][0], truth[index][1] - truth[index - 1][1]),
            motion=motion,
            yaw_rad=current.cumulative_yaw,
            policy=service._calibration_policy,
        )
        return reason or "unmatched_gps_step"
    return "unmatched_gps_step"


def _clamp_applied(service, current, previous_estimate, result):
    if result is None or previous_estimate is None:
        return False
    calibration = current.frozen_calibration_result if current.calibration_frozen else current.calibration_result
    if not calibration or not calibration.ready or not current.gps_anchor or not current.camera_anchor:
        return False
    rotation = np.asarray(calibration.rotation_matrix_2x2, dtype=np.float64)
    camera_delta = np.asarray((current.cumulative_dx_px, current.cumulative_dy_px)) - np.asarray(current.camera_anchor)
    raw = np.asarray(current.gps_anchor[:2]) + float(calibration.scale) * (rotation @ camera_delta)
    return not np.allclose(raw, (result.translation_x, result.translation_y), atol=1e-9)


def _serialize_calibration(calibration) -> dict[str, object]:
    if calibration is None:
        return {
            "ready": False, "sample_count": 0, "inlier_count": 0, "inlier_ratio": 0.0,
            "scale": None, "rotation_deg": None, "rms_residual": None,
            "directional_diversity": 0.0, "failure_reason": "insufficient_samples",
        }
    rotation_deg = None
    if calibration.rotation_matrix_2x2 is not None:
        matrix = calibration.rotation_matrix_2x2
        rotation_deg = math.degrees(math.atan2(matrix[1][0], matrix[0][0]))
    return {
        "ready": calibration.ready,
        "sample_count": calibration.sample_count,
        "inlier_count": calibration.inlier_count,
        "inlier_ratio": calibration.inlier_count / calibration.sample_count if calibration.sample_count else 0.0,
        "scale": calibration.scale,
        "rotation_deg": rotation_deg,
        "rms_residual": calibration.rms_residual,
        "directional_diversity": calibration.directional_diversity,
        "failure_reason": calibration.failure_reason,
    }


def _scenario_report(settings, rows, calibration_history, outputs, truth, partition):
    loss_rows = rows[partition.loss_start : partition.recovery_start]
    recovery_rows = rows[partition.recovery_start :]
    loss_errors = [float(row["position_error"]) for row in loss_rows if row["position_error"] is not None]
    ready_frames = [row["frame_index"] for row in rows if row["calibration_ready"]]
    ready_before_loss = bool(rows[partition.loss_start - 1]["calibration_ready"])
    first_loss = loss_rows[0]
    first_motion_preserved = bool(
        first_loss["vo_valid"]
        and first_loss["predicted_translation_x"] is not None
        and first_loss["camera_anchor"] is not None
        and not np.allclose(
            first_loss["camera_anchor"],
            (first_loss["cumulative_camera_x_px"], first_loss["cumulative_camera_y_px"]),
        )
    )
    z_correct = all(
        row["predicted_translation_z"] is None
        or math.isclose(float(row["predicted_translation_z"]), truth[partition.loss_start - 1][2])
        for row in loss_rows
    )
    recovery_truth = (
        all(
            math.isclose(float(recovery_rows[0][key]), truth[partition.recovery_start][offset])
            for key, offset in (("predicted_translation_x", 0), ("predicted_translation_y", 1), ("predicted_translation_z", 2))
        )
        if recovery_rows
        else None
    )
    stale = any(bool(row["stale_result"]) for row in rows)
    finite = not any(bool(row["non_finite_output"]) for row in rows)
    compatibility = _official_compatibility(outputs, rows)
    final_calibration = calibration_history[partition.loss_start - 1]
    final_result = all((
        ready_before_loss,
        first_motion_preserved,
        z_correct,
        recovery_truth is not False,
        not stale,
        finite,
        compatibility["compatible"],
    ))
    return {
        "frames": rows,
        "vo_diagnostics": [
            {key: row[key] for key in (
                "frame_index", "frame_name", "vo_valid", "camera_dx_px", "camera_dy_px",
                "camera_yaw", "tracked_points", "inliers", "inlier_ratio", "rms_residual", "failure_reason"
            )}
            for row in rows
        ],
        "calibration_history": calibration_history,
        "gps_loss_transition": {
            "frame_index": first_loss["frame_index"],
            "last_valid_gps_anchor": first_loss["gps_anchor"],
            "camera_anchor": first_loss["camera_anchor"],
            "frozen_calibration_snapshot": final_calibration,
            "first_unhealthy_frame_vo_delta": [first_loss["camera_dx_px"], first_loss["camera_dy_px"]],
            "vo_step_included": first_motion_preserved,
            "translation": [first_loss["predicted_translation_x"], first_loss["predicted_translation_y"], first_loss["predicted_translation_z"]],
        },
        "gps_loss_predictions": loss_rows,
        "recovery": {
            "first_frame_index": recovery_rows[0]["frame_index"] if recovery_rows else None,
            "ground_truth_returned_immediately": recovery_truth,
            "anchor_renewed": bool(recovery_rows and recovery_rows[0]["gps_anchor"] == list(truth[partition.recovery_start])),
            "calibration_history_preserved": bool(recovery_rows and recovery_rows[0]["calibration_sample_count"] >= final_calibration["sample_count"]),
            "recovery_count": recovery_rows[0]["recovery_healthy_count"] if recovery_rows else 0,
            "stale_prediction": bool(recovery_rows and recovery_rows[0]["stale_result"]),
        },
        "error_metrics": {
            "valid_prediction_count": len(loss_errors),
            "none_count": sum(row["predicted_translation_x"] is None for row in loss_rows),
            "mean_position_error": float(np.mean(loss_errors)) if loss_errors else None,
            "p95_position_error": float(np.percentile(loss_errors, 95)) if loss_errors else None,
            "max_position_error": max(loss_errors) if loss_errors else None,
        },
        "official_schema_compatibility": compatibility,
        "final_summary": {
            "frame_count": len(rows),
            "healthy_gps_frames": partition.healthy_count,
            "gps_loss_frames": partition.loss_count,
            "recovery_frames": partition.recovery_count,
            "vo_valid_count": sum(bool(row["vo_valid"]) for row in rows),
            "vo_valid_steps_during_healthy_gps": sum(
                bool(row["vo_valid"]) for row in rows[: partition.loss_start]
            ),
            "calibration_samples": final_calibration["sample_count"],
            "calibration_ready_frame": ready_frames[0] if ready_frames else None,
            "calibration_ready_before_loss": ready_before_loss,
            "calibration_ready": ready_before_loss,
            "estimated_scale": final_calibration["scale"],
            "estimated_rotation_deg": final_calibration["rotation_deg"],
            "first_unhealthy_frame_movement_preserved": first_motion_preserved,
            "z_policy": settings.localization_z_policy,
            "z_policy_correct": z_correct,
            "recovery_first_frame_returned_ground_truth": recovery_truth,
            "stale_prediction_detected": stale,
            "finite_outputs": finite,
            "final_result": "PASS" if final_result else "FAIL",
            "prediction_submission": "DISABLED",
        },
    }


def _official_compatibility(outputs, rows):
    try:
        serialized = []
        for index, (output, row) in enumerate(zip(outputs, rows), start=1):
            response = CompetitionResponse.from_task_results(
                response_id=index,
                user="offline-validation",
                frame=str(row["frame_name"]),
                detected_objects=[],
                detected_translation=output,
                matched_reference_objects=[],
            )
            serialized.append(response.model_dump(mode="json")["detected_translations"])
        return {"compatible": True, "serialized_results": serialized}
    except Exception as exc:
        return {"compatible": False, "error": f"{type(exc).__name__}: {exc}"}


def _controlled_inputs(frame_count: int, invalid_frame: int | None = None):
    steps = ((4, 0), (0, 4), (-3, 2), (2, -3), (5, 1), (1, 5), (-4, 1), (2, 4), (-2, -4), (4, -2), (3, 3))
    motions = []
    for frame_index in range(1, frame_count):
        if frame_index == invalid_frame:
            motions.append(AffineMotionResult(failure_reason="low_quality"))
        else:
            dx, dy = steps[(frame_index - 1) % len(steps)]
            motions.append(AffineMotionResult(
                delta_x_px=float(dx), delta_y_px=float(dy), delta_yaw_rad=0.0,
                tracked_points=100, inlier_count=90, inlier_ratio=0.9,
                rms_residual=0.1, quality_valid=True,
            ))
    arrays = []
    frames = []
    for index in range(frame_count):
        image = np.full((64, 64, 3), index % 255, dtype=np.uint8)
        arrays.append(image)
        frames.append(OfflineFrame(f"synthetic_{index:03d}.png", f"synthetic://{index}", str(index).encode()))
    decoder = lambda data: arrays[int(data.decode())].copy()
    return frames, motions, decoder


def _camera_positions(motions: Sequence[AffineMotionResult]) -> list[tuple[float, float]]:
    x = y = yaw = 0.0
    positions = [(x, y)]
    for motion in motions:
        if motion.quality_valid:
            yaw += float(motion.delta_yaw_rad or 0.0)
            cosine, sine = math.cos(-yaw), math.sin(-yaw)
            dx, dy = float(motion.delta_x_px or 0.0), float(motion.delta_y_px or 0.0)
            x += cosine * dx - sine * dy
            y += sine * dx + cosine * dy
        positions.append((x, y))
    return positions


def _truth_from_positions(positions, scale, rotation_deg, z, outliers=()):
    rotation = _rotation_matrix(rotation_deg)
    truth = []
    for index, position in enumerate(positions):
        xy = np.asarray((100.0, 200.0)) + scale * (rotation @ np.asarray(position))
        if index in outliers:
            xy += np.asarray((4.0, -3.0))
        truth.append((float(xy[0]), float(xy[1]), float(z)))
    return truth


async def run_controlled_scenario(
    settings: Settings,
    *,
    scale: float = 0.01,
    rotation_deg: float = 15.0,
    z: float = 100.0,
    frame_count: int = 24,
    healthy_count: int = 12,
    loss_count: int = 6,
    invalid_frame: int | None = 14,
    outliers: Sequence[int] = (),
    session_id: str = "task2-gps-controlled",
) -> dict[str, object]:
    partition = ScenarioPartition(healthy_count, loss_count, frame_count - healthy_count - loss_count)
    frames, motions, decoder = _controlled_inputs(frame_count, invalid_frame)
    truth = _truth_from_positions(_camera_positions(motions), scale, rotation_deg, z, outliers)
    return await run_service_scenario(
        settings, frames, motions, truth, partition, session_id=session_id,
        camera_provider=FixedCameraProvider(64, 64), image_decoder=decoder,
    )


async def run_validation(
    settings: Settings,
    options: ValidationOptions,
    *,
    emit: Callable[[str], None] = print,
    prepass_processor: PrepassProcessor = process_loaded_sequence,
) -> dict[str, object]:
    validate_options(options)
    settings.validate_localization_vo()
    loaded, rejected = discover_and_validate_frames(options.images_dir, settings)
    partition = make_partition(
        len(loaded),
        options.healthy_frames,
        options.loss_frames,
        options.recovery_frames,
    )
    prepass_frames, _, _ = await prepass_processor(
        settings, loaded, session_id="task2-gps-prepass", video_name="task2-gps-prepass", respect_enabled=True
    )
    if len(prepass_frames) != len(loaded):
        raise ValueError("production VO prepass did not return one report per frame")
    motions = [_motion_from_frame_report(frame) for frame in prepass_frames[1:]]
    truth = truth_from_prepass(
        prepass_frames, scale=options.synthetic_scale,
        rotation_deg=options.synthetic_rotation_deg, z=options.synthetic_z,
    )
    offline = [OfflineFrame(item.path.name, str(item.path), item.content) for item in loaded]
    real = await run_service_scenario(
        settings, offline, motions, truth, partition, session_id="task2-gps-real-sequence"
    )

    controlled = await run_controlled_scenario(
        settings, scale=options.synthetic_scale, rotation_deg=options.synthetic_rotation_deg,
        z=options.synthetic_z,
    )
    insufficient = await run_controlled_scenario(
        settings, scale=options.synthetic_scale, rotation_deg=options.synthetic_rotation_deg,
        z=options.synthetic_z, frame_count=9, healthy_count=4, loss_count=3,
        invalid_frame=None, session_id="task2-gps-insufficient",
    )
    outlier = await run_controlled_scenario(
        settings, scale=options.synthetic_scale, rotation_deg=options.synthetic_rotation_deg,
        z=options.synthetic_z, invalid_frame=None, outliers=(5,), session_id="task2-gps-outlier",
    )
    report = {
        "config": task2_config_report(settings),
        "synthetic_truth_config": {
            "label": "controlled synthetic GPS input; not a real-world accuracy test",
            "scale": options.synthetic_scale,
            "rotation_deg": options.synthetic_rotation_deg,
            "z": options.synthetic_z,
        },
        "input": {
            "images_dir": str(options.images_dir.resolve()),
            "frame_count": len(loaded),
            "resolution_rejections": rejected,
            "frame_split": {
                "healthy_frames": partition.healthy_count,
                "loss_frames": partition.loss_count,
                "recovery_frames": partition.recovery_count,
                "source": "cli" if options.healthy_frames is not None else "automatic",
            },
        },
        **real,
        "controlled_fixture_validation": controlled,
        "insufficient_calibration_scenario": insufficient,
        "outlier_calibration_scenario": outlier,
    }
    _add_accuracy(report, options)
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "task2_gps_loss_recovery.json"
    csv_path = output_dir / "task2_gps_loss_recovery_sequence.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, report["frames"])
    _emit_summary(report, json_path, csv_path, emit)
    return report


def _add_accuracy(report, options):
    for key in ("controlled_fixture_validation", "outlier_calibration_scenario"):
        scenario = report[key]
        summary = scenario["final_summary"]
        estimated_scale = summary["estimated_scale"]
        estimated_rotation = summary["estimated_rotation_deg"]
        summary["expected_scale"] = options.synthetic_scale
        summary["scale_absolute_error"] = abs(estimated_scale - options.synthetic_scale) if estimated_scale is not None else None
        summary["expected_rotation_deg"] = options.synthetic_rotation_deg
        summary["rotation_error_deg"] = _angle_error(estimated_rotation, options.synthetic_rotation_deg) if estimated_rotation is not None else None


def _angle_error(actual, expected):
    return abs((float(actual) - float(expected) + 180.0) % 360.0 - 180.0)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "gt_x": row["ground_truth_x"], "gt_y": row["ground_truth_y"], "gt_z": row["ground_truth_z"],
                "calibration_ready": row["calibration_ready"],
                "scale_estimate": row["calibration_scale"],
                "pred_x": row["predicted_translation_x"], "pred_y": row["predicted_translation_y"], "pred_z": row["predicted_translation_z"],
            })


def _emit_summary(report, json_path, csv_path, emit):
    real = report["final_summary"]
    real_errors = report["error_metrics"]
    controlled = report["controlled_fixture_validation"]["final_summary"]
    errors = report["controlled_fixture_validation"]["error_metrics"]
    emit("===== TASK 2 GPS LOSS / RECOVERY VALIDATION =====")
    emit(f"Frames: {real['frame_count']}")
    emit(f"Healthy GPS frames: {real['healthy_gps_frames']}")
    emit(f"GPS loss frames: {real['gps_loss_frames']}")
    emit(f"Recovery frames: {real['recovery_frames']}")
    emit(f"VO valid: {real['vo_valid_count']}")
    emit(f"VO valid steps during healthy GPS: {real['vo_valid_steps_during_healthy_gps']}")
    emit(f"Calibration samples before loss: {real['calibration_samples']}")
    emit(f"Calibration ready frame: {real['calibration_ready_frame']}")
    emit(f"Calibration ready: {_yes_no(real['calibration_ready'])}")
    emit(f"GPS-loss valid predictions: {real_errors['valid_prediction_count']}")
    emit(f"GPS-loss None count: {real_errors['none_count']}")
    emit(f"First unhealthy frame movement preserved: {_yes_no(real['first_unhealthy_frame_movement_preserved'])}")
    emit(f"Mean 2D error: {_fmt(real_errors['mean_position_error'])}")
    emit(f"P95 2D error: {_fmt(real_errors['p95_position_error'])}")
    emit(f"Max 2D error: {_fmt(real_errors['max_position_error'])}")
    emit(f"Z policy correct: {_yes_no(real['z_policy_correct'])}")
    emit(f"Expected scale: {controlled['expected_scale']}")
    emit(f"Estimated scale: {_fmt(controlled['estimated_scale'])}")
    emit(f"Scale error: {_fmt(controlled['scale_absolute_error'])}")
    emit(f"Expected rotation: {controlled['expected_rotation_deg']} deg")
    emit(f"Estimated rotation: {_fmt(controlled['estimated_rotation_deg'])} deg")
    emit(f"Rotation error: {_fmt(controlled['rotation_error_deg'])} deg")
    emit(f"Controlled GPS-loss valid predictions: {errors['valid_prediction_count']}")
    emit(f"Controlled GPS-loss None count: {errors['none_count']}")
    emit(f"Controlled GPS-loss mean 2D error: {_fmt(errors['mean_position_error'])}")
    emit(f"Controlled GPS-loss p95 error: {_fmt(errors['p95_position_error'])}")
    emit(f"Controlled GPS-loss max error: {_fmt(errors['max_position_error'])}")
    emit(f"Controlled first unhealthy frame movement preserved: {_yes_no(controlled['first_unhealthy_frame_movement_preserved'])}")
    emit(f"Controlled Z policy correct: {_yes_no(controlled['z_policy_correct'])}")
    emit(f"Controlled recovery first frame returned ground truth: {_yes_no(controlled['recovery_first_frame_returned_ground_truth'])}")
    emit(f"Controlled stale prediction detected: {_yes_no(controlled['stale_prediction_detected'])}")
    emit(f"Real sequence result: {real['final_result']}")
    emit(f"Controlled fixture result: {controlled['final_result']}")
    emit(f"JSON: {json_path}")
    emit(f"CSV: {csv_path}")
    emit("Prediction submission: DISABLED")


def _finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value):
    return "n/a" if value is None else f"{float(value):.8f}"


def _yes_no(value):
    return "YES" if value else "NO"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = ValidationOptions(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        synthetic_scale=args.synthetic_scale,
        synthetic_rotation_deg=args.synthetic_rotation_deg,
        synthetic_z=args.synthetic_z,
        healthy_frames=args.healthy_frames,
        loss_frames=args.loss_frames,
        recovery_frames=args.recovery_frames,
    )
    try:
        asyncio.run(run_validation(get_settings(), options))
    except Exception as exc:
        print(f"Task 2 GPS loss/recovery validation: FAIL ({type(exc).__name__}: {exc})")
        print("Prediction submission: DISABLED")
        return EXIT_INPUT
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
