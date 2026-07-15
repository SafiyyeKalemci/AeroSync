from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings, get_settings
from app.schemas import CompetitionResponse, DetectedTranslation, ImageModality
from app.services.common import FrameContext
from app.services.localization.affine_vo import AffineVisualOdometry
from app.services.localization.alignment import fit_calibration
from app.services.localization.calibration import make_calibration_sample
from app.services.localization.camera_model import CameraModel, CameraModelProvider
from app.services.localization.continuity import evaluate_continuity
from app.services.localization.interface import LocalizationSessionState
from app.services.localization.service import AffineLocalizationService
from app.services.localization.state import AffineMotionResult, VisualOdometryState
from scripts.validate_task1_detection import LoadedImage, load_local_image

EXIT_OK = 0
EXIT_CONFIG = 10
EXIT_INPUT = 20

CSV_COLUMNS = (
    "frame_index",
    "frame_name",
    "vo_valid",
    "dx",
    "dy",
    "dz",
    "cumulative_x",
    "cumulative_y",
    "cumulative_z",
    "cumulative_dx_px",
    "cumulative_dy_px",
    "rotation",
    "inliers",
    "inlier_ratio",
    "gps_scale_ready",
    "gps_scale",
    "status",
    "failure_reason",
)


@dataclass(frozen=True, slots=True)
class ValidationOptions:
    output_dir: Path
    image: Path | None = None
    next_image: Path | None = None
    additional_images: tuple[Path, ...] = ()
    images_dir: Path | None = None
    synthetic: bool = False
    synthetic_gps: bool = False
    save_visualizations: bool = False


@dataclass(frozen=True, slots=True)
class VOCall:
    previous_gray: np.ndarray
    current_gray: np.ndarray
    camera: CameraModel
    result: AffineMotionResult


class RecordingAffineVO:
    """Transparent instrumentation around the production AffineVisualOdometry."""

    def __init__(self, vo: AffineVisualOdometry) -> None:
        self._vo = vo
        self.calls: list[VOCall] = []

    def estimate(self, previous_gray, current_gray, camera):
        result = self._vo.estimate(previous_gray, current_gray, camera)
        self.calls.append(VOCall(previous_gray, current_gray, camera, result))
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production AffineVO/localization pipeline icin tamamen offline validator"
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--next-image", type=Path)
    parser.add_argument("--additional-image", action="append", type=Path, default=[])
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-gps", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--save-visualizations", action="store_true")
    return parser


def task2_config_report(settings: Settings) -> dict[str, object]:
    return {
        "LOCALIZATION_ENABLED": settings.localization_enabled,
        "LOCALIZATION_VO_ENABLED": settings.localization_vo_enabled,
        "LOCALIZATION_MIN_FEATURES": settings.localization_min_features,
        "LOCALIZATION_MAX_FEATURES": settings.localization_max_features,
        "LOCALIZATION_FEATURE_QUALITY_LEVEL": settings.localization_feature_quality_level,
        "LOCALIZATION_FEATURE_MIN_DISTANCE": settings.localization_feature_min_distance,
        "LOCALIZATION_LK_WIN_SIZE": settings.localization_lk_win_size,
        "LOCALIZATION_LK_MAX_LEVEL": settings.localization_lk_max_level,
        "LOCALIZATION_LK_FB_ERROR_THRESHOLD": settings.localization_lk_fb_error_threshold,
        "LOCALIZATION_RANSAC_ITERATIONS": settings.localization_ransac_iterations,
        "LOCALIZATION_RANSAC_RESIDUAL_THRESHOLD": settings.localization_ransac_residual_threshold,
        "LOCALIZATION_MIN_INLIERS": settings.localization_min_inliers,
        "LOCALIZATION_MIN_INLIER_RATIO": settings.localization_min_inlier_ratio,
        "LOCALIZATION_MAX_FRAME_GAP": settings.localization_max_frame_gap,
        "LOCALIZATION_WARMUP_FRAMES": settings.localization_warmup_frames,
        "LOCALIZATION_FREEZE_THRESHOLD": settings.localization_freeze_threshold,
        "LOCALIZATION_CAMERA_WIDTH": settings.localization_camera_width,
        "LOCALIZATION_CAMERA_HEIGHT": settings.localization_camera_height,
        "LOCALIZATION_CAMERA_FX": settings.localization_camera_fx,
        "LOCALIZATION_CAMERA_FY": settings.localization_camera_fy,
        "LOCALIZATION_CAMERA_CX": settings.localization_camera_cx,
        "LOCALIZATION_CAMERA_CY": settings.localization_camera_cy,
        "LOCALIZATION_CAMERA_DISTORTION": settings.localization_camera_distortion,
        "LOCALIZATION_CALIBRATION_ENABLED": settings.localization_calibration_enabled,
        "LOCALIZATION_CALIBRATION_MIN_SAMPLES": settings.localization_calibration_min_samples,
        "LOCALIZATION_CALIBRATION_MAX_SAMPLES": settings.localization_calibration_max_samples,
        "LOCALIZATION_CALIBRATION_MIN_CAMERA_STEP_PX": settings.localization_calibration_min_camera_step_px,
        "LOCALIZATION_CALIBRATION_MIN_GPS_STEP": settings.localization_calibration_min_gps_step,
        "LOCALIZATION_CALIBRATION_MAX_RMS_RESIDUAL": settings.localization_calibration_max_rms_residual,
        "LOCALIZATION_CALIBRATION_MIN_INLIER_RATIO": settings.localization_calibration_min_inlier_ratio,
        "LOCALIZATION_CALIBRATION_MIN_DIRECTIONAL_DIVERSITY": settings.localization_calibration_min_directional_diversity,
        "LOCALIZATION_CALIBRATION_SCALE_MIN": settings.localization_calibration_scale_min,
        "LOCALIZATION_CALIBRATION_SCALE_MAX": settings.localization_calibration_scale_max,
        "LOCALIZATION_CALIBRATION_EXPECTED_MAX_FRAME": settings.localization_calibration_expected_max_frame,
        "LOCALIZATION_MAX_DELTA_PER_FRAME": settings.localization_max_delta_per_frame,
        "LOCALIZATION_Z_POLICY": settings.localization_z_policy,
    }


def camera_intrinsics_report(settings: Settings) -> dict[str, object]:
    provider = CameraModelProvider(settings)
    model = provider.for_resolution(
        settings.localization_camera_width, settings.localization_camera_height
    )
    values = (
        settings.localization_camera_fx,
        settings.localization_camera_fy,
        settings.localization_camera_cx,
        settings.localization_camera_cy,
    )
    checks = {
        "fx_finite_positive": math.isfinite(values[0]) and values[0] > 0,
        "fy_finite_positive": math.isfinite(values[1]) and values[1] > 0,
        "cx_inside_frame": math.isfinite(values[2]) and 0 <= values[2] <= settings.localization_camera_width,
        "cy_inside_frame": math.isfinite(values[3]) and 0 <= values[3] <= settings.localization_camera_height,
        "configured_resolution_positive": settings.localization_camera_width > 0 and settings.localization_camera_height > 0,
        "camera_model_available": model is not None,
    }
    return {
        "width": settings.localization_camera_width,
        "height": settings.localization_camera_height,
        "fx": settings.localization_camera_fx,
        "fy": settings.localization_camera_fy,
        "cx": settings.localization_camera_cx,
        "cy": settings.localization_camera_cy,
        "distortion": settings.localization_camera_distortion,
        "checks": checks,
        "valid": all(checks.values()),
    }


def affine_matrix_from_motion(motion: AffineMotionResult, camera: CameraModel) -> list[list[float]] | None:
    """Derive the matrix represented by production dx/dy/yaw; do not re-estimate it."""
    if not motion.quality_valid:
        return None
    assert motion.delta_x_px is not None and motion.delta_y_px is not None and motion.delta_yaw_rad is not None
    yaw = motion.delta_yaw_rad
    return [
        [1.0, yaw, -motion.delta_x_px - yaw * camera.cy],
        [-yaw, 1.0, -motion.delta_y_px + yaw * camera.cx],
    ]


def motion_diagnostics(motion: AffineMotionResult | None, camera: CameraModel | None = None) -> dict[str, object]:
    if motion is None:
        return {
            "transform_valid": False,
            "affine_matrix": None,
            "translation_x_px": None,
            "translation_y_px": None,
            "rotation_yaw_rad": None,
            "rotation_yaw_deg": None,
            "scale_estimate": None,
            "tracked_points": 0,
            "correspondence_count": 0,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "reprojection_error": None,
            "failure_reason": "initialization_or_reset",
            "feature_coordinates_available": False,
        }
    yaw = motion.delta_yaw_rad
    return {
        "transform_valid": motion.quality_valid,
        "affine_matrix": affine_matrix_from_motion(motion, camera) if camera else None,
        "translation_x_px": motion.delta_x_px,
        "translation_y_px": motion.delta_y_px,
        "rotation_yaw_rad": yaw,
        "rotation_yaw_deg": math.degrees(yaw) if yaw is not None else None,
        "scale_estimate": None,
        "scale_note": "production AffineVO estimates tx/ty/yaw, not affine scale",
        "tracked_points": motion.tracked_points,
        "correspondence_count": motion.tracked_points,
        "inlier_count": motion.inlier_count,
        "inlier_ratio": motion.inlier_ratio,
        "reprojection_error": motion.rms_residual,
        "failure_reason": motion.failure_reason,
        "feature_coordinates_available": False,
        "feature_note": "production result exposes counts but not point coordinates",
    }


async def process_loaded_sequence(
    settings: Settings,
    loaded_images: Sequence[LoadedImage],
    *,
    session_id: str = "task2-localization-validation",
    video_name: str = "task2-localization-validation",
    respect_enabled: bool = True,
) -> tuple[list[dict[str, object]], list[DetectedTranslation | None], list[VOCall]]:
    if respect_enabled and (
        not settings.localization_enabled or not settings.localization_vo_enabled
    ):
        frames = []
        for index, loaded in enumerate(loaded_images):
            compatible = (
                loaded.metadata["width"] == settings.localization_camera_width
                and loaded.metadata["height"] == settings.localization_camera_height
            )
            frames.append(
                _disabled_frame_report(index, loaded, settings, compatible)
            )
        return frames, [None] * len(frames), []

    content_by_path = {str(item.path): item.content for item in loaded_images}

    async def local_reader(source: str, _timeout: float) -> bytes:
        if source not in content_by_path:
            raise ValueError("only preloaded local frames are accepted")
        return content_by_path[source]

    service = AffineLocalizationService(settings, image_reader=local_reader)
    recorder = RecordingAffineVO(service._vo)
    service._vo = recorder
    state = LocalizationSessionState(session_id)
    frame_reports: list[dict[str, object]] = []
    outputs: list[DetectedTranslation | None] = []
    for index, loaded in enumerate(loaded_images):
        before = state.vo_state
        previous_available = bool(before and before.previous_gray is not None)
        before_call_count = len(recorder.calls)
        frame = FrameContext(
            frame_id=f"task2-local-frame-{index}",
            image_url=str(loaded.path),
            video_name=video_name,
            session_id=session_id,
            gps_health_status=None,
            gps_x=None,
            gps_y=None,
            gps_z=None,
            frame_index=index,
            image_modality=ImageModality.RGB,
        )
        result = await service.process_frame(frame, state)
        outputs.append(result)
        after = state.vo_state
        call = recorder.calls[-1] if len(recorder.calls) > before_call_count else None
        motion = call.result if call else (after.last_motion_result if after else None)
        camera = call.camera if call else CameraModelProvider(settings).for_resolution(
            int(loaded.metadata["width"]), int(loaded.metadata["height"])
        )
        accepted = bool(after and after.previous_frame_id == frame.frame_id)
        frame_reports.append(
            _frame_report(
                settings,
                index,
                loaded,
                previous_available,
                accepted,
                after,
                state.frame_count,
                motion,
                camera,
                result,
                implementation_diagnostic=not settings.localization_enabled,
            )
        )
    return frame_reports, outputs, recorder.calls


def _disabled_frame_report(index, loaded, settings, camera_compatible):
    return {
        "frame_index": index,
        "frame_name": loaded.path.name,
        "frame_path": str(loaded.path),
        "width": loaded.metadata["width"],
        "height": loaded.metadata["height"],
        "timestamp": loaded.path.stat().st_mtime,
        "localization_service_enabled": False,
        "implementation_diagnostic_mode": False,
        "initialization_state": "disabled_by_configuration",
        "service_frame_count": 0,
        "vo_frame_count": 0,
        "warmup_count": 0,
        "calibration_state": "inactive",
        "calibration_frame_count": 0,
        "calibration_expected_max_frame": settings.localization_calibration_expected_max_frame,
        "previous_frame_available": False,
        "current_frame_accepted": False,
        "camera_resolution_compatible": camera_compatible,
        "calibration_result": None,
        "vo_diagnostics": motion_diagnostics(None),
        "pose": _empty_pose("disabled"),
        "localization_result": None,
        "status": "disabled",
        "failure_reason": (
            "localization_disabled_by_configuration"
            if not settings.localization_enabled
            else "localization_vo_disabled_by_configuration"
        ),
    }


def _frame_report(settings, index, loaded, previous_available, accepted, state, envelope_frame_count, motion, camera, result, implementation_diagnostic):
    calibration = state.calibration_result if state else None
    calibration_ready = bool(calibration and calibration.ready)
    diagnostics = motion_diagnostics(motion, camera)
    result_payload = result.model_dump(mode="json") if result is not None else None
    status = (
        "initialization"
        if not previous_available and accepted
        else "vo_valid"
        if motion and motion.quality_valid
        else "safe_empty"
    )
    failure = diagnostics["failure_reason"]
    if camera is None:
        failure = "camera_resolution_incompatible"
        status = "rejected"
    pose = {
        "relative_dx_px": motion.delta_x_px if motion else None,
        "relative_dy_px": motion.delta_y_px if motion else None,
        "relative_dz": None,
        "relative_yaw_rad": motion.delta_yaw_rad if motion else None,
        "relative_pitch": None,
        "relative_roll": None,
        "cumulative_dx_px": state.cumulative_dx_px if state else 0.0,
        "cumulative_dy_px": state.cumulative_dy_px if state else 0.0,
        "cumulative_yaw_rad": state.cumulative_yaw if state else 0.0,
        "cumulative_x": result.translation_x if result else None,
        "cumulative_y": result.translation_y if result else None,
        "cumulative_z": result.translation_z if result else None,
        "status": status,
    }
    return {
        "frame_index": index,
        "frame_name": loaded.path.name,
        "frame_path": str(loaded.path),
        "width": loaded.metadata["width"],
        "height": loaded.metadata["height"],
        "timestamp": loaded.path.stat().st_mtime if loaded.path.exists() else None,
        "localization_service_enabled": settings.localization_enabled,
        "implementation_diagnostic_mode": implementation_diagnostic,
        "initialization_state": "initialized" if not previous_available and accepted else "tracking" if accepted else "rejected",
        "service_frame_count": envelope_frame_count,
        "vo_frame_count": state.frame_count if state else 0,
        "warmup_count": state.warmup_count if state else 0,
        "calibration_state": "ready" if calibration_ready else "collecting" if state and state.calibration_samples else "inactive",
        "calibration_frame_count": len(state.calibration_samples) if state else 0,
        "calibration_expected_max_frame": settings.localization_calibration_expected_max_frame,
        "previous_frame_available": previous_available,
        "current_frame_accepted": accepted,
        "camera_resolution_compatible": camera is not None,
        "calibration_result": _serialize_calibration(calibration),
        "vo_diagnostics": diagnostics,
        "pose": pose,
        "localization_result": result_payload,
        "status": status,
        "failure_reason": failure,
    }


def _empty_pose(status):
    return {
        "relative_dx_px": None,
        "relative_dy_px": None,
        "relative_dz": None,
        "relative_yaw_rad": None,
        "relative_pitch": None,
        "relative_roll": None,
        "cumulative_dx_px": 0.0,
        "cumulative_dy_px": 0.0,
        "cumulative_yaw_rad": 0.0,
        "cumulative_x": None,
        "cumulative_y": None,
        "cumulative_z": None,
        "status": status,
    }


def synthetic_gps_calibration(settings: Settings) -> dict[str, object]:
    service = AffineLocalizationService(settings)
    policy = service._calibration_policy
    camera_steps = (
        (10.0, 0.0),
        (0.0, 10.0),
        (-10.0, 0.0),
        (0.0, -10.0),
        (7.0, 7.0),
        (-7.0, 7.0),
        (-7.0, -7.0),
        (7.0, -7.0),
    )
    while len(camera_steps) < policy.min_samples:
        camera_steps += camera_steps
    camera_steps = camera_steps[: max(policy.min_samples, 8)]
    target_scale = max(policy.scale_min * 10, min(0.01, policy.scale_max / 10))
    motion = AffineMotionResult(
        delta_x_px=1.0,
        delta_y_px=1.0,
        delta_yaw_rad=0.0,
        tracked_points=100,
        inlier_count=95,
        inlier_ratio=0.95,
        rms_residual=0.1,
        quality_valid=True,
    )
    samples = []
    rejected: list[str] = []
    small, reason = make_calibration_sample(
        frame_index=0,
        sequence=0,
        camera_delta=(0.0, 0.0),
        gps_delta=(0.0, 0.0),
        motion=motion,
        yaw_rad=0.0,
        policy=policy,
    )
    if small is None and reason:
        rejected.append(reason)
    for index, camera_delta in enumerate(camera_steps, start=1):
        gps_delta = (target_scale * camera_delta[0], target_scale * camera_delta[1])
        sample, reason = make_calibration_sample(
            frame_index=index,
            sequence=index,
            camera_delta=camera_delta,
            gps_delta=gps_delta,
            motion=motion,
            yaw_rad=0.0,
            policy=policy,
        )
        if sample is not None:
            samples.append(sample)
        elif reason:
            rejected.append(reason)
    result = fit_calibration(samples, policy)
    return {
        "mode": "controlled_production_calibration_components",
        "active": settings.localization_calibration_enabled,
        "target_scale": target_scale,
        "sample_count": len(samples),
        "rejected_sample_count": len(rejected),
        "rejected_reasons": rejected,
        "scale_ready": result.ready,
        "estimated_scale": result.scale,
        "scale_confidence": _divide(result.inlier_count, result.sample_count),
        "scale_confidence_note": "derived calibration inlier ratio",
        "inlier_count": result.inlier_count,
        "rms_residual": result.rms_residual,
        "directional_diversity": result.directional_diversity,
        "rotation_matrix_2x2": result.rotation_matrix_2x2,
        "calibration_reason": result.failure_reason,
    }


async def run_synthetic_vo(settings: Settings) -> dict[str, object]:
    camera = CameraModelProvider(settings).for_resolution(
        settings.localization_camera_width, settings.localization_camera_height
    )
    if camera is None:
        raise ValueError("configured camera model is unavailable")
    service = AffineLocalizationService(settings)
    vo = service._vo
    base = _synthetic_feature_frame(camera.width, camera.height)
    translation_matrix = np.asarray([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]], np.float32)
    translated = cv2.warpAffine(base, translation_matrix, (camera.width, camera.height))
    rotation_matrix = cv2.getRotationMatrix2D((camera.cx, camera.cy), 2.0, 1.0)
    rotated = cv2.warpAffine(base, rotation_matrix, (camera.width, camera.height))
    insufficient_a = np.zeros_like(base)
    insufficient_b = np.full_like(base, 1)
    cases = {
        "identical_frame": vo.estimate(base, base.copy(), camera),
        "small_translation": vo.estimate(base, translated, camera),
        "rotation": vo.estimate(base, rotated, camera),
        "insufficient_features": vo.estimate(insufficient_a, insufficient_b, camera),
        "resolution_change": vo.estimate(base, base[:-1, :], camera),
    }
    continuity = _synthetic_continuity(settings, base)
    return {
        "implementation_diagnostic_mode": True,
        "production_config_enabled": settings.localization_enabled,
        "cases": {
            name: motion_diagnostics(result, camera)
            for name, result in cases.items()
        },
        "continuity": continuity,
        "visualization_frames": (base, translated, rotated),
    }


def _synthetic_continuity(settings, base):
    fingerprint = b"a" * 16
    state = VisualOdometryState(
        previous_gray=base,
        previous_frame_id="f0",
        previous_frame_index=0,
        video_name="video-a",
        image_shape=base.shape[:2],
        modality=ImageModality.RGB,
        previous_fingerprint=fingerprint,
    )
    def frame(frame_id, index, video="video-a"):
        return FrameContext(frame_id, frame_id, video, "session-a", None, None, None, None, index, ImageModality.RGB)
    gap = evaluate_continuity(state, frame("f2", 2), shape=base.shape[:2], modality=ImageModality.RGB, fingerprint=b"b" * 16, max_frame_gap=settings.localization_max_frame_gap)
    video = evaluate_continuity(state, frame("f1", 1, "video-b"), shape=base.shape[:2], modality=ImageModality.RGB, fingerprint=b"b" * 16, max_frame_gap=settings.localization_max_frame_gap)
    repeated = evaluate_continuity(state, frame("f1", 1), shape=base.shape[:2], modality=ImageModality.RGB, fingerprint=fingerprint, max_frame_gap=settings.localization_max_frame_gap)
    resolution = evaluate_continuity(state, frame("f1", 1), shape=(base.shape[0] - 1, base.shape[1]), modality=ImageModality.RGB, fingerprint=b"b" * 16, max_frame_gap=settings.localization_max_frame_gap)
    new_session = evaluate_continuity(VisualOdometryState(), frame("f0", 0), shape=base.shape[:2], modality=ImageModality.RGB, fingerprint=fingerprint, max_frame_gap=settings.localization_max_frame_gap)
    return {
        "frame_gap": {"action": gap.action.value, "reason": gap.event},
        "video_change": {"action": video.action.value, "reason": video.event},
        "identical_image": {"action": repeated.action.value, "reason": repeated.event},
        "resolution_change": {"action": resolution.action.value, "reason": resolution.event},
        "session_reset": {"action": new_session.action.value, "reason": new_session.event},
    }


def validate_official_compatibility(results: Sequence[DetectedTranslation | None]) -> dict[str, object]:
    issues: list[str] = []
    serialized = []
    for index, result in enumerate(results):
        if result is None:
            serialized.append(None)
            continue
        payload = result.model_dump(mode="json")
        expected = {"translation_x", "translation_y", "translation_z"}
        if set(payload) != expected:
            issues.append(f"result {index}: field mismatch")
        if not all(isinstance(payload[key], (int, float)) and math.isfinite(float(payload[key])) for key in expected):
            issues.append(f"result {index}: non-finite numeric field")
        serialized.append(payload)
        response = CompetitionResponse.from_task_results(
            response_id=index + 1,
            user="offline-validator",
            frame=f"frame-{index}",
            detected_objects=[],
            detected_translation=result,
            matched_reference_objects=[],
        )
        if len(response.detected_translations) != 1:
            issues.append(f"result {index}: CompetitionResponse mapping failed")
    return {
        "compatible": not issues,
        "issues": issues,
        "serialized_results": serialized,
        "none_representation": "safe empty detected_translations list",
        "result_mapper_expected_fields": ["translation_x", "translation_y", "translation_z"],
    }


async def run_validation(
    settings: Settings,
    options: ValidationOptions,
    *,
    emit: Callable[[str], None] = print,
) -> dict[str, object]:
    settings.validate_localization_vo()
    output_dir = options.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = task2_config_report(settings)
    camera = camera_intrinsics_report(settings)
    _emit_config(config, camera, emit)

    frames: list[dict[str, object]] = []
    results: list[DetectedTranslation | None] = []
    synthetic_vo: dict[str, object] | None = None
    loaded_images: list[LoadedImage] = []
    if options.synthetic:
        synthetic_vo = await run_synthetic_vo(settings)
        frames = _synthetic_frame_rows(synthetic_vo, settings)
        results = [None] * len(frames)
        if options.save_visualizations:
            _save_synthetic_visualizations(output_dir, synthetic_vo)
    else:
        paths = discover_input_images(options)
        loaded_images = [load_local_image(path) for path in paths]
        frames, results, _ = await process_loaded_sequence(settings, loaded_images, respect_enabled=True)
        if options.save_visualizations:
            _save_sequence_visualizations(output_dir, loaded_images, frames)

    gps = synthetic_gps_calibration(settings) if options.synthetic_gps else {
        "active": settings.localization_calibration_enabled,
        "input_available": False,
        "scale_ready": False,
        "estimated_scale": None,
        "calibration_reason": "gps_input_unavailable_in_offline_frames",
    }
    compatibility = validate_official_compatibility(results)
    report = {
        "config": config,
        "camera_intrinsics": camera,
        "frames": frames,
        "vo_diagnostics": [frame["vo_diagnostics"] for frame in frames],
        "calibration_state": [
            {
                "frame_index": frame["frame_index"],
                "state": frame["calibration_state"],
                "sample_count": frame["calibration_frame_count"],
                "expected_max_frame": frame["calibration_expected_max_frame"],
                "result": frame.get("calibration_result"),
            }
            for frame in frames
        ],
        "gps_scale_diagnostics": gps,
        "localization_results": [result.model_dump(mode="json") if result else None for result in results],
        "official_result_compatibility": compatibility,
        "synthetic_vo": _without_visualization_arrays(synthetic_vo),
        "final_summary": {
            "frame_count": len(frames),
            "vo_valid_count": sum(bool(frame["vo_diagnostics"]["transform_valid"]) for frame in frames),
            "safe_empty_count": sum(frame["localization_result"] is None for frame in frames),
            "localization_enabled": settings.localization_enabled,
            "camera_intrinsics_valid": camera["valid"],
            "gps_scale_ready": bool(gps.get("scale_ready")),
            "official_result_compatible": compatibility["compatible"],
            "prediction_submission": "DISABLED",
        },
    }
    json_path = output_dir / "task2_localization_validation.json"
    csv_path = output_dir / "task2_localization_sequence.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_sequence_csv(csv_path, frames)
    _emit_summary(report, json_path, csv_path, emit)
    emit("Prediction submission: DISABLED")
    return report


def discover_input_images(options: ValidationOptions) -> list[Path]:
    if options.images_dir is not None:
        if options.image is not None or options.next_image is not None or options.additional_images:
            raise ValueError("--images-dir cannot be combined with individual images")
        root = options.images_dir.expanduser().resolve()
        if not root.is_dir():
            raise ValueError("images directory does not exist")
        paths = sorted(
            (path for path in root.iterdir() if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp"}),
            key=_natural_key,
        )
    else:
        paths = [path for path in (options.image, options.next_image, *options.additional_images) if path is not None]
    if not paths:
        raise ValueError("at least one image or --synthetic is required")
    resolved = [Path(path).expanduser().resolve() for path in paths]
    if not all(path.is_file() for path in resolved):
        raise ValueError("one or more image paths do not exist")
    return resolved


def _synthetic_feature_frame(width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width), np.uint8)
    step = max(12, min(width, height) // 12)
    radius = max(2, step // 8)
    for y in range(step, height - step, step):
        for x in range(step, width - step, step):
            cv2.circle(image, (x, y), radius, 255, -1)
            cv2.line(image, (x - radius, y), (x + radius, y), 128, 1)
    return image


def _synthetic_frame_rows(synthetic, settings):
    rows = []
    mapping = (
        (0, "initialization", None),
        (1, "identical_frame", synthetic["cases"]["identical_frame"]),
        (2, "small_translation", synthetic["cases"]["small_translation"]),
        (3, "rotation", synthetic["cases"]["rotation"]),
        (4, "insufficient_features", synthetic["cases"]["insufficient_features"]),
        (5, "resolution_change", synthetic["cases"]["resolution_change"]),
    )
    cumulative_x = cumulative_y = cumulative_yaw = 0.0
    for index, name, diagnostics in mapping:
        diagnostic = diagnostics or motion_diagnostics(None)
        if diagnostic["transform_valid"]:
            cumulative_x += float(diagnostic["translation_x_px"])
            cumulative_y += float(diagnostic["translation_y_px"])
            cumulative_yaw += float(diagnostic["rotation_yaw_rad"])
        status = "initialization" if index == 0 else "vo_valid" if diagnostic["transform_valid"] else "safe_empty"
        rows.append(
            {
                "frame_index": index,
                "frame_name": name,
                "frame_path": None,
                "width": None,
                "height": None,
                "timestamp": None,
                "localization_service_enabled": False,
                "implementation_diagnostic_mode": True,
                "initialization_state": "initialized" if index == 0 else "tracking",
                "service_frame_count": index + 1,
                "vo_frame_count": index + 1,
                "warmup_count": max(0, index),
                "calibration_state": "inactive",
                "calibration_frame_count": 0,
                "calibration_expected_max_frame": settings.localization_calibration_expected_max_frame,
                "previous_frame_available": index > 0,
                "current_frame_accepted": True,
                "camera_resolution_compatible": True,
                "calibration_result": None,
                "vo_diagnostics": diagnostic,
                "pose": {
                    "relative_dx_px": diagnostic["translation_x_px"],
                    "relative_dy_px": diagnostic["translation_y_px"],
                    "relative_dz": None,
                    "relative_yaw_rad": diagnostic["rotation_yaw_rad"],
                    "relative_pitch": None,
                    "relative_roll": None,
                    "cumulative_dx_px": cumulative_x,
                    "cumulative_dy_px": cumulative_y,
                    "cumulative_yaw_rad": cumulative_yaw,
                    "cumulative_x": None,
                    "cumulative_y": None,
                    "cumulative_z": None,
                    "status": status,
                },
                "localization_result": None,
                "status": status,
                "failure_reason": diagnostic["failure_reason"],
            }
        )
    return rows


def _without_visualization_arrays(synthetic):
    if synthetic is None:
        return None
    return {key: value for key, value in synthetic.items() if key != "visualization_frames"}


def _save_synthetic_visualizations(output_dir, synthetic):
    trajectory = []
    for index, image in enumerate(synthetic["visualization_frames"]):
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        name = ("initialization", "translation", "rotation")[index]
        cv2.putText(canvas, name, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if index:
            diagnostic = synthetic["cases"]["small_translation" if index == 1 else "rotation"]
            dx = diagnostic.get("translation_x_px") or 0.0
            dy = diagnostic.get("translation_y_px") or 0.0
            center = (canvas.shape[1] // 2, canvas.shape[0] // 2)
            cv2.arrowedLine(canvas, center, (round(center[0] + dx * 10), round(center[1] + dy * 10)), (0, 0, 255), 2)
            trajectory.append((dx, dy))
        cv2.imwrite(str(output_dir / f"frame_{index:03d}_vo.jpg"), canvas)
    _save_trajectory(output_dir / "trajectory.png", trajectory)


def _save_sequence_visualizations(output_dir, loaded_images, frames):
    trajectory = []
    for loaded, frame in zip(loaded_images, frames, strict=True):
        canvas = loaded.image.copy()
        diagnostic = frame["vo_diagnostics"]
        center = (canvas.shape[1] // 2, canvas.shape[0] // 2)
        dx = diagnostic.get("translation_x_px") or 0.0
        dy = diagnostic.get("translation_y_px") or 0.0
        cv2.arrowedLine(canvas, center, (round(center[0] + dx * 10), round(center[1] + dy * 10)), (0, 0, 255), 3)
        text = f"{frame['status']} dx={dx:.2f} dy={dy:.2f} inliers={diagnostic.get('inlier_count', 0)}"
        cv2.putText(canvas, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "feature coordinates unavailable from production result", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / f"frame_{frame['frame_index']:03d}_vo.jpg"), canvas)
        trajectory.append((frame["pose"]["cumulative_dx_px"], frame["pose"]["cumulative_dy_px"]))
    _save_trajectory(output_dir / "trajectory.png", trajectory)


def _save_trajectory(path, trajectory):
    canvas = np.full((600, 800, 3), 255, np.uint8)
    origin = np.asarray([400.0, 300.0])
    points = [origin]
    for x, y in trajectory:
        points.append(origin + np.asarray([float(x), float(y)]) * 5.0)
    for first, second in zip(points, points[1:]):
        cv2.line(canvas, tuple(np.round(first).astype(int)), tuple(np.round(second).astype(int)), (255, 0, 0), 2)
    cv2.putText(canvas, "Offline cumulative VO trajectory (pixels)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(str(path), canvas)


def _write_sequence_csv(path, frames):
    rows = []
    for frame in frames:
        diagnostic, pose = frame["vo_diagnostics"], frame["pose"]
        calibration = frame.get("calibration_result") or {}
        rows.append(
            {
                "frame_index": frame["frame_index"],
                "frame_name": frame["frame_name"],
                "vo_valid": diagnostic["transform_valid"],
                "dx": diagnostic["translation_x_px"],
                "dy": diagnostic["translation_y_px"],
                "dz": pose["relative_dz"],
                "cumulative_x": pose["cumulative_x"],
                "cumulative_y": pose["cumulative_y"],
                "cumulative_z": pose["cumulative_z"],
                "cumulative_dx_px": pose["cumulative_dx_px"],
                "cumulative_dy_px": pose["cumulative_dy_px"],
                "rotation": diagnostic["rotation_yaw_rad"],
                "inliers": diagnostic["inlier_count"],
                "inlier_ratio": diagnostic["inlier_ratio"],
                "gps_scale_ready": bool(calibration.get("ready", False)),
                "gps_scale": calibration.get("scale"),
                "status": frame["status"],
                "failure_reason": frame["failure_reason"],
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _serialize_calibration(calibration):
    if calibration is None:
        return None
    return {
        "ready": calibration.ready,
        "sample_count": calibration.sample_count,
        "inlier_count": calibration.inlier_count,
        "scale": calibration.scale,
        "rms_residual": calibration.rms_residual,
        "directional_diversity": calibration.directional_diversity,
        "failure_reason": calibration.failure_reason,
    }


def _emit_config(config, camera, emit):
    emit("Task 2 production config:")
    for key, value in config.items():
        emit(f"  {key}={value}")
    emit(f"Camera intrinsics validation: {'OK' if camera['valid'] else 'FAIL'}")


def _emit_summary(report, json_path, csv_path, emit):
    summary = report["final_summary"]
    emit(f"Frames: {summary['frame_count']}; VO valid: {summary['vo_valid_count']}; safe empty: {summary['safe_empty_count']}")
    if not summary["localization_enabled"]:
        emit("Localization: implemented; disabled by production configuration")
    gps = report["gps_scale_diagnostics"]
    emit(f"GPS scale ready: {gps.get('scale_ready')} scale={gps.get('estimated_scale')} reason={gps.get('calibration_reason')}")
    emit(f"Official result compatibility: {'OK' if summary['official_result_compatible'] else 'FAIL'}")
    emit(f"JSON: {json_path}")
    emit(f"CSV: {csv_path}")


def _natural_key(path):
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name))


def _divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _finite_report_values(value):
    if isinstance(value, dict):
        return all(_finite_report_values(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_report_values(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = ValidationOptions(
        output_dir=args.output_dir,
        image=args.image,
        next_image=args.next_image,
        additional_images=tuple(args.additional_image),
        images_dir=args.images_dir,
        synthetic=args.synthetic,
        synthetic_gps=args.synthetic_gps,
        save_visualizations=args.save_visualizations,
    )
    try:
        asyncio.run(run_validation(get_settings(), options))
    except Exception as exc:
        print(f"Task 2 localization validation: FAIL ({type(exc).__name__}: {exc})")
        print("Prediction submission: DISABLED")
        return EXIT_INPUT
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
