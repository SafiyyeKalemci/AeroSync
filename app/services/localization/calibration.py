from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from app.services.localization.state import AffineMotionResult

Vector2 = tuple[float, float]


@dataclass(frozen=True)
class CalibrationSample:
    frame_index: int | None
    sequence: int
    camera_delta_px_2d: Vector2
    gps_delta_xy: Vector2
    camera_motion_quality: float
    gps_movement_magnitude: float
    yaw_rad: float
    timestamp: datetime


@dataclass(frozen=True)
class CalibrationResult:
    ready: bool = False
    sample_count: int = 0
    inlier_count: int = 0
    rotation_matrix_2x2: tuple[tuple[float, float], tuple[float, float]] | None = None
    scale: float | None = None
    rms_residual: float | None = None
    scale_median: float | None = None
    scale_mad: float | None = None
    motion_span: float = 0.0
    directional_diversity: float = 0.0
    failure_reason: str | None = "insufficient_samples"


@dataclass(frozen=True)
class CalibrationPolicy:
    min_samples: int
    max_samples: int
    min_camera_step_px: float
    min_gps_step: float
    max_rms_residual: float
    min_inlier_ratio: float
    min_directional_diversity: float
    scale_min: float
    scale_max: float
    outlier_mad_factor: float
    allow_reflection: bool


def make_calibration_sample(
    *,
    frame_index: int | None,
    sequence: int,
    camera_delta: Vector2,
    gps_delta: Vector2,
    motion: AffineMotionResult,
    yaw_rad: float,
    policy: CalibrationPolicy,
) -> tuple[CalibrationSample | None, str | None]:
    values = (*camera_delta, *gps_delta, yaw_rad, motion.inlier_ratio)
    if not all(math.isfinite(value) for value in values):
        return None, "non_finite_sample"
    if not motion.quality_valid:
        return None, "motion_quality_invalid"
    camera_norm = math.hypot(*camera_delta)
    gps_norm = math.hypot(*gps_delta)
    if camera_norm < policy.min_camera_step_px:
        return None, "camera_step_too_small"
    if gps_norm < policy.min_gps_step:
        return None, "gps_step_too_small"
    return CalibrationSample(
        frame_index=frame_index,
        sequence=sequence,
        camera_delta_px_2d=camera_delta,
        gps_delta_xy=gps_delta,
        camera_motion_quality=motion.inlier_ratio,
        gps_movement_magnitude=gps_norm,
        yaw_rad=yaw_rad,
        timestamp=datetime.now(timezone.utc),
    ), None

