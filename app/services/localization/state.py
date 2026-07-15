from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from app.schemas import ImageModality

if TYPE_CHECKING:
    from app.services.localization.calibration import CalibrationResult, CalibrationSample


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class AffineMotionResult:
    delta_x_px: float | None = None
    delta_y_px: float | None = None
    delta_yaw_rad: float | None = None
    tracked_points: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    rms_residual: float | None = None
    quality_valid: bool = False
    failure_reason: str | None = None


@dataclass
class VisualOdometryState:
    previous_gray: np.ndarray | None = None
    previous_frame_id: str | None = None
    previous_frame_index: int | None = None
    video_name: str | None = None
    image_shape: tuple[int, int] | None = None
    modality: ImageModality | None = None
    frame_count: int = 0
    warmup_count: int = 0
    cumulative_yaw: float = 0.0
    cumulative_dx_px: float = 0.0
    cumulative_dy_px: float = 0.0
    last_motion_result: AffineMotionResult | None = None
    duplicate_cache: AffineMotionResult | None = None
    previous_fingerprint: bytes | None = None
    freeze_detected: bool = False
    last_access_time: datetime = field(default_factory=utc_now)
    previous_valid_gps: tuple[float, float, float] | None = None
    last_valid_gps: tuple[float, float, float] | None = None
    previous_gps_frame_index: int | None = None
    camera_step_samples: list[tuple[float, float]] = field(default_factory=list)
    gps_step_samples: list[tuple[float, float]] = field(default_factory=list)
    calibration_samples: list["CalibrationSample"] = field(default_factory=list)
    calibration_result: "CalibrationResult | None" = None
    frozen_calibration_result: "CalibrationResult | None" = None
    calibration_frozen: bool = False
    gps_anchor: tuple[float, float, float] | None = None
    camera_anchor: tuple[float, float] | None = None
    last_estimate: tuple[float, float, float] | None = None
    gps_health_transition: str | None = None
    healthy_sample_count: int = 0
    unhealthy_frame_count: int = 0
    recovery_healthy_count: int = 0
    expected_window_warned: bool = False
