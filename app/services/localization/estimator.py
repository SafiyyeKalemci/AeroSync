from __future__ import annotations

import math

import numpy as np

from app.schemas import DetectedTranslation
from app.services.localization.calibration import CalibrationResult


def estimate_translation(
    *,
    calibration: CalibrationResult,
    gps_anchor: tuple[float, float, float] | None,
    camera_anchor: tuple[float, float] | None,
    camera_position: tuple[float, float],
    last_estimate: tuple[float, float, float] | None,
    max_delta_per_frame: float,
    z_policy: str,
) -> DetectedTranslation | None:
    if (
        not calibration.ready
        or calibration.rotation_matrix_2x2 is None
        or calibration.scale is None
        or gps_anchor is None
        or camera_anchor is None
    ):
        return None
    values = (*gps_anchor, *camera_anchor, *camera_position, calibration.scale)
    if not all(math.isfinite(value) for value in values):
        return None
    if z_policy == "return_none_if_schema_allows":
        return None
    delta_camera = np.asarray(camera_position) - np.asarray(camera_anchor)
    rotation = np.asarray(calibration.rotation_matrix_2x2)
    delta_xy = calibration.scale * (rotation @ delta_camera)
    predicted_xy = np.asarray(gps_anchor[:2]) + delta_xy
    if last_estimate is not None and max_delta_per_frame > 0:
        previous_xy = np.asarray(last_estimate[:2])
        step = predicted_xy - previous_xy
        norm = float(np.linalg.norm(step))
        if norm > max_delta_per_frame:
            predicted_xy = previous_xy + step * (max_delta_per_frame / norm)
    z = gps_anchor[2]
    output = (float(predicted_xy[0]), float(predicted_xy[1]), float(z))
    if not all(math.isfinite(value) for value in output):
        return None
    return DetectedTranslation(
        translation_x=output[0], translation_y=output[1], translation_z=output[2]
    )

