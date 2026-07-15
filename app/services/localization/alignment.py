from __future__ import annotations

import math

import numpy as np

from app.services.localization.calibration import CalibrationPolicy, CalibrationResult, CalibrationSample


def fit_calibration(
    samples: list[CalibrationSample], policy: CalibrationPolicy
) -> CalibrationResult:
    count = len(samples)
    if count < policy.min_samples:
        return CalibrationResult(sample_count=count, failure_reason="insufficient_samples")
    camera = np.asarray([item.camera_delta_px_2d for item in samples], dtype=np.float64)
    gps = np.asarray([item.gps_delta_xy for item in samples], dtype=np.float64)
    if not np.isfinite(camera).all() or not np.isfinite(gps).all():
        return CalibrationResult(sample_count=count, failure_reason="non_finite_samples")

    camera_norm = np.linalg.norm(camera, axis=1)
    gps_norm = np.linalg.norm(gps, axis=1)
    ratios = gps_norm / camera_norm
    median = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - median)))
    robust_sigma = 1.4826 * mad
    tolerance = max(policy.outlier_mad_factor * robust_sigma, abs(median) * 0.02, 1e-12)
    scale_mask = np.abs(ratios - median) <= tolerance
    inlier_count = int(scale_mask.sum())
    inlier_ratio = inlier_count / count
    if inlier_count < policy.min_samples or inlier_ratio < policy.min_inlier_ratio:
        return CalibrationResult(
            sample_count=count, inlier_count=inlier_count,
            scale_median=median, scale_mad=mad,
            failure_reason="low_inlier_ratio",
        )

    camera_inliers = camera[scale_mask]
    gps_inliers = gps[scale_mask]
    covariance = camera_inliers.T @ gps_inliers
    try:
        u, singular, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return CalibrationResult(sample_count=count, inlier_count=inlier_count, failure_reason="alignment_failed")
    unconstrained = vt.T @ u.T
    determinant = float(np.linalg.det(unconstrained))
    if determinant < 0 and not policy.allow_reflection:
        return CalibrationResult(
            sample_count=count, inlier_count=inlier_count,
            scale_median=median, scale_mad=mad,
            failure_reason="reflection_detected",
        )
    rotation = unconstrained

    filtered_ratios = ratios[scale_mask]
    scale = float(np.median(filtered_ratios))
    motion_span = float(np.sum(np.linalg.norm(camera_inliers, axis=1)))
    direction_matrix = camera_inliers / np.linalg.norm(camera_inliers, axis=1, keepdims=True)
    direction_singular = np.linalg.svd(direction_matrix, compute_uv=False)
    directional_diversity = float(
        direction_singular[-1] / direction_singular[0]
    ) if direction_singular[0] > 1e-12 else 0.0
    predicted = scale * (rotation @ camera_inliers.T).T
    point_residuals = np.linalg.norm(predicted - gps_inliers, axis=1)
    rms = float(math.sqrt(float(np.mean(np.square(point_residuals)))))

    common = dict(
        sample_count=count,
        inlier_count=inlier_count,
        rotation_matrix_2x2=(
            (float(rotation[0, 0]), float(rotation[0, 1])),
            (float(rotation[1, 0]), float(rotation[1, 1])),
        ),
        scale=scale,
        rms_residual=rms,
        scale_median=median,
        scale_mad=mad,
        motion_span=motion_span,
        directional_diversity=directional_diversity,
    )
    if not math.isfinite(scale) or not policy.scale_min <= scale <= policy.scale_max:
        return CalibrationResult(**common, failure_reason="scale_out_of_range")
    if directional_diversity < policy.min_directional_diversity:
        return CalibrationResult(**common, failure_reason="insufficient_directional_diversity")
    if not math.isfinite(rms) or rms > policy.max_rms_residual:
        return CalibrationResult(**common, failure_reason="high_residual")
    return CalibrationResult(**common, ready=True, failure_reason=None)

