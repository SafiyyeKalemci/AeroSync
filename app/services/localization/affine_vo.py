from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from app.services.localization.camera_model import CameraModel
from app.services.localization.state import AffineMotionResult

FeatureDetector = Callable[[np.ndarray], np.ndarray | None]
Tracker = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]]


@dataclass(frozen=True)
class AffineVOConfig:
    min_features: int
    max_features: int
    feature_quality_level: float
    feature_min_distance: float
    lk_win_size: int
    lk_max_level: int
    lk_fb_error_threshold: float
    ransac_iterations: int
    ransac_residual_threshold: float
    min_inliers: int
    min_inlier_ratio: float
    freeze_threshold: float


class AffineVisualOdometry:
    """Shi-Tomasi + forward/backward LK + robust 2D tx/ty/yaw estimation."""

    def __init__(
        self,
        config: AffineVOConfig,
        *,
        feature_detector: FeatureDetector | None = None,
        tracker: Tracker | None = None,
        random_seed: int = 0,
    ) -> None:
        self.config = config
        self._feature_detector = feature_detector or self._detect_features
        self._tracker = tracker or self._track
        self._random_seed = random_seed

    def _detect_features(self, gray: np.ndarray) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_features,
            qualityLevel=self.config.feature_quality_level,
            minDistance=self.config.feature_min_distance,
        )

    def _track(
        self, source: np.ndarray, target: np.ndarray, points: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        return cv2.calcOpticalFlowPyrLK(
            source,
            target,
            points,
            None,
            winSize=(self.config.lk_win_size, self.config.lk_win_size),
            maxLevel=self.config.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    def estimate(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        camera: CameraModel,
    ) -> AffineMotionResult:
        if previous_gray.shape != current_gray.shape:
            return AffineMotionResult(failure_reason="shape_changed")
        difference = float(np.mean(cv2.absdiff(previous_gray, current_gray)))
        if difference <= self.config.freeze_threshold:
            return AffineMotionResult(failure_reason="freeze")

        points0 = self._feature_detector(previous_gray)
        if points0 is None or len(points0) < self.config.min_features:
            return AffineMotionResult(
                tracked_points=0 if points0 is None else len(points0),
                failure_reason="insufficient_features",
            )
        points1, status_forward, _ = self._tracker(previous_gray, current_gray, points0)
        if points1 is None or status_forward is None:
            return AffineMotionResult(failure_reason="tracking_failed")
        points0_back, status_backward, _ = self._tracker(current_gray, previous_gray, points1)
        if points0_back is None or status_backward is None:
            return AffineMotionResult(failure_reason="tracking_failed")

        previous = points0.reshape(-1, 2).astype(np.float64)
        current = points1.reshape(-1, 2).astype(np.float64)
        backward = points0_back.reshape(-1, 2).astype(np.float64)
        status = status_forward.reshape(-1).astype(bool) & status_backward.reshape(-1).astype(bool)
        finite = np.isfinite(previous).all(1) & np.isfinite(current).all(1) & np.isfinite(backward).all(1)
        height, width = current_gray.shape[:2]
        in_bounds = (
            (previous[:, 0] >= 0) & (previous[:, 0] < width)
            & (previous[:, 1] >= 0) & (previous[:, 1] < height)
            & (current[:, 0] >= 0) & (current[:, 0] < width)
            & (current[:, 1] >= 0) & (current[:, 1] < height)
            & (backward[:, 0] >= 0) & (backward[:, 0] < width)
            & (backward[:, 1] >= 0) & (backward[:, 1] < height)
        )
        fb_error = np.linalg.norm(backward - previous, axis=1)
        valid = status & finite & in_bounds & (fb_error <= self.config.lk_fb_error_threshold)
        previous = previous[valid]
        current = current[valid]
        tracked = len(previous)
        if tracked < self.config.min_features:
            return AffineMotionResult(tracked_points=tracked, failure_reason="insufficient_features")
        return self._robust_fit(previous, current, camera)

    @staticmethod
    def _system(previous: np.ndarray, current: np.ndarray, camera: CameraModel) -> tuple[np.ndarray, np.ndarray]:
        count = len(previous)
        a = np.zeros((2 * count, 3), dtype=np.float64)
        b = np.empty(2 * count, dtype=np.float64)
        displacement = current - previous
        a[0::2, 0] = -1.0
        a[0::2, 2] = previous[:, 1] - camera.cy
        a[1::2, 1] = -1.0
        a[1::2, 2] = -(previous[:, 0] - camera.cx)
        b[0::2] = displacement[:, 0]
        b[1::2] = displacement[:, 1]
        return a, b

    @staticmethod
    def _point_residuals(a: np.ndarray, b: np.ndarray, params: np.ndarray) -> np.ndarray:
        component = (a @ params - b).reshape(-1, 2)
        return np.linalg.norm(component, axis=1)

    def _robust_fit(
        self, previous: np.ndarray, current: np.ndarray, camera: CameraModel
    ) -> AffineMotionResult:
        tracked = len(previous)
        a, b = self._system(previous, current, camera)
        rng = np.random.default_rng(self._random_seed)
        best_mask: np.ndarray | None = None
        best_count = 0
        for _ in range(self.config.ransac_iterations):
            indices = rng.choice(tracked, 3, replace=False)
            rows = np.sort(np.concatenate((2 * indices, 2 * indices + 1)))
            try:
                params = np.linalg.lstsq(a[rows], b[rows], rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(params).all():
                continue
            mask = self._point_residuals(a, b, params) <= self.config.ransac_residual_threshold
            count = int(mask.sum())
            if count > best_count:
                best_count, best_mask = count, mask
        if best_mask is None or best_count < 3:
            return AffineMotionResult(tracked_points=tracked, failure_reason="ransac_failed")

        rows = np.repeat(best_mask, 2)
        try:
            refined = np.linalg.lstsq(a[rows], b[rows], rcond=None)[0]
        except np.linalg.LinAlgError:
            return AffineMotionResult(tracked_points=tracked, failure_reason="ransac_failed")
        residuals = self._point_residuals(a, b, refined)
        final_mask = residuals <= self.config.ransac_residual_threshold
        inlier_count = int(final_mask.sum())
        ratio = inlier_count / tracked
        rms = float(math.sqrt(float(np.mean(np.square(residuals[final_mask]))))) if inlier_count else math.inf
        quality = (
            np.isfinite(refined).all()
            and inlier_count >= self.config.min_inliers
            and ratio >= self.config.min_inlier_ratio
            and math.isfinite(rms)
            and rms <= self.config.ransac_residual_threshold
        )
        reason = None if quality else "low_quality"
        return AffineMotionResult(
            delta_x_px=float(refined[0]) if quality else None,
            delta_y_px=float(refined[1]) if quality else None,
            delta_yaw_rad=float(refined[2]) if quality else None,
            tracked_points=tracked,
            inlier_count=inlier_count,
            inlier_ratio=ratio,
            rms_residual=rms if math.isfinite(rms) else None,
            quality_valid=quality,
            failure_reason=reason,
        )

