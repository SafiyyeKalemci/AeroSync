from __future__ import annotations

import logging
import math

from app.core.config import Settings
from app.services.matching.descriptor_types import (
    CoarseMatchSet,
    HomographyResult,
    MatchingQuality,
    ProjectedPolygon,
)

logger = logging.getLogger(__name__)


class HomographyEstimator:
    def __init__(
        self,
        settings: Settings,
        *,
        min_inliers: int | None = None,
        min_inlier_ratio: float | None = None,
        max_rms_reprojection_error: float | None = None,
    ) -> None:
        if settings.matching_homography_method != "USAC_MAGSAC":
            raise ValueError("Yalniz MATCHING_HOMOGRAPHY_METHOD=USAC_MAGSAC desteklenir.")
        self._threshold = settings.matching_homography_reprojection_threshold
        self._confidence = settings.matching_homography_confidence
        self._iterations = settings.matching_homography_max_iterations
        self._min_inliers = (
            settings.matching_homography_min_inliers if min_inliers is None else min_inliers
        )
        self._min_ratio = (
            settings.matching_homography_min_inlier_ratio
            if min_inlier_ratio is None else min_inlier_ratio
        )
        self._max_rms = (
            settings.matching_homography_max_rms_reprojection_error
            if max_rms_reprojection_error is None else max_rms_reprojection_error
        )
        if self._threshold <= 0 or self._iterations < 1 or self._min_inliers < 4:
            raise ValueError("Homography config gecersiz.")
        if not 0 < self._confidence < 1 or not 0 <= self._min_ratio <= 1 or self._max_rms <= 0:
            raise ValueError("Homography kalite config gecersiz.")

    def estimate(self, matches: CoarseMatchSet) -> HomographyResult:
        import cv2
        import numpy as np

        if matches.failure_reason or matches.correspondence_count < 4:
            return self._invalid("insufficient_correspondences")
        source = np.asarray(matches.reference_points_px, dtype=np.float32)
        target = np.asarray(matches.frame_points_px, dtype=np.float32)
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
            return self._invalid("point_shape_invalid")
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            return self._invalid("points_non_finite")
        try:
            matrix, mask = cv2.findHomography(
                source,
                target,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=self._threshold,
                maxIters=self._iterations,
                confidence=self._confidence,
            )
        except cv2.error:
            logger.exception("matching_homography_failed", extra={"event": "matching_homography_failed"})
            return self._invalid("opencv_exception")
        if matrix is None or mask is None:
            logger.info("matching_homography_failed", extra={"event": "matching_homography_failed"})
            return self._invalid("homography_not_found")
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            return self._invalid("homography_matrix_invalid", matrix=matrix)
        normalizer = matrix[2, 2]
        if not math.isfinite(float(normalizer)) or abs(float(normalizer)) < 1e-12:
            return self._invalid("homography_not_normalizable", matrix=matrix)
        matrix = matrix / normalizer
        determinant = float(np.linalg.det(matrix))
        condition = float(np.linalg.cond(matrix))
        if not math.isfinite(determinant) or abs(determinant) < 1e-10:
            return self._invalid("homography_singular", matrix=matrix)
        if not math.isfinite(condition) or condition > 1e12:
            return self._invalid("homography_ill_conditioned", matrix=matrix)
        inlier_mask = np.asarray(mask).reshape(-1).astype(bool)
        if len(inlier_mask) != len(source):
            return self._invalid("inlier_mask_invalid", matrix=matrix)
        inlier_count = int(inlier_mask.sum())
        inlier_ratio = inlier_count / len(source)
        if inlier_count < self._min_inliers or inlier_ratio < self._min_ratio:
            logger.info(
                "matching_homography_low_inliers",
                extra={"event": "matching_homography_low_inliers",
                       "inlier_count": inlier_count, "inlier_ratio": inlier_ratio},
            )
            return HomographyResult(
                matrix, inlier_mask, inlier_count, inlier_ratio, float("inf"), False,
                "low_inliers",
            )
        projected = cv2.perspectiveTransform(source[inlier_mask].reshape(-1, 1, 2), matrix).reshape(-1, 2)
        errors = np.linalg.norm(projected - target[inlier_mask], axis=1)
        rms = float(np.sqrt(np.mean(np.square(errors))))
        if not math.isfinite(rms) or rms > self._max_rms:
            logger.info(
                "matching_homography_high_reprojection_error",
                extra={"event": "matching_homography_high_reprojection_error",
                       "rms_reprojection_error": rms},
            )
            return HomographyResult(
                matrix, inlier_mask, inlier_count, inlier_ratio, rms, False,
                "high_reprojection_error",
            )
        return HomographyResult(matrix, inlier_mask, inlier_count, inlier_ratio, rms, True)

    @staticmethod
    def _invalid(reason: str, *, matrix=None) -> HomographyResult:
        logger.info(
            "matching_homography_failed",
            extra={"event": "matching_homography_failed", "reason": reason},
        )
        return HomographyResult(matrix, None, 0, 0.0, float("inf"), False, reason)


class ProjectedPolygonValidator:
    def __init__(self, settings: Settings) -> None:
        self._min_area = settings.matching_geometry_min_projected_area_px
        self._max_frame_ratio = settings.matching_geometry_max_frame_area_ratio
        self._min_visible = settings.matching_geometry_min_visible_ratio
        self._min_edge = settings.matching_geometry_min_edge_length_px
        self._max_aspect = settings.matching_geometry_max_aspect_ratio
        self._max_distortion = settings.matching_geometry_max_perspective_distortion
        if (
            self._min_area <= 0 or not 0 < self._max_frame_ratio <= 1
            or not 0 <= self._min_visible <= 1 or self._min_edge <= 0
            or self._max_aspect < 1 or self._max_distortion < 1
        ):
            raise ValueError("Projected polygon config gecersiz.")

    def project_and_validate(
        self,
        matrix,
        *,
        reference_width: int,
        reference_height: int,
        frame_width: int,
        frame_height: int,
    ) -> ProjectedPolygon:
        import cv2
        import numpy as np

        if reference_width <= 0 or reference_height <= 0:
            return self._invalid("reference_size_invalid")
        corners = np.asarray(
            [[0, 0], [reference_width, 0], [reference_width, reference_height], [0, reference_height]],
            dtype=np.float32,
        )
        try:
            points = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        except cv2.error:
            return self._invalid("perspective_transform_failed")
        return self.validate_points(points, frame_width=frame_width, frame_height=frame_height)

    def validate_points(self, points, *, frame_width: int, frame_height: int) -> ProjectedPolygon:
        import cv2
        import numpy as np

        polygon = np.asarray(points, dtype=np.float32)
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            return self._invalid("polygon_points_invalid", polygon)
        if frame_width <= 0 or frame_height <= 0:
            return self._invalid("frame_size_invalid", polygon)
        contour = polygon.reshape(-1, 1, 2)
        if not bool(cv2.isContourConvex(contour)):
            return self._invalid("polygon_not_convex", polygon)
        signed_area = float(cv2.contourArea(contour, oriented=True))
        area = abs(signed_area)
        if area < self._min_area:
            return self._invalid("polygon_area_too_small", polygon, area)
        frame_area = float(frame_width * frame_height)
        if area > frame_area * self._max_frame_ratio:
            return self._invalid("polygon_area_too_large", polygon, area)
        edges = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
        if not np.isfinite(edges).all() or float(edges.min()) < self._min_edge:
            return self._invalid("polygon_edge_too_short", polygon, area)
        x_span = float(np.ptp(polygon[:, 0]))
        y_span = float(np.ptp(polygon[:, 1]))
        if min(x_span, y_span) <= 0 or max(x_span / y_span, y_span / x_span) > self._max_aspect:
            return self._invalid("polygon_aspect_ratio", polygon, area)
        opposite_ratios = (
            max(edges[0], edges[2]) / max(min(edges[0], edges[2]), 1e-9),
            max(edges[1], edges[3]) / max(min(edges[1], edges[3]), 1e-9),
        )
        if max(opposite_ratios) > self._max_distortion:
            return self._invalid("polygon_perspective_distortion", polygon, area)
        frame_polygon = np.asarray(
            [[0, 0], [frame_width, 0], [frame_width, frame_height], [0, frame_height]],
            dtype=np.float32,
        )
        try:
            visible_area, _ = cv2.intersectConvexConvex(polygon, frame_polygon)
        except cv2.error:
            return self._invalid("polygon_intersection_failed", polygon, area)
        visible_area = float(max(0.0, visible_area))
        visible_ratio = visible_area / area
        if visible_area <= 0 or visible_ratio < self._min_visible:
            return self._invalid("polygon_visibility", polygon, area, visible_area, visible_ratio)
        return ProjectedPolygon(polygon, area, visible_area, visible_ratio, True)

    @staticmethod
    def _invalid(
        reason: str,
        points=None,
        raw_area: float = 0.0,
        visible_area: float = 0.0,
        visible_ratio: float = 0.0,
    ) -> ProjectedPolygon:
        logger.info(
            "matching_geometry_invalid",
            extra={"event": "matching_geometry_invalid", "reason": reason},
        )
        return ProjectedPolygon(points, raw_area, visible_area, visible_ratio, False, reason)


class ConfidenceScorer:
    def __init__(self, settings: Settings) -> None:
        self._minimum_similarity = settings.matching_coarse_min_similarity
        self._maximum_rms = settings.matching_homography_max_rms_reprojection_error
        self._minimum_confidence = settings.matching_min_confidence
        self._weights = (
            settings.matching_confidence_weight_inlier,
            settings.matching_confidence_weight_similarity,
            settings.matching_confidence_weight_reprojection,
            settings.matching_confidence_weight_visibility,
            settings.matching_confidence_weight_coverage,
        )
        if any(weight < 0 or not math.isfinite(weight) for weight in self._weights):
            raise ValueError("Matching confidence agirliklari sonlu ve negatif olmayan olmali.")
        if sum(self._weights) <= 0:
            raise ValueError("Matching confidence agirlik toplami pozitif olmali.")

    def score(
        self,
        matches: CoarseMatchSet,
        homography: HomographyResult,
        polygon: ProjectedPolygon,
    ) -> MatchingQuality | None:
        similarity_raw = (matches.mean_similarity + matches.median_similarity) / 2.0
        denominator = max(1e-9, 1.0 - self._minimum_similarity)
        similarity = min(1.0, max(0.0, (similarity_raw - self._minimum_similarity) / denominator))
        reprojection = min(1.0, max(0.0, 1.0 - homography.rms_reprojection_error / self._maximum_rms))
        visibility = min(1.0, max(0.0, polygon.visible_ratio))
        coverage = min(1.0, max(0.0, math.sqrt(matches.spatial_coverage / 0.25)))
        components = (homography.inlier_ratio, similarity, reprojection, visibility, coverage)
        total_weight = sum(self._weights)
        confidence = sum(weight * value for weight, value in zip(self._weights, components)) / total_weight
        if not math.isfinite(confidence):
            return None
        confidence = min(1.0, max(0.0, confidence))
        quality = MatchingQuality(
            confidence=confidence,
            inlier_ratio=homography.inlier_ratio,
            similarity_score=similarity,
            reprojection_score=reprojection,
            visibility_score=visibility,
            coverage_score=coverage,
        )
        if confidence < self._minimum_confidence:
            logger.info(
                "matching_confidence_below_threshold",
                extra={"event": "matching_confidence_below_threshold", "confidence": confidence},
            )
            return None
        return quality
