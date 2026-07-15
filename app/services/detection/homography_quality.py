from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

QualityLevel = Literal["high", "intermediate", "low"]


@dataclass(frozen=True, slots=True)
class HomographyQualityDecision:
    accepted: bool
    quality_level: QualityLevel
    reason: str
    matches: int
    inliers: int
    inlier_ratio: float
    condition_number: float | None
    reprojection_error: float | None
    spatial_coverage: float | None
    projected_overlap: float | None


class HomographyQualityGate:
    """Fixed-compatible or adaptive validation for an already estimated homography."""

    def __init__(
        self,
        *,
        mode: str,
        fixed_min_inlier_ratio: float,
        high_inlier_ratio: float,
        low_inlier_ratio: float,
        min_matches: int,
        min_inliers: int,
        max_condition_number: float,
        max_reprojection_error_px: float,
        min_spatial_coverage: float,
        min_projected_overlap_ratio: float,
    ) -> None:
        self.mode = mode
        self.fixed_min_inlier_ratio = fixed_min_inlier_ratio
        self.high_inlier_ratio = high_inlier_ratio
        self.low_inlier_ratio = low_inlier_ratio
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.max_condition_number = max_condition_number
        self.max_reprojection_error_px = max_reprojection_error_px
        self.min_spatial_coverage = min_spatial_coverage
        self.min_projected_overlap_ratio = min_projected_overlap_ratio

    def evaluate(
        self,
        matrix: object,
        previous_points: object,
        current_points: object,
        inlier_mask: object,
        *,
        frame_width: int,
        frame_height: int,
    ) -> HomographyQualityDecision:
        import cv2
        import numpy as np

        previous = np.asarray(previous_points, dtype=np.float64).reshape(-1, 2)
        current = np.asarray(current_points, dtype=np.float64).reshape(-1, 2)
        mask = np.asarray(inlier_mask).reshape(-1).astype(bool)
        matches = int(len(previous))
        inliers = int(mask.sum()) if mask.size == matches else 0
        ratio = inliers / matches if matches else 0.0
        level = self._level(ratio)
        candidate = np.asarray(matrix, dtype=np.float64)
        if (
            previous.shape != current.shape
            or mask.size != matches
            or candidate.shape != (3, 3)
            or not np.isfinite(candidate).all()
            or abs(float(candidate[2, 2])) < 1e-12
        ):
            return self._decision(False, level, "invalid_homography", matches, inliers, ratio)
        candidate = candidate / candidate[2, 2]
        determinant = float(np.linalg.det(candidate))
        condition = float(np.linalg.cond(candidate))
        if (
            not math.isfinite(determinant)
            or abs(determinant) < 1e-8
            or abs(determinant) > 1e8
        ):
            return self._decision(
                False, level, "implausible_determinant", matches, inliers, ratio,
                condition,
            )
        if not math.isfinite(condition) or condition > self.max_condition_number:
            return self._decision(
                False, level, "excessive_condition_number", matches, inliers, ratio,
                condition,
            )
        geometry = self._geometry_metrics(
            candidate, frame_width, frame_height
        )
        if geometry is None:
            return self._decision(
                False, level, "invalid_corner_projection", matches, inliers, ratio,
                condition,
            )
        projected_overlap = geometry
        inlier_previous = previous[mask]
        inlier_current = current[mask]
        reprojection = self._symmetric_reprojection_error(
            candidate, inlier_previous, inlier_current
        )
        coverage = self._spatial_coverage(
            inlier_previous, inlier_current, frame_width, frame_height
        )
        common = dict(
            matches=matches,
            inliers=inliers,
            inlier_ratio=ratio,
            condition_number=condition,
            reprojection_error=reprojection,
            spatial_coverage=coverage,
            projected_overlap=projected_overlap,
        )
        if matches < self.min_matches:
            return HomographyQualityDecision(False, level, "insufficient_matches", **common)
        if inliers < self.min_inliers:
            return HomographyQualityDecision(False, level, "insufficient_inliers", **common)
        if self.mode == "fixed":
            accepted = ratio >= self.fixed_min_inlier_ratio
            return HomographyQualityDecision(
                accepted,
                level,
                "fixed_accepted" if accepted else "low_inlier_ratio",
                **common,
            )
        if ratio >= self.high_inlier_ratio:
            return HomographyQualityDecision(True, "high", "adaptive_high_accepted", **common)
        if ratio < self.low_inlier_ratio:
            return HomographyQualityDecision(False, "low", "low_inlier_ratio", **common)
        if reprojection is None or reprojection > self.max_reprojection_error_px:
            return HomographyQualityDecision(
                False, "intermediate", "high_reprojection_error", **common
            )
        if coverage is None or coverage < self.min_spatial_coverage:
            return HomographyQualityDecision(
                False, "intermediate", "low_spatial_coverage", **common
            )
        if projected_overlap < self.min_projected_overlap_ratio:
            return HomographyQualityDecision(
                False, "intermediate", "low_projected_overlap", **common
            )
        return HomographyQualityDecision(
            True, "intermediate", "adaptive_intermediate_accepted", **common
        )

    def reject_without_estimate(
        self, reason: str, *, matches: int = 0
    ) -> HomographyQualityDecision:
        return self._decision(False, "low", reason, matches, 0, 0.0)

    def _level(self, ratio: float) -> QualityLevel:
        if ratio >= self.high_inlier_ratio:
            return "high"
        if ratio >= self.low_inlier_ratio:
            return "intermediate"
        return "low"

    def _geometry_metrics(
        self, matrix: object, width: int, height: int
    ) -> float | None:
        import cv2
        import numpy as np

        corners = np.array(
            [[[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]]],
            dtype=np.float32,
        )
        try:
            projected = cv2.perspectiveTransform(corners, matrix)[0]
        except cv2.error:
            return None
        if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
            return None
        frame_area = max(1.0, float((width - 1) * (height - 1)))
        projected_area = abs(float(cv2.contourArea(projected)))
        area_ratio = projected_area / frame_area
        if not 0.1 <= area_ratio <= 10.0:
            return None
        diagonal = math.hypot(width, height)
        if float(np.max(np.linalg.norm(projected - corners[0], axis=1))) > 2.0 * diagonal:
            return None
        frame_polygon = corners[0].astype(np.float32)
        try:
            overlap_area, _ = cv2.intersectConvexConvex(
                projected.astype(np.float32), frame_polygon
            )
        except cv2.error:
            return None
        return float(overlap_area) / max(min(projected_area, frame_area), 1e-12)

    @staticmethod
    def _symmetric_reprojection_error(
        matrix: object, previous: object, current: object
    ) -> float | None:
        import cv2
        import numpy as np

        if len(previous) == 0:
            return None
        try:
            inverse = np.linalg.inv(matrix)
            forward = cv2.perspectiveTransform(
                np.asarray(previous, np.float32).reshape(1, -1, 2), matrix
            )[0]
            backward = cv2.perspectiveTransform(
                np.asarray(current, np.float32).reshape(1, -1, 2), inverse
            )[0]
        except (cv2.error, np.linalg.LinAlgError):
            return None
        forward_error = np.linalg.norm(forward - current, axis=1)
        backward_error = np.linalg.norm(backward - previous, axis=1)
        combined = np.concatenate([forward_error, backward_error])
        if not np.isfinite(combined).all():
            return None
        return float(np.sqrt(np.mean(np.square(combined))))

    @staticmethod
    def _spatial_coverage(
        previous: object, current: object, width: int, height: int
    ) -> float | None:
        import cv2
        import numpy as np

        if len(previous) < 3 or len(current) < 3:
            return None
        frame_area = max(1.0, float((width - 1) * (height - 1)))
        coverages = []
        for points in (previous, current):
            hull = cv2.convexHull(np.asarray(points, np.float32))
            coverages.append(abs(float(cv2.contourArea(hull))) / frame_area)
        return min(coverages)

    @staticmethod
    def _decision(
        accepted: bool,
        quality_level: QualityLevel,
        reason: str,
        matches: int,
        inliers: int,
        inlier_ratio: float,
        condition_number: float | None = None,
    ) -> HomographyQualityDecision:
        return HomographyQualityDecision(
            accepted,
            quality_level,
            reason,
            matches,
            inliers,
            inlier_ratio,
            condition_number,
            None,
            None,
            None,
        )


def quality_gate_from_settings(
    settings: object,
    *,
    mode: str | None = None,
    fixed_min_inlier_ratio: float | None = None,
) -> HomographyQualityGate:
    return HomographyQualityGate(
        mode=mode or getattr(settings, "detection_motion_homography_quality_gate"),
        fixed_min_inlier_ratio=(
            fixed_min_inlier_ratio
            if fixed_min_inlier_ratio is not None
            else getattr(settings, "detection_motion_homography_min_inlier_ratio")
        ),
        high_inlier_ratio=getattr(
            settings, "detection_motion_homography_adaptive_high_inlier_ratio"
        ),
        low_inlier_ratio=getattr(
            settings, "detection_motion_homography_adaptive_low_inlier_ratio"
        ),
        min_matches=getattr(settings, "detection_motion_homography_min_features"),
        min_inliers=getattr(settings, "detection_motion_homography_min_inliers"),
        max_condition_number=getattr(
            settings, "detection_motion_homography_max_condition_number"
        ),
        max_reprojection_error_px=getattr(
            settings,
            "detection_motion_homography_adaptive_max_reprojection_error_px",
        ),
        min_spatial_coverage=getattr(
            settings, "detection_motion_homography_adaptive_min_spatial_coverage"
        ),
        min_projected_overlap_ratio=getattr(
            settings,
            "detection_motion_homography_adaptive_min_projected_overlap_ratio",
        ),
    )
