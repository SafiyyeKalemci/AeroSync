from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from app.schemas import LandingStatus

logger = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class LandingPolicy:
    edge_margin_px: float
    edge_margin_ratio: float
    min_intersection_pixels: float
    occupancy_ratio: float
    use_center_check: bool
    use_bottom_center_check: bool
    min_area_pixels: float


@dataclass(frozen=True)
class IntersectionMetrics:
    intersection_area: float
    landing_area: float
    obstacle_area: float
    intersection_over_landing_area: float
    obstacle_center_inside: bool
    obstacle_bottom_center_inside: bool


def _valid_bbox(box: BBox) -> bool:
    return (
        len(box) == 4
        and all(math.isfinite(value) for value in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def intersection_metrics(landing: BBox, obstacle: BBox) -> IntersectionMetrics:
    if not _valid_bbox(landing) or not _valid_bbox(obstacle):
        raise ValueError("Bounding box coordinates must be finite and ordered.")
    lx1, ly1, lx2, ly2 = landing
    ox1, oy1, ox2, oy2 = obstacle
    width = max(0.0, min(lx2, ox2) - max(lx1, ox1))
    height = max(0.0, min(ly2, oy2) - max(ly1, oy1))
    intersection = width * height
    landing_area = (lx2 - lx1) * (ly2 - ly1)
    obstacle_area = (ox2 - ox1) * (oy2 - oy1)
    center = ((ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0)
    bottom_center = ((ox1 + ox2) / 2.0, oy2)

    def inside(point: tuple[float, float]) -> bool:
        return lx1 <= point[0] <= lx2 and ly1 <= point[1] <= ly2

    return IntersectionMetrics(
        intersection_area=intersection,
        landing_area=landing_area,
        obstacle_area=obstacle_area,
        intersection_over_landing_area=intersection / landing_area,
        obstacle_center_inside=inside(center),
        obstacle_bottom_center_inside=inside(bottom_center),
    )


class LandingAnalyzer:
    """Pure bounding-box landing policy for one UAP/UAI candidate."""

    def __init__(self, policy: LandingPolicy) -> None:
        self._policy = policy

    def analyze(
        self,
        *,
        raw_bbox: BBox,
        clipped_bbox: BBox,
        frame_width: int,
        frame_height: int,
        obstacles: list[BBox],
    ) -> LandingStatus:
        if frame_width <= 0 or frame_height <= 0 or not _valid_bbox(raw_bbox) or not _valid_bbox(clipped_bbox):
            logger.warning("landing_geometry_invalid")
            return LandingStatus.NOT_APPLICABLE

        margin = max(
            self._policy.edge_margin_px,
            min(frame_width, frame_height) * self._policy.edge_margin_ratio,
        )
        x1, y1, x2, y2 = raw_bbox
        if x1 < -margin or y1 < -margin or x2 > frame_width + margin or y2 > frame_height + margin:
            logger.info("landing_area_out_of_frame")
            return LandingStatus.UNSUITABLE

        cx1, cy1, cx2, cy2 = clipped_bbox
        landing_area = (cx2 - cx1) * (cy2 - cy1)
        if landing_area < self._policy.min_area_pixels:
            logger.warning("landing_area_too_small", extra={"landing_area": landing_area})
            return LandingStatus.NOT_APPLICABLE

        for obstacle in obstacles:
            if not _valid_bbox(obstacle):
                logger.warning("landing_geometry_invalid")
                continue
            metrics = intersection_metrics(clipped_bbox, obstacle)
            occupied = metrics.intersection_area >= self._policy.min_intersection_pixels and (
                metrics.intersection_over_landing_area >= self._policy.occupancy_ratio
                or (self._policy.use_center_check and metrics.obstacle_center_inside)
                or (
                    self._policy.use_bottom_center_check
                    and metrics.obstacle_bottom_center_inside
                )
            )
            if occupied:
                logger.info(
                    "landing_obstacle_detected",
                    extra={
                        "intersection_area": metrics.intersection_area,
                        "intersection_over_landing_area": metrics.intersection_over_landing_area,
                        "obstacle_area": metrics.obstacle_area,
                    },
                )
                return LandingStatus.UNSUITABLE

        logger.info("landing_area_clear")
        return LandingStatus.SUITABLE
