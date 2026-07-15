from __future__ import annotations

import logging
import math

from app.core.config import Settings
from app.services.matching.descriptor_types import ProjectedPolygon, ValidatedBoundingBox

logger = logging.getLogger(__name__)


class ProjectedBoundingBoxValidator:
    def __init__(self, settings: Settings) -> None:
        self._min_width = settings.matching_bbox_min_width_px
        self._min_height = settings.matching_bbox_min_height_px
        self._min_area = settings.matching_bbox_min_area_px
        self._max_frame_ratio = settings.matching_bbox_max_frame_area_ratio
        if (
            self._min_width <= 0 or self._min_height <= 0 or self._min_area <= 0
            or not 0 < self._max_frame_ratio <= 1
        ):
            raise ValueError("Matching bbox config gecersiz.")

    def validate(
        self,
        polygon: ProjectedPolygon,
        *,
        frame_width: int,
        frame_height: int,
    ) -> ValidatedBoundingBox | None:
        import numpy as np

        if not polygon.valid or polygon.points is None or frame_width <= 0 or frame_height <= 0:
            return self._invalid("polygon_or_frame_invalid")
        points = np.asarray(polygon.points, dtype=np.float64)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            return self._invalid("bbox_points_invalid")
        raw = (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )
        if not all(math.isfinite(value) for value in raw):
            return self._invalid("bbox_non_finite")
        clipped = (
            min(max(raw[0], 0.0), float(frame_width)),
            min(max(raw[1], 0.0), float(frame_height)),
            min(max(raw[2], 0.0), float(frame_width)),
            min(max(raw[3], 0.0), float(frame_height)),
        )
        width = clipped[2] - clipped[0]
        height = clipped[3] - clipped[1]
        area = width * height
        if width < self._min_width or height < self._min_height or area < self._min_area:
            return self._invalid("bbox_too_small")
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return self._invalid("bbox_empty")
        if area > frame_width * frame_height * self._max_frame_ratio:
            return self._invalid("bbox_too_large")
        return ValidatedBoundingBox(raw_box=raw, clipped_box=clipped, area=area)

    @staticmethod
    def _invalid(reason: str) -> None:
        logger.info(
            "matching_bbox_invalid",
            extra={"event": "matching_bbox_invalid", "reason": reason},
        )
        return None
