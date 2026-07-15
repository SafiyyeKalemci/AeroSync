from __future__ import annotations

import logging
from enum import StrEnum

from app.core.config import Settings
from app.schemas import ImageModality, MatchedReferenceObject

logger = logging.getLogger(__name__)


class PipelineKind(StrEnum):
    SAME_MODAL = "dinov2_aliked_lightglue"
    CROSS_MODAL = "xoftr"


class ModalityDetector:
    """Conservative RGB/thermal estimator derived from the Safiyye heuristic."""

    thermal_threshold = 6.0
    rgb_threshold = 12.0

    def detect(self, image) -> ImageModality:
        import numpy as np

        if getattr(image, "ndim", 0) == 2:
            return ImageModality.THERMAL
        if getattr(image, "ndim", 0) != 3 or image.shape[2] < 3:
            return ImageModality.UNKNOWN
        blue = image[:, :, 0].astype(np.float32)
        green = image[:, :, 1].astype(np.float32)
        red = image[:, :, 2].astype(np.float32)
        channel_difference = float(
            np.mean(np.abs(blue - green)) + np.mean(np.abs(green - red))
        )
        if channel_difference <= self.thermal_threshold:
            return ImageModality.THERMAL
        if channel_difference >= self.rgb_threshold:
            return ImageModality.RGB
        return ImageModality.UNKNOWN


class PipelineSelector:
    def select(
        self,
        reference_modality: ImageModality,
        frame_modality: ImageModality,
    ) -> PipelineKind:
        if (
            reference_modality is not ImageModality.UNKNOWN
            and frame_modality is not ImageModality.UNKNOWN
            and reference_modality is not frame_modality
        ):
            return PipelineKind.CROSS_MODAL
        return PipelineKind.SAME_MODAL


class BoundingBoxValidator:
    def __init__(self, settings: Settings) -> None:
        self._min_confidence = settings.matching_min_confidence
        self._min_area = settings.matching_min_bbox_area
        self._max_area_ratio = settings.matching_max_bbox_area_ratio

    def validate(
        self,
        *,
        object_id: int,
        raw_box: dict[str, float | int] | None,
        image_width: int,
        image_height: int,
    ) -> MatchedReferenceObject | None:
        if raw_box is None or image_width <= 0 or image_height <= 0:
            return None
        try:
            confidence = float(raw_box.get("confidence", raw_box.get("conf", 0.0)))
            if confidence < self._min_confidence or confidence > 1.0:
                return None
            x1 = min(max(float(raw_box["top_left_x"]), 0.0), float(image_width))
            y1 = min(max(float(raw_box["top_left_y"]), 0.0), float(image_height))
            x2 = min(max(float(raw_box["bottom_right_x"]), 0.0), float(image_width))
            y2 = min(max(float(raw_box["bottom_right_y"]), 0.0), float(image_height))
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        area = (x2 - x1) * (y2 - y1)
        image_area = float(image_width * image_height)
        if area < self._min_area or area > image_area * self._max_area_ratio:
            return None
        return MatchedReferenceObject(
            object_id=object_id,
            top_left_x=x1,
            top_left_y=y1,
            bottom_right_x=x2,
            bottom_right_y=y2,
            confidence=confidence,
        )
