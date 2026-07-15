from __future__ import annotations

import math
from dataclasses import dataclass

from app.schemas import DetectedObject, MotionStatus, ObjectClass
from app.services.detection.homography_motion import (
    HomographyComputation,
    HomographyDiagnostics,
    HomographyMotionAnalyzer,
    HomographyMotionField,
)
from app.services.detection.motion_analyzer import BBox


@dataclass(frozen=True, slots=True)
class ResidualStatistics:
    median_x: float
    median_y: float
    vector_magnitude: float
    magnitude_p50: float
    magnitude_p75: float
    magnitude_p90: float
    valid_pixel_count: int


@dataclass(frozen=True, slots=True)
class LocalMotionMeasurement:
    vehicle_index: int
    bbox: BBox
    homography_valid: bool
    homography_quality_level: str | None
    vehicle_statistics: ResidualStatistics | None
    background_statistics: ResidualStatistics | None
    corrected_residual_x: float | None
    corrected_residual_y: float | None
    corrected_residual_magnitude: float | None
    background_valid_ratio: float
    stationary_threshold: float
    moving_threshold: float
    final_result: MotionStatus
    decision_reason: str


@dataclass(frozen=True, slots=True)
class LocalMotionAnalysis:
    homography_valid: bool
    diagnostics: HomographyDiagnostics
    measurements: tuple[LocalMotionMeasurement, ...]


class HomographyLocalMotionAnalyzer:
    """Compares each vehicle residual vector with its local background ring."""

    def __init__(
        self,
        homography_analyzer: HomographyMotionAnalyzer,
        *,
        ring_expansion_ratio: float,
        min_background_pixels: int,
        stationary_threshold_px: float,
        moving_threshold_px: float,
        min_valid_ratio: float,
    ) -> None:
        self.homography_analyzer = homography_analyzer
        self.ring_expansion_ratio = ring_expansion_ratio
        self.min_background_pixels = min_background_pixels
        self.stationary_threshold_px = stationary_threshold_px
        self.moving_threshold_px = moving_threshold_px
        self.min_valid_ratio = min_valid_ratio

    def to_grayscale(self, image: object) -> object:
        return self.homography_analyzer.to_grayscale(image)

    def is_frozen(self, previous_gray: object, current_gray: object) -> bool:
        return self.homography_analyzer.is_frozen(previous_gray, current_gray)

    def analyze(
        self,
        previous_gray: object,
        current_gray: object,
        current_detections: list[DetectedObject],
        exclusion_boxes: list[BBox],
        *,
        homography_computation: HomographyComputation | None = None,
    ) -> LocalMotionAnalysis:
        computation = homography_computation or self.homography_analyzer.analyze_pair(
            previous_gray, current_gray, exclusion_boxes
        )
        measurements = tuple(
            self._measure(index, detection, current_detections, computation)
            for index, detection in enumerate(current_detections)
            if detection.cls is ObjectClass.TASIT
        )
        return LocalMotionAnalysis(
            homography_valid=computation.field is not None,
            diagnostics=computation.diagnostics,
            measurements=measurements,
        )

    def _measure(
        self,
        index: int,
        detection: DetectedObject,
        all_detections: list[DetectedObject],
        computation: HomographyComputation,
    ) -> LocalMotionMeasurement:
        bbox = _bbox(detection)
        field = computation.field
        if field is None:
            return self._result(
                index,
                bbox,
                computation.diagnostics,
                final=MotionStatus.UNKNOWN,
                reason="invalid_homography",
            )

        vehicle_mask = self._vehicle_mask(field, bbox)
        vehicle = _statistics(field, vehicle_mask)
        if (
            vehicle is None
            or vehicle.valid_pixel_count < self.homography_analyzer.min_valid_pixels
        ):
            return self._result(
                index,
                bbox,
                computation.diagnostics,
                vehicle=vehicle,
                final=MotionStatus.UNKNOWN,
                reason="insufficient_vehicle_pixels",
            )

        ring_mask, intended_ring_pixels = self._background_ring_mask(
            field, bbox, all_detections
        )
        background = _statistics(field, ring_mask)
        background_pixels = background.valid_pixel_count if background else 0
        valid_ratio = min(1.0, background_pixels / max(intended_ring_pixels, 1))
        if background_pixels < self.min_background_pixels:
            return self._result(
                index,
                bbox,
                computation.diagnostics,
                vehicle=vehicle,
                background=background,
                background_valid_ratio=valid_ratio,
                final=MotionStatus.UNKNOWN,
                reason="insufficient_background_pixels",
            )
        if valid_ratio < self.min_valid_ratio:
            return self._result(
                index,
                bbox,
                computation.diagnostics,
                vehicle=vehicle,
                background=background,
                background_valid_ratio=valid_ratio,
                final=MotionStatus.UNKNOWN,
                reason="insufficient_background_ratio",
            )
        assert background is not None
        corrected_x = vehicle.median_x - background.median_x
        corrected_y = vehicle.median_y - background.median_y
        corrected_magnitude = math.hypot(corrected_x, corrected_y)
        if not math.isfinite(corrected_magnitude):
            return self._result(
                index,
                bbox,
                computation.diagnostics,
                vehicle=vehicle,
                background=background,
                background_valid_ratio=valid_ratio,
                final=MotionStatus.UNKNOWN,
                reason="non_finite_corrected_residual",
            )
        if corrected_magnitude <= self.stationary_threshold_px:
            final, reason = MotionStatus.STATIONARY, "corrected_stationary"
        elif corrected_magnitude >= self.moving_threshold_px:
            final, reason = MotionStatus.MOVING, "corrected_moving"
        else:
            final, reason = MotionStatus.UNKNOWN, "corrected_hysteresis"
        return self._result(
            index,
            bbox,
            computation.diagnostics,
            vehicle=vehicle,
            background=background,
            corrected_x=corrected_x,
            corrected_y=corrected_y,
            corrected_magnitude=corrected_magnitude,
            background_valid_ratio=valid_ratio,
            final=final,
            reason=reason,
        )

    def _vehicle_mask(self, field: HomographyMotionField, bbox: BBox) -> object:
        import numpy as np

        mask = np.zeros(field.valid_mask.shape, dtype=bool)
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        crop = self.homography_analyzer.inner_crop_ratio
        _fill(mask, (x1 + width * crop, y1 + height * crop, x2 - width * crop, y2 - height * crop), field, True)
        return mask

    def _background_ring_mask(
        self,
        field: HomographyMotionField,
        bbox: BBox,
        detections: list[DetectedObject],
    ) -> tuple[object, int]:
        import numpy as np

        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        dx, dy = width * self.ring_expansion_ratio, height * self.ring_expansion_ratio
        expanded = (x1 - dx, y1 - dy, x2 + dx, y2 + dy)
        ring = np.zeros(field.valid_mask.shape, dtype=bool)
        _fill(ring, expanded, field, True)
        _fill(ring, bbox, field, False)
        intended_outer_width = max(0, math.ceil((expanded[2] - expanded[0]) * field.scale_x))
        intended_outer_height = max(0, math.ceil((expanded[3] - expanded[1]) * field.scale_y))
        intended_inner_width = max(0, math.ceil((x2 - x1) * field.scale_x))
        intended_inner_height = max(0, math.ceil((y2 - y1) * field.scale_y))
        intended = max(
            0,
            intended_outer_width * intended_outer_height
            - intended_inner_width * intended_inner_height,
        )
        for item in detections:
            _fill(ring, _bbox(item), field, False)
        ring &= np.asarray(field.valid_mask, dtype=bool)
        return ring, intended

    def _result(
        self,
        index: int,
        bbox: BBox,
        diagnostics: HomographyDiagnostics,
        *,
        vehicle: ResidualStatistics | None = None,
        background: ResidualStatistics | None = None,
        corrected_x: float | None = None,
        corrected_y: float | None = None,
        corrected_magnitude: float | None = None,
        background_valid_ratio: float = 0.0,
        final: MotionStatus,
        reason: str,
    ) -> LocalMotionMeasurement:
        return LocalMotionMeasurement(
            vehicle_index=index,
            bbox=bbox,
            homography_valid=diagnostics.valid,
            homography_quality_level=diagnostics.quality_level,
            vehicle_statistics=vehicle,
            background_statistics=background,
            corrected_residual_x=corrected_x,
            corrected_residual_y=corrected_y,
            corrected_residual_magnitude=corrected_magnitude,
            background_valid_ratio=background_valid_ratio,
            stationary_threshold=self.stationary_threshold_px,
            moving_threshold=self.moving_threshold_px,
            final_result=final,
            decision_reason=reason,
        )


def _statistics(field: HomographyMotionField, selection: object) -> ResidualStatistics | None:
    import numpy as np

    flow = np.asarray(field.flow)
    selected = np.asarray(selection, dtype=bool) & np.asarray(field.valid_mask, dtype=bool)
    selected &= np.isfinite(flow).all(axis=2)
    count = int(selected.sum())
    if count == 0:
        return None
    x = flow[:, :, 0][selected] / field.scale_x
    y = flow[:, :, 1][selected] / field.scale_y
    magnitudes = np.hypot(x, y)
    values = (
        float(np.median(x)),
        float(np.median(y)),
        float(np.percentile(magnitudes, 50)),
        float(np.percentile(magnitudes, 75)),
        float(np.percentile(magnitudes, 90)),
    )
    if not all(math.isfinite(value) for value in values):
        return None
    median_x, median_y, p50, p75, p90 = values
    return ResidualStatistics(
        median_x=median_x,
        median_y=median_y,
        vector_magnitude=math.hypot(median_x, median_y),
        magnitude_p50=p50,
        magnitude_p75=p75,
        magnitude_p90=p90,
        valid_pixel_count=count,
    )


def _fill(mask: object, bbox: BBox, field: HomographyMotionField, value: bool) -> None:
    x1, y1, x2, y2 = bbox
    left = max(0, min(mask.shape[1], math.floor(x1 * field.scale_x)))
    top = max(0, min(mask.shape[0], math.floor(y1 * field.scale_y)))
    right = max(0, min(mask.shape[1], math.ceil(x2 * field.scale_x)))
    bottom = max(0, min(mask.shape[0], math.ceil(y2 * field.scale_y)))
    if right > left and bottom > top:
        mask[top:bottom, left:right] = value


def _bbox(item: DetectedObject) -> BBox:
    return item.top_left_x, item.top_left_y, item.bottom_right_x, item.bottom_right_y
