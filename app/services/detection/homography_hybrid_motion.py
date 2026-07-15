from __future__ import annotations

from dataclasses import dataclass

from app.schemas import DetectedObject, MotionStatus
from app.services.detection.homography_bbox_motion import (
    BBoxMotionMeasurement,
    HomographyBBoxMotionAnalyzer,
)
from app.services.detection.homography_motion import (
    HomographyComputation,
    HomographyDiagnostics,
    VehicleMotionMeasurement,
)
from app.services.detection.motion_analyzer import BBox


@dataclass(frozen=True, slots=True)
class HybridMotionMeasurement:
    current_index: int
    bbox_result: MotionStatus
    flow_result: MotionStatus
    final_result: MotionStatus
    bbox_iou: float | None
    bbox_center_residual_px: float | None
    bbox_association_score: float | None
    flow_residual_px: float | None
    homography_quality_level: str | None
    decision_reason: str
    projected_bbox: BBox | None = None
    previous_index: int | None = None


@dataclass(frozen=True, slots=True)
class HybridMotionAnalysis:
    homography_valid: bool
    diagnostics: HomographyDiagnostics
    measurements: tuple[HybridMotionMeasurement, ...]


class HomographyHybridMotionAnalyzer:
    """Conservatively combines existing bbox and residual-flow evidence."""

    def __init__(
        self,
        bbox_analyzer: HomographyBBoxMotionAnalyzer,
        *,
        strong_moving_residual_px: float,
        min_association_score: float,
        min_iou: float,
    ) -> None:
        self.bbox_analyzer = bbox_analyzer
        self.homography_analyzer = bbox_analyzer.homography_analyzer
        self.strong_moving_residual_px = strong_moving_residual_px
        self.min_association_score = min_association_score
        self.min_iou = min_iou

    def to_grayscale(self, image: object) -> object:
        return self.homography_analyzer.to_grayscale(image)

    def is_frozen(self, previous_gray: object, current_gray: object) -> bool:
        return self.homography_analyzer.is_frozen(previous_gray, current_gray)

    def analyze(
        self,
        previous_gray: object,
        current_gray: object,
        previous_detections: list[DetectedObject],
        current_detections: list[DetectedObject],
        exclusion_boxes: list[BBox],
        *,
        homography_computation: HomographyComputation | None = None,
    ) -> HybridMotionAnalysis:
        computation = homography_computation or self.homography_analyzer.analyze_pair(
            previous_gray, current_gray, exclusion_boxes
        )
        bbox_analysis = self.bbox_analyzer.analyze(
            previous_gray,
            current_gray,
            previous_detections,
            current_detections,
            exclusion_boxes,
            homography_computation=computation,
        )
        measurements: list[HybridMotionMeasurement] = []
        for bbox in bbox_analysis.measurements:
            detection = current_detections[bbox.current_index]
            flow = self.homography_analyzer.measure_vehicle(
                computation.field,
                (
                    detection.top_left_x,
                    detection.top_left_y,
                    detection.bottom_right_x,
                    detection.bottom_right_y,
                ),
            )
            final, reason = self._decide(bbox, flow, computation.diagnostics)
            measurements.append(
                HybridMotionMeasurement(
                    current_index=bbox.current_index,
                    bbox_result=bbox.status,
                    flow_result=flow.status,
                    final_result=final,
                    bbox_iou=bbox.iou,
                    bbox_center_residual_px=bbox.center_residual_px,
                    bbox_association_score=bbox.association_score,
                    flow_residual_px=flow.residual_motion_magnitude,
                    homography_quality_level=computation.diagnostics.quality_level,
                    decision_reason=reason,
                    projected_bbox=bbox.projected_bbox,
                    previous_index=bbox.previous_index,
                )
            )
        return HybridMotionAnalysis(
            homography_valid=computation.field is not None,
            diagnostics=computation.diagnostics,
            measurements=tuple(measurements),
        )

    def _decide(
        self,
        bbox: BBoxMotionMeasurement,
        flow: VehicleMotionMeasurement,
        diagnostics: HomographyDiagnostics,
    ) -> tuple[MotionStatus, str]:
        if not diagnostics.valid:
            return MotionStatus.UNKNOWN, "invalid_homography"
        if bbox.reason == "edge_unreliable":
            return MotionStatus.UNKNOWN, "frame_edge"

        association_reliable = (
            bbox.previous_index is not None
            and bbox.association_score is not None
            and bbox.association_score >= self.min_association_score
            and bbox.iou is not None
            and bbox.iou >= self.min_iou
        )
        bbox_result = bbox.status if association_reliable else MotionStatus.UNKNOWN

        if bbox_result in {MotionStatus.STATIONARY, MotionStatus.MOVING}:
            if flow.status is bbox_result:
                return bbox_result, "bbox_and_flow_agree"
            if flow.status in {MotionStatus.STATIONARY, MotionStatus.MOVING}:
                return MotionStatus.UNKNOWN, "bbox_flow_conflict"
            return bbox_result, "reliable_bbox_flow_unknown"

        if (
            flow.status is MotionStatus.MOVING
            and flow.residual_motion_magnitude is not None
            and flow.residual_motion_magnitude >= self.strong_moving_residual_px
            and diagnostics.quality_level == "high"
        ):
            return MotionStatus.MOVING, "strong_flow_with_high_quality_homography"
        if not association_reliable:
            return MotionStatus.UNKNOWN, "bbox_association_unreliable"
        return MotionStatus.UNKNOWN, "insufficient_consistent_evidence"
