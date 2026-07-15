from __future__ import annotations

import math
from dataclasses import dataclass

from app.schemas import DetectedObject, MotionStatus, ObjectClass
from app.services.detection.homography_motion import (
    HomographyComputation,
    HomographyDiagnostics,
    HomographyMotionAnalyzer,
)
from app.services.detection.motion_analyzer import BBox


@dataclass(frozen=True, slots=True)
class ProjectedVehicle:
    previous_index: int
    previous_bbox: BBox
    projected_bbox: BBox
    projected_polygon: tuple[tuple[float, float], ...]
    projected_center: tuple[float, float]
    visible_ratio: float
    edge_reliable: bool


@dataclass(frozen=True, slots=True)
class BBoxMotionMeasurement:
    current_index: int
    status: MotionStatus
    previous_index: int | None = None
    previous_bbox: BBox | None = None
    projected_bbox: BBox | None = None
    iou: float | None = None
    center_residual_px: float | None = None
    size_ratio: float | None = None
    association_score: float | None = None
    reason: str = "unmatched"


@dataclass(frozen=True, slots=True)
class BBoxMotionAnalysis:
    homography_valid: bool
    diagnostics: HomographyDiagnostics
    measurements: tuple[BBoxMotionMeasurement, ...]


class HomographyBBoxMotionAnalyzer:
    """Projects previous YOLO vehicle boxes using the shared homography estimator."""

    def __init__(
        self,
        homography_analyzer: HomographyMotionAnalyzer,
        *,
        match_min_iou: float,
        match_max_center_distance_ratio: float,
        match_min_score: float,
        stationary_threshold_px: float,
        moving_threshold_px: float,
        min_size_ratio: float,
        max_size_ratio: float,
        min_visible_ratio: float,
    ) -> None:
        self.homography_analyzer = homography_analyzer
        self.match_min_iou = match_min_iou
        self.match_max_center_distance_ratio = match_max_center_distance_ratio
        self.match_min_score = match_min_score
        self.stationary_threshold_px = stationary_threshold_px
        self.moving_threshold_px = moving_threshold_px
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.min_visible_ratio = min_visible_ratio

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
    ) -> BBoxMotionAnalysis:
        computation = homography_computation or self.homography_analyzer.analyze_pair(
            previous_gray, current_gray, exclusion_boxes
        )
        current_vehicles = [
            (index, item)
            for index, item in enumerate(current_detections)
            if item.cls is ObjectClass.TASIT
        ]
        if computation.field is None:
            return BBoxMotionAnalysis(
                False,
                computation.diagnostics,
                tuple(
                    BBoxMotionMeasurement(index, MotionStatus.UNKNOWN, reason="invalid_homography")
                    for index, _ in current_vehicles
                ),
            )
        frame_height, frame_width = current_gray.shape[:2]
        previous_vehicles = [
            (index, item)
            for index, item in enumerate(previous_detections)
            if item.cls is ObjectClass.TASIT
        ]
        projected = [
            item
            for index, detection in previous_vehicles
            if (
                item := self._project_vehicle(
                    index,
                    _bbox(detection),
                    computation.field.homography,
                    frame_width,
                    frame_height,
                )
            )
            is not None
        ]
        candidates: list[tuple[float, int, int, float, float, float]] = []
        for projected_index, previous in enumerate(projected):
            for current_position, (_, current) in enumerate(current_vehicles):
                current_bbox = _bbox(current)
                iou = _iou(previous.projected_bbox, current_bbox)
                center_distance = _point_bbox_center_distance(
                    previous.projected_center, current_bbox
                )
                normalization = max(
                    _diagonal(previous.projected_bbox), _diagonal(current_bbox), 1.0
                )
                center_ratio = center_distance / normalization
                size_ratio = _area(current_bbox) / max(_area(previous.projected_bbox), 1e-12)
                if (
                    iou < self.match_min_iou
                    or center_ratio > self.match_max_center_distance_ratio
                    or size_ratio < self.min_size_ratio
                    or size_ratio > self.max_size_ratio
                ):
                    continue
                center_quality = max(
                    0.0, 1.0 - center_ratio / self.match_max_center_distance_ratio
                )
                size_quality = min(size_ratio, 1.0 / size_ratio)
                score = 0.6 * iou + 0.3 * center_quality + 0.1 * size_quality
                if score >= self.match_min_score:
                    candidates.append(
                        (score, projected_index, current_position, iou, center_distance, size_ratio)
                    )
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        used_previous: set[int] = set()
        used_current: set[int] = set()
        matches: dict[int, BBoxMotionMeasurement] = {}
        for score, projected_index, current_position, iou, distance, size_ratio in candidates:
            if projected_index in used_previous or current_position in used_current:
                continue
            used_previous.add(projected_index)
            used_current.add(current_position)
            previous = projected[projected_index]
            current_index, current = current_vehicles[current_position]
            edge_reliable = previous.edge_reliable and not _touches_edge(
                _bbox(current), frame_width, frame_height
            )
            status, reason = self._classify(distance, iou, edge_reliable)
            matches[current_position] = BBoxMotionMeasurement(
                current_index=current_index,
                status=status,
                previous_index=previous.previous_index,
                previous_bbox=previous.previous_bbox,
                projected_bbox=previous.projected_bbox,
                iou=iou,
                center_residual_px=distance,
                size_ratio=size_ratio,
                association_score=score,
                reason=reason,
            )
        measurements = tuple(
            matches.get(
                position,
                BBoxMotionMeasurement(current_index, MotionStatus.UNKNOWN, reason="unmatched"),
            )
            for position, (current_index, _) in enumerate(current_vehicles)
        )
        return BBoxMotionAnalysis(True, computation.diagnostics, measurements)

    def _project_vehicle(
        self,
        previous_index: int,
        bbox: BBox,
        homography: object,
        frame_width: int,
        frame_height: int,
    ) -> ProjectedVehicle | None:
        import cv2
        import numpy as np

        x1, y1, x2, y2 = bbox
        corners = np.array(
            [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32
        )
        try:
            polygon = cv2.perspectiveTransform(corners, homography)[0]
        except (cv2.error, TypeError, ValueError):
            return None
        if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):
            return None
        polygon_area = abs(float(cv2.contourArea(polygon)))
        if not math.isfinite(polygon_area) or polygon_area <= 1e-6:
            return None
        raw = (
            float(polygon[:, 0].min()),
            float(polygon[:, 1].min()),
            float(polygon[:, 0].max()),
            float(polygon[:, 1].max()),
        )
        clipped = (
            min(max(raw[0], 0.0), float(frame_width)),
            min(max(raw[1], 0.0), float(frame_height)),
            min(max(raw[2], 0.0), float(frame_width)),
            min(max(raw[3], 0.0), float(frame_height)),
        )
        if _area(clipped) <= 0:
            return None
        visible_ratio = _area(clipped) / max(_area(raw), 1e-12)
        points = tuple((float(point[0]), float(point[1])) for point in polygon)
        center = np.array(
            [[[(x1 + x2) / 2, (y1 + y2) / 2]]], dtype=np.float32
        )
        projected_center_array = cv2.perspectiveTransform(center, homography)[0, 0]
        if not np.isfinite(projected_center_array).all():
            return None
        projected_center = (
            float(projected_center_array[0]),
            float(projected_center_array[1]),
        )
        return ProjectedVehicle(
            previous_index,
            bbox,
            clipped,
            points,
            projected_center,
            visible_ratio,
            visible_ratio >= self.min_visible_ratio,
        )

    def _classify(
        self, center_distance: float, iou: float, edge_reliable: bool
    ) -> tuple[MotionStatus, str]:
        if not edge_reliable:
            return MotionStatus.UNKNOWN, "edge_unreliable"
        if center_distance <= self.stationary_threshold_px and iou >= self.match_min_iou:
            return MotionStatus.STATIONARY, "stationary"
        if center_distance >= self.moving_threshold_px:
            return MotionStatus.MOVING, "moving"
        return MotionStatus.UNKNOWN, "hysteresis"


def _bbox(item: DetectedObject) -> BBox:
    return item.top_left_x, item.top_left_y, item.bottom_right_x, item.bottom_right_y


def _area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _iou(first: BBox, second: BBox) -> float:
    intersection = (
        max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
        * max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    )
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _point_bbox_center_distance(point: tuple[float, float], bbox: BBox) -> float:
    bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return math.hypot(point[0] - bbox_center[0], point[1] - bbox_center[1])


def _diagonal(bbox: BBox) -> float:
    return math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])


def _touches_edge(bbox: BBox, width: int, height: int) -> bool:
    tolerance = 1.0
    return (
        bbox[0] <= tolerance
        or bbox[1] <= tolerance
        or bbox[2] >= width - tolerance
        or bbox[3] >= height - tolerance
    )
