from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from app.schemas import DetectedObject, MotionStatus, ObjectClass
from app.services.detection.homography_bbox_motion import HomographyBBoxMotionAnalyzer
from app.services.detection.homography_motion import (
    HomographyComputation,
    HomographyDiagnostics,
    HomographyMotionAnalyzer,
)
from app.services.detection.motion_analyzer import BBox


class SceneQuality(str, Enum):
    LOW_PARALLAX = "low_parallax"
    HIGH_PARALLAX = "high_parallax"
    UNRELIABLE = "unreliable"


@dataclass(frozen=True, slots=True)
class AdaptiveSceneDiagnostics:
    selected_method: str
    scene_quality: SceneQuality
    selection_reason: str
    homography_quality: str | None
    inlier_ratio: float
    reprojection_error: float | None
    spatial_coverage: float | None
    background_residual_median: float | None
    background_residual_p75: float | None
    background_residual_p90: float | None
    background_residual_p95: float | None
    background_grid_spread: float | None
    background_spatial_variance: float | None
    background_gradient_x: float | None
    background_gradient_y: float | None
    valid_background_ratio: float
    valid_background_pixels: int
    grid_medians: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class AdaptiveMotionMeasurement:
    current_index: int
    status: MotionStatus


@dataclass(frozen=True, slots=True)
class AdaptiveMotionAnalysis:
    homography_valid: bool
    diagnostics: HomographyDiagnostics
    scene: AdaptiveSceneDiagnostics
    measurements: tuple[AdaptiveMotionMeasurement, ...]


class HomographyAdaptiveMotionAnalyzer:
    """Selects one existing analyzer; it does not alter either decision algorithm."""

    def __init__(self, homography_analyzer: HomographyMotionAnalyzer,
                 bbox_analyzer: HomographyBBoxMotionAnalyzer, *,
                 background_median_max: float, background_p90_max: float,
                 grid_spread_max: float, min_valid_background_ratio: float,
                 grid_rows: int = 4, grid_columns: int = 4) -> None:
        self.homography_analyzer = homography_analyzer
        self.bbox_analyzer = bbox_analyzer
        self.background_median_max = background_median_max
        self.background_p90_max = background_p90_max
        self.grid_spread_max = grid_spread_max
        self.min_valid_background_ratio = min_valid_background_ratio
        self.grid_rows = grid_rows
        self.grid_columns = grid_columns

    def to_grayscale(self, image: object) -> object:
        return self.homography_analyzer.to_grayscale(image)

    def is_frozen(self, previous_gray: object, current_gray: object) -> bool:
        return self.homography_analyzer.is_frozen(previous_gray, current_gray)

    def analyze(self, previous_gray: object, current_gray: object,
                previous_detections: list[DetectedObject],
                current_detections: list[DetectedObject],
                exclusion_boxes: list[BBox], *,
                homography_computation: HomographyComputation | None = None
                ) -> AdaptiveMotionAnalysis:
        computation = homography_computation or self.homography_analyzer.analyze_pair(
            previous_gray, current_gray, exclusion_boxes)
        vehicles = [(index, item) for index, item in enumerate(current_detections)
                    if item.cls is ObjectClass.TASIT]
        scene = self._classify_scene(computation, [_bbox(item) for _, item in vehicles])
        if scene.selected_method == "homography":
            measurements = tuple(AdaptiveMotionMeasurement(
                index, self.homography_analyzer.classify_vehicle(computation.field, _bbox(item)))
                for index, item in vehicles)
        elif scene.selected_method == "homography_bbox":
            analysis = self.bbox_analyzer.analyze(
                previous_gray, current_gray, previous_detections, current_detections,
                exclusion_boxes, homography_computation=computation)
            measurements = tuple(AdaptiveMotionMeasurement(item.current_index, item.status)
                                 for item in analysis.measurements)
        else:
            measurements = tuple(AdaptiveMotionMeasurement(index, MotionStatus.UNKNOWN)
                                 for index, _ in vehicles)
        return AdaptiveMotionAnalysis(computation.field is not None,
                                      computation.diagnostics, scene, measurements)

    def _classify_scene(self, computation: HomographyComputation,
                        vehicle_boxes: list[BBox]) -> AdaptiveSceneDiagnostics:
        diagnostics = computation.diagnostics
        if computation.field is None:
            return self._unreliable(diagnostics, f"homography_{diagnostics.reason}")
        field = computation.field
        import numpy as np
        flow = np.asarray(field.flow, dtype=np.float64)
        valid = np.asarray(field.valid_mask, dtype=bool)
        domain = np.ones(valid.shape, dtype=bool)
        for x1, y1, x2, y2 in vehicle_boxes:
            left = max(0, min(valid.shape[1], int(math.floor(x1 * field.scale_x))))
            top = max(0, min(valid.shape[0], int(math.floor(y1 * field.scale_y))))
            right = max(0, min(valid.shape[1], int(math.ceil(x2 * field.scale_x))))
            bottom = max(0, min(valid.shape[0], int(math.ceil(y2 * field.scale_y))))
            domain[top:bottom, left:right] = False
        background = domain & valid & np.isfinite(flow).all(axis=2)
        count = int(background.sum())
        ratio = count / int(domain.sum()) if domain.any() else 0.0
        if not diagnostics.quality_accepted or diagnostics.quality_level == "low":
            return self._unreliable(diagnostics, "homography_quality_low", ratio, count)
        if ratio < self.min_valid_background_ratio or count < 4:
            return self._unreliable(diagnostics, "insufficient_background", ratio, count)
        magnitude = np.hypot(flow[:, :, 0] / field.scale_x,
                             flow[:, :, 1] / field.scale_y)
        values = magnitude[background]
        median, p75, p90, p95 = (float(v) for v in np.percentile(values, (50, 75, 90, 95)))
        variance = float(np.var(values))
        grid = self._grid_medians(magnitude, background)
        finite_grid = np.asarray([v for v in grid if v is not None])
        if finite_grid.size < max(4, self.grid_rows):
            return self._unreliable(diagnostics, "insufficient_background_grid", ratio,
                                    count, median, p75, p90, p95, variance, grid)
        spread = float(finite_grid.max() - finite_grid.min())
        gradient_x, gradient_y = self._grid_gradients(grid)
        low = (median <= self.background_median_max and
               p90 <= self.background_p90_max and spread <= self.grid_spread_max)
        return AdaptiveSceneDiagnostics(
            "homography" if low else "homography_bbox",
            SceneQuality.LOW_PARALLAX if low else SceneQuality.HIGH_PARALLAX,
            "background_residual_low_and_homogeneous" if low else
            "background_residual_high_or_spatially_nonuniform",
            diagnostics.quality_level, diagnostics.inlier_ratio,
            diagnostics.reprojection_error, diagnostics.spatial_coverage,
            median, p75, p90, p95, spread, variance, gradient_x, gradient_y,
            ratio, count, grid)

    def _grid_medians(self, magnitude: object, valid: object) -> tuple[float | None, ...]:
        import numpy as np
        values, mask = np.asarray(magnitude), np.asarray(valid, dtype=bool)
        rows = np.array_split(np.arange(values.shape[0]), self.grid_rows)
        columns = np.array_split(np.arange(values.shape[1]), self.grid_columns)
        result: list[float | None] = []
        for row in rows:
            for column in columns:
                cell_values = values[np.ix_(row, column)]
                cell_mask = mask[np.ix_(row, column)]
                result.append(float(np.median(cell_values[cell_mask])) if cell_mask.any() else None)
        return tuple(result)

    def _grid_gradients(self, grid: tuple[float | None, ...]) -> tuple[float, float]:
        import numpy as np
        array = np.asarray([np.nan if v is None else v for v in grid]).reshape(
            self.grid_rows, self.grid_columns)
        return _finite_median(np.abs(np.diff(array, axis=1))), _finite_median(
            np.abs(np.diff(array, axis=0)))

    @staticmethod
    def _unreliable(diagnostics: HomographyDiagnostics, reason: str,
                    ratio: float = 0.0, count: int = 0,
                    median: float | None = None, p75: float | None = None,
                    p90: float | None = None, p95: float | None = None,
                    variance: float | None = None,
                    grid: tuple[float | None, ...] = ()) -> AdaptiveSceneDiagnostics:
        return AdaptiveSceneDiagnostics(
            "unknown", SceneQuality.UNRELIABLE, reason, diagnostics.quality_level,
            diagnostics.inlier_ratio, diagnostics.reprojection_error,
            diagnostics.spatial_coverage, median, p75, p90, p95, None, variance,
            None, None, ratio, count, grid)


def _finite_median(values: object) -> float:
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else 0.0


def _bbox(item: DetectedObject) -> BBox:
    return item.top_left_x, item.top_left_y, item.bottom_right_x, item.bottom_right_y
