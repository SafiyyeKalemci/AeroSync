from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from app.core.config import Settings
from app.schemas import DetectedObject, LandingStatus, MotionStatus, ObjectClass
from app.services.common import FrameContext
from app.services.detection.class_mapping import YoloClassMapper
from app.services.detection.homography_adaptive_motion import (
    AdaptiveMotionAnalysis,
    HomographyAdaptiveMotionAnalyzer,
)
from app.services.detection.homography_bbox_motion import (
    BBoxMotionAnalysis,
    HomographyBBoxMotionAnalyzer,
)
from app.services.detection.homography_motion import HomographyMotionAnalyzer
from app.services.detection.homography_hybrid_motion import (
    HomographyHybridMotionAnalyzer,
    HybridMotionAnalysis,
)
from app.services.detection.homography_local_motion import (
    HomographyLocalMotionAnalyzer,
    LocalMotionAnalysis,
)
from app.services.detection.homography_quality import quality_gate_from_settings
from app.services.detection.interface import DetectionService
from app.services.detection.landing_analyzer import LandingAnalyzer, LandingPolicy
from app.services.detection.motion_analyzer import BBox, MotionAnalyzer, MotionField
from app.services.detection.session_state import DetectionSessionState
from app.services.detection.session_store import DetectionSessionStore
from app.services.detection.yolo_runtime import YoloRuntime
from app.utils.images import read_image_bytes

logger = logging.getLogger(__name__)

ImageReader = Callable[[str, float], Awaitable[bytes]]
ImageDecoder = Callable[[bytes], object]


@dataclass(frozen=True)
class _DetectionCandidate:
    cls: ObjectClass
    confidence: float
    raw_bbox: BBox
    clipped_bbox: BBox

    def as_detected_object(self) -> DetectedObject:
        x1, y1, x2, y2 = self.clipped_bbox
        return DetectedObject(
            cls=self.cls,
            top_left_x=x1,
            top_left_y=y1,
            bottom_right_x=x2,
            bottom_right_y=y2,
            confidence=self.confidence,
            motion_status=MotionStatus.UNKNOWN,
            landing_status=LandingStatus.NOT_APPLICABLE,
        )


def decode_opencv_image(content: bytes) -> object:
    if not content:
        raise ValueError("Görüntü içeriği boş.")
    import cv2
    import numpy as np

    encoded = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("OpenCV görüntüyü decode edemedi.")
    return image


def _first(value: object) -> object:
    try:
        return value[0]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return value


def _to_list(value: object) -> list[object]:
    candidate = value
    if hasattr(candidate, "tolist"):
        candidate = candidate.tolist()
    if isinstance(candidate, (list, tuple)):
        if len(candidate) == 1 and isinstance(candidate[0], (list, tuple)):
            candidate = candidate[0]
        return list(candidate)
    return []


def _validated_bbox(
    raw_bbox: Iterable[object], width: int, height: int
) -> tuple[BBox, BBox] | None:
    try:
        values = [float(value) for value in raw_bbox]
    except (TypeError, ValueError, OverflowError):
        return None
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    raw = (x1, y1, x2, y2)
    clipped = (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return raw, clipped


class YoloDetectionService(DetectionService):
    """Single-frame YOLO detection with session-based vehicle motion analysis."""

    def __init__(
        self,
        settings: Settings,
        *,
        runtime: YoloRuntime | None = None,
        class_mapper: YoloClassMapper | None = None,
        image_reader: ImageReader = read_image_bytes,
        image_decoder: ImageDecoder = decode_opencv_image,
        motion_analyzer: (
            MotionAnalyzer | HomographyMotionAnalyzer | HomographyBBoxMotionAnalyzer | HomographyHybridMotionAnalyzer | HomographyLocalMotionAnalyzer | HomographyAdaptiveMotionAnalyzer | None
        ) = None,
        landing_analyzer: LandingAnalyzer | None = None,
        session_store: DetectionSessionStore | None = None,
    ) -> None:
        settings.validate_detection_motion()
        settings.validate_detection_landing()
        self._settings = settings
        self._runtime = runtime or YoloRuntime(
            settings.detection_model_path,
            settings.detection_confidence,
            settings.detection_iou,
        )
        self._class_mapper = class_mapper or YoloClassMapper()
        self._image_reader = image_reader
        self._image_decoder = image_decoder
        self._motion_analyzer = motion_analyzer or self._build_motion_analyzer(settings)
        self._landing_analyzer = landing_analyzer or LandingAnalyzer(
            LandingPolicy(
                edge_margin_px=settings.detection_landing_edge_margin_px,
                edge_margin_ratio=settings.detection_landing_edge_margin_ratio,
                min_intersection_pixels=settings.detection_landing_min_intersection_pixels,
                occupancy_ratio=settings.detection_landing_occupancy_ratio,
                use_center_check=settings.detection_landing_use_center_check,
                use_bottom_center_check=settings.detection_landing_use_bottom_center_check,
                min_area_pixels=settings.detection_landing_min_area_pixels,
            )
        )
        self._sessions = session_store if session_store is not None else DetectionSessionStore(
            ttl_seconds=settings.detection_motion_session_ttl_seconds,
            max_sessions=settings.detection_motion_max_sessions,
        )

    @staticmethod
    def _build_motion_analyzer(
        settings: Settings,
    ) -> MotionAnalyzer | HomographyMotionAnalyzer | HomographyBBoxMotionAnalyzer | HomographyHybridMotionAnalyzer | HomographyLocalMotionAnalyzer | HomographyAdaptiveMotionAnalyzer:
        common = {
            "min_valid_pixels": settings.detection_motion_min_valid_pixels,
            "inner_crop_ratio": settings.detection_motion_inner_crop_ratio,
            "flow_downscale": settings.detection_motion_flow_downscale,
            "freeze_threshold": settings.detection_motion_freeze_threshold,
        }
        if settings.detection_motion_method in {
            "homography",
            "homography_bbox",
            "homography_hybrid",
            "homography_local",
            "homography_adaptive",
        }:
            homography = HomographyMotionAnalyzer(
                min_features=settings.detection_motion_homography_min_features,
                min_inliers=settings.detection_motion_homography_min_inliers,
                min_inlier_ratio=settings.detection_motion_homography_min_inlier_ratio,
                ransac_threshold=settings.detection_motion_homography_ransac_threshold,
                max_condition_number=settings.detection_motion_homography_max_condition_number,
                residual_threshold_px=(
                    settings.detection_motion_homography_residual_threshold_px
                ),
                quality_gate=quality_gate_from_settings(settings),
                **common,
            )
            if settings.detection_motion_method == "homography":
                return homography
            if settings.detection_motion_method == "homography_local":
                return HomographyLocalMotionAnalyzer(
                    homography,
                    ring_expansion_ratio=(
                        settings.detection_motion_local_ring_expansion_ratio
                    ),
                    min_background_pixels=(
                        settings.detection_motion_local_min_background_pixels
                    ),
                    stationary_threshold_px=(
                        settings.detection_motion_local_stationary_threshold_px
                    ),
                    moving_threshold_px=(
                        settings.detection_motion_local_moving_threshold_px
                    ),
                    min_valid_ratio=settings.detection_motion_local_min_valid_ratio,
                )
            bbox = HomographyBBoxMotionAnalyzer(
                homography,
                match_min_iou=settings.detection_motion_bbox_match_min_iou,
                match_max_center_distance_ratio=(
                    settings.detection_motion_bbox_match_max_center_distance_ratio
                ),
                match_min_score=settings.detection_motion_bbox_match_min_score,
                stationary_threshold_px=(
                    settings.detection_motion_bbox_stationary_threshold_px
                ),
                moving_threshold_px=settings.detection_motion_bbox_moving_threshold_px,
                min_size_ratio=settings.detection_motion_bbox_min_size_ratio,
                max_size_ratio=settings.detection_motion_bbox_max_size_ratio,
                min_visible_ratio=settings.detection_motion_bbox_min_visible_ratio,
            )
            if settings.detection_motion_method == "homography_bbox":
                return bbox
            if settings.detection_motion_method == "homography_adaptive":
                return HomographyAdaptiveMotionAnalyzer(
                    homography,
                    bbox,
                    background_median_max=settings.detection_motion_adaptive_background_median_max,
                    background_p90_max=settings.detection_motion_adaptive_background_p90_max,
                    grid_spread_max=settings.detection_motion_adaptive_grid_spread_max,
                    min_valid_background_ratio=settings.detection_motion_adaptive_min_valid_background_ratio,
                )
            return HomographyHybridMotionAnalyzer(
                bbox,
                strong_moving_residual_px=(
                    settings.detection_motion_hybrid_strong_moving_residual_px
                ),
                min_association_score=(
                    settings.detection_motion_hybrid_min_association_score
                ),
                min_iou=settings.detection_motion_hybrid_min_iou,
            )
        return MotionAnalyzer(
            threshold_px=settings.detection_motion_threshold_px,
            **common,
        )

    async def reset_session(self, session_id: str) -> None:
        self._sessions.reset(session_id)

    async def reset_all_sessions(self) -> None:
        self._sessions.reset_all()

    async def process_frame(self, frame: FrameContext) -> list[DetectedObject]:
        state = self._sessions.get_or_create(frame.session_id)
        async with state.lock:
            if state.last_processed_frame_id == frame.frame_id:
                cached = state.cached_result()
                logger.info(
                    "motion_duplicate_frame",
                    extra={"session_id": frame.session_id, "frame_id": frame.frame_id},
                )
                if cached is not None:
                    state.touch()
                    return cached
            return await self._process_locked(frame, state)

    async def _process_locked(
        self, frame: FrameContext, state: DetectionSessionState
    ) -> list[DetectedObject]:
        try:
            content = await self._image_reader(
                frame.image_url, self._settings.http_timeout_seconds
            )
            image = await asyncio.to_thread(self._image_decoder, content)
            shape = getattr(image, "shape", ())
            if len(shape) < 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
                raise ValueError("Decoded görüntü boyutu geçersiz.")
            height, width = int(shape[0]), int(shape[1])
            raw_results = await asyncio.to_thread(self._runtime.predict, image)
        except Exception:
            logger.error(
                "detection_frame_prepare_failed",
                extra={"frame_id": frame.frame_id, "session_id": frame.session_id},
                exc_info=True,
            )
            return []

        candidates: list[_DetectionCandidate] = []
        for result in raw_results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                candidate = self._convert_box(box, width, height)
                if candidate is not None:
                    candidates.append(candidate)
        detections = [candidate.as_detected_object() for candidate in candidates]
        if not self._settings.detection_motion_enabled:
            return self._apply_landing(detections, candidates, width, height)
        try:
            outcome = await asyncio.to_thread(
                self._analyze_motion,
                frame,
                state,
                image,
                detections,
            )
        except Exception:
            logger.error(
                "motion_analysis_failed",
                extra={"session_id": frame.session_id, "frame_id": frame.frame_id},
                exc_info=True,
            )
            return self._apply_landing(detections, candidates, width, height)
        gray, results, reset_warmup, frozen = outcome
        results = self._apply_landing(results, candidates, width, height)
        shape = (int(gray.shape[0]), int(gray.shape[1]))
        state.replace_baseline(
            video_name=frame.video_name,
            frame_gray=gray,
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            shape=shape,
            result=results,
            reset_warmup=reset_warmup,
            frozen=frozen,
        )
        return results

    def _apply_landing(
        self,
        detections: list[DetectedObject],
        candidates: list[_DetectionCandidate],
        width: int,
        height: int,
    ) -> list[DetectedObject]:
        if not self._settings.detection_landing_enabled:
            return detections
        obstacles = [
            candidate.clipped_bbox
            for candidate in candidates
            if candidate.cls in {ObjectClass.TASIT, ObjectClass.INSAN}
        ]
        results: list[DetectedObject] = []
        for item, candidate in zip(detections, candidates, strict=True):
            if candidate.cls not in {ObjectClass.UAP, ObjectClass.UAI}:
                results.append(item)
                continue
            try:
                status = self._landing_analyzer.analyze(
                    raw_bbox=candidate.raw_bbox,
                    clipped_bbox=candidate.clipped_bbox,
                    frame_width=width,
                    frame_height=height,
                    obstacles=obstacles,
                )
            except Exception:
                logger.warning("landing_analysis_failed", exc_info=True)
                status = LandingStatus.NOT_APPLICABLE
            results.append(item.model_copy(update={"landing_status": status}))
        return results

    def _analyze_motion(
        self,
        frame: FrameContext,
        state: DetectionSessionState,
        image: object,
        detections: list[DetectedObject],
    ) -> tuple[object, list[DetectedObject], bool, bool]:
        gray = self._motion_analyzer.to_grayscale(image)
        shape = (int(gray.shape[0]), int(gray.shape[1]))
        reset_warmup = False
        frozen = False
        field: MotionField | object | None = None
        bbox_analysis: BBoxMotionAnalysis | None = None
        hybrid_analysis: HybridMotionAnalysis | None = None
        local_analysis: LocalMotionAnalysis | None = None
        adaptive_analysis: AdaptiveMotionAnalysis | None = None
        can_compare = state.previous_frame_gray is not None
        if not can_compare:
            logger.info("motion_first_frame", extra={"session_id": frame.session_id})
        elif state.video_name != frame.video_name:
            reset_warmup = True
            can_compare = False
            logger.info("motion_gap_detected", extra={"reason": "video_changed", "session_id": frame.session_id})
        elif state.previous_shape != shape:
            reset_warmup = True
            can_compare = False
            logger.info("motion_shape_changed", extra={"session_id": frame.session_id})
        elif state.previous_frame_index is None or frame.frame_index is None:
            reset_warmup = True
            can_compare = False
            logger.info("motion_gap_detected", extra={"reason": "frame_index_missing", "session_id": frame.session_id})
        else:
            gap = frame.frame_index - state.previous_frame_index
            if gap <= 0:
                reset_warmup = True
                can_compare = False
                logger.info("motion_out_of_order", extra={"frame_gap": gap, "session_id": frame.session_id})
            elif gap > self._settings.detection_motion_max_frame_gap:
                reset_warmup = True
                can_compare = False
                logger.info("motion_gap_detected", extra={"frame_gap": gap, "session_id": frame.session_id})

        if can_compare:
            frozen = self._motion_analyzer.is_frozen(state.previous_frame_gray, gray)
            if frozen:
                logger.info("motion_freeze_detected", extra={"session_id": frame.session_id})
            elif any(item.cls is ObjectClass.TASIT for item in detections):
                exclusions: list[BBox] = [
                    (
                        item.top_left_x,
                        item.top_left_y,
                        item.bottom_right_x,
                        item.bottom_right_y,
                    )
                    for item in detections
                    if item.cls in {ObjectClass.TASIT, ObjectClass.INSAN}
                ]
                if isinstance(self._motion_analyzer, HomographyAdaptiveMotionAnalyzer):
                    adaptive_analysis = self._motion_analyzer.analyze(
                        state.previous_frame_gray,
                        gray,
                        state.last_result or [],
                        detections,
                        exclusions,
                    )
                    if adaptive_analysis.homography_valid:
                        field = adaptive_analysis
                elif isinstance(self._motion_analyzer, HomographyLocalMotionAnalyzer):
                    local_analysis = self._motion_analyzer.analyze(
                        state.previous_frame_gray,
                        gray,
                        detections,
                        exclusions,
                    )
                    if local_analysis.homography_valid:
                        field = local_analysis
                elif isinstance(self._motion_analyzer, HomographyHybridMotionAnalyzer):
                    hybrid_analysis = self._motion_analyzer.analyze(
                        state.previous_frame_gray,
                        gray,
                        state.last_result or [],
                        detections,
                        exclusions,
                    )
                    if hybrid_analysis.homography_valid:
                        field = hybrid_analysis
                elif isinstance(self._motion_analyzer, HomographyBBoxMotionAnalyzer):
                    bbox_analysis = self._motion_analyzer.analyze(
                        state.previous_frame_gray,
                        gray,
                        state.last_result or [],
                        detections,
                        exclusions,
                    )
                    if bbox_analysis.homography_valid:
                        field = bbox_analysis
                else:
                    field = self._motion_analyzer.compute_flow(
                        state.previous_frame_gray,
                        gray,
                        exclusions,
                    )

        next_warmup_count = 1 if reset_warmup else state.warmup_count + 1
        confident = (
            can_compare
            and not frozen
            and field is not None
            and next_warmup_count > self._settings.detection_motion_warmup_frames
        )
        results: list[DetectedObject] = []
        bbox_statuses = {
            measurement.current_index: measurement.status
            for measurement in (bbox_analysis.measurements if bbox_analysis else ())
        }
        hybrid_statuses = {
            measurement.current_index: measurement.final_result
            for measurement in (hybrid_analysis.measurements if hybrid_analysis else ())
        }
        local_statuses = {
            measurement.vehicle_index: measurement.final_result
            for measurement in (local_analysis.measurements if local_analysis else ())
        }
        adaptive_statuses = {
            measurement.current_index: measurement.status
            for measurement in (adaptive_analysis.measurements if adaptive_analysis else ())
        }
        for index, item in enumerate(detections):
            status = MotionStatus.UNKNOWN
            if confident and item.cls is ObjectClass.TASIT:
                if isinstance(self._motion_analyzer, HomographyAdaptiveMotionAnalyzer):
                    status = adaptive_statuses.get(index, MotionStatus.UNKNOWN)
                elif isinstance(self._motion_analyzer, HomographyLocalMotionAnalyzer):
                    status = local_statuses.get(index, MotionStatus.UNKNOWN)
                elif isinstance(self._motion_analyzer, HomographyHybridMotionAnalyzer):
                    status = hybrid_statuses.get(index, MotionStatus.UNKNOWN)
                elif isinstance(self._motion_analyzer, HomographyBBoxMotionAnalyzer):
                    status = bbox_statuses.get(index, MotionStatus.UNKNOWN)
                else:
                    status = self._motion_analyzer.classify_vehicle(
                        field,
                        (
                            item.top_left_x,
                            item.top_left_y,
                            item.bottom_right_x,
                            item.bottom_right_y,
                        ),
                    )
            results.append(item.model_copy(update={"motion_status": status}))
        return gray, results, reset_warmup, frozen

    def _convert_box(
        self, box: object, width: int, height: int
    ) -> _DetectionCandidate | None:
        try:
            raw_class = _first(getattr(box, "cls"))
            raw_confidence = float(_first(getattr(box, "conf")))
            raw_bbox = _to_list(getattr(box, "xyxy"))
        except (AttributeError, TypeError, ValueError, OverflowError):
            logger.warning("detection_malformed_yolo_box")
            return None
        object_class = self._class_mapper.resolve(raw_class)
        if object_class is None:
            return None
        if not math.isfinite(raw_confidence) or not 0.0 <= raw_confidence <= 1.0:
            logger.warning("detection_invalid_confidence")
            return None
        if raw_confidence < self._runtime.confidence:
            return None
        boxes = _validated_bbox(raw_bbox, width, height)
        if boxes is None:
            logger.warning("detection_invalid_bbox")
            return None
        raw, clipped = boxes
        return _DetectionCandidate(
            cls=object_class,
            confidence=raw_confidence,
            raw_bbox=raw,
            clipped_bbox=clipped,
        )
