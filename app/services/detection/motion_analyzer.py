from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from app.schemas import MotionStatus

logger = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]
FlowCalculator = Callable[[object, object], object]


def _farneback(previous_gray: object, current_gray: object) -> object:
    import cv2

    return cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )


@dataclass(frozen=True)
class MotionField:
    flow: object
    global_x: float
    global_y: float
    scale_x: float
    scale_y: float
    valid_pixel_count: int


class MotionAnalyzer:
    """Safe Farnebäck/global-median baseline; not rotation/perspective compensation."""

    def __init__(
        self,
        *,
        threshold_px: float,
        min_valid_pixels: int,
        inner_crop_ratio: float,
        flow_downscale: float,
        freeze_threshold: float,
        flow_calculator: FlowCalculator = _farneback,
    ) -> None:
        self.threshold_px = threshold_px
        self.min_valid_pixels = min_valid_pixels
        self.inner_crop_ratio = inner_crop_ratio
        self.flow_downscale = flow_downscale
        self.freeze_threshold = freeze_threshold
        self._flow_calculator = flow_calculator

    def to_grayscale(self, image: object) -> object:
        import cv2

        shape = getattr(image, "shape", ())
        if len(shape) == 2:
            return image.copy()
        if len(shape) == 3 and shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(shape) == 3 and shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        raise ValueError("Motion analizi için desteklenmeyen görüntü şekli")

    def is_frozen(self, previous_gray: object, current_gray: object) -> bool:
        import numpy as np

        previous = np.asarray(previous_gray)
        current = np.asarray(current_gray)
        if previous.shape != current.shape or previous.size == 0:
            return False
        difference = float(np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))))
        return math.isfinite(difference) and difference <= self.freeze_threshold

    def compute_flow(
        self,
        previous_gray: object,
        current_gray: object,
        exclusion_boxes: Iterable[BBox],
    ) -> MotionField | None:
        import cv2
        import numpy as np

        previous = np.asarray(previous_gray)
        current = np.asarray(current_gray)
        if previous.shape != current.shape or previous.ndim != 2 or previous.size == 0:
            return None
        original_height, original_width = current.shape
        if self.flow_downscale < 1.0:
            target_width = max(1, int(round(original_width * self.flow_downscale)))
            target_height = max(1, int(round(original_height * self.flow_downscale)))
            size = (target_width, target_height)
            previous = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
            current = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
        flow = np.asarray(self._flow_calculator(previous, current))
        if flow.shape != (current.shape[0], current.shape[1], 2):
            logger.warning("motion_insufficient_flow", extra={"reason": "invalid_shape"})
            return None
        scale_x = flow.shape[1] / original_width
        scale_y = flow.shape[0] / original_height
        valid = np.isfinite(flow).all(axis=2)
        for x1, y1, x2, y2 in exclusion_boxes:
            left = max(0, min(flow.shape[1], int(math.floor(x1 * scale_x))))
            top = max(0, min(flow.shape[0], int(math.floor(y1 * scale_y))))
            right = max(0, min(flow.shape[1], int(math.ceil(x2 * scale_x))))
            bottom = max(0, min(flow.shape[0], int(math.ceil(y2 * scale_y))))
            if right > left and bottom > top:
                valid[top:bottom, left:right] = False
        count = int(valid.sum())
        if count < self.min_valid_pixels:
            logger.warning("motion_insufficient_flow", extra={"valid_pixel_count": count})
            return None
        global_x = float(np.median(flow[:, :, 0][valid]))
        global_y = float(np.median(flow[:, :, 1][valid]))
        if not math.isfinite(global_x) or not math.isfinite(global_y):
            logger.warning("motion_insufficient_flow", extra={"reason": "non_finite_global"})
            return None
        return MotionField(flow, global_x, global_y, scale_x, scale_y, count)

    def classify_vehicle(self, field: MotionField | None, bbox: BBox) -> MotionStatus:
        import numpy as np

        if field is None:
            return MotionStatus.UNKNOWN
        flow = np.asarray(field.flow)
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        x1 += width * self.inner_crop_ratio
        x2 -= width * self.inner_crop_ratio
        y1 += height * self.inner_crop_ratio
        y2 -= height * self.inner_crop_ratio
        left = max(0, min(flow.shape[1], int(math.floor(x1 * field.scale_x))))
        top = max(0, min(flow.shape[0], int(math.floor(y1 * field.scale_y))))
        right = max(0, min(flow.shape[1], int(math.ceil(x2 * field.scale_x))))
        bottom = max(0, min(flow.shape[0], int(math.ceil(y2 * field.scale_y))))
        if right <= left or bottom <= top:
            return MotionStatus.UNKNOWN
        roi = flow[top:bottom, left:right]
        valid = np.isfinite(roi).all(axis=2)
        count = int(valid.sum())
        if count < self.min_valid_pixels:
            logger.warning("motion_insufficient_flow", extra={"valid_pixel_count": count})
            return MotionStatus.UNKNOWN
        residual_x = float(np.median(roi[:, :, 0][valid] - field.global_x)) / field.scale_x
        residual_y = float(np.median(roi[:, :, 1][valid] - field.global_y)) / field.scale_y
        magnitude = math.hypot(residual_x, residual_y)
        if not math.isfinite(magnitude):
            return MotionStatus.UNKNOWN
        return MotionStatus.MOVING if magnitude > self.threshold_px else MotionStatus.STATIONARY
