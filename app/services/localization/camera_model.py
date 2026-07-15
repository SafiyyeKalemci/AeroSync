from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]

    def valid(self) -> bool:
        values = (self.fx, self.fy, self.cx, self.cy, *self.distortion)
        return (
            self.width > 0
            and self.height > 0
            and self.fx > 0
            and self.fy > 0
            and all(math.isfinite(value) for value in values)
            and 0 <= self.cx <= self.width
            and 0 <= self.cy <= self.height
        )

    def undistort(self, gray):
        if not self.distortion or not any(self.distortion):
            return gray
        import cv2
        import numpy as np

        matrix = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return cv2.undistort(gray, matrix, np.asarray(self.distortion, dtype=np.float64))


def parse_distortion(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("LOCALIZATION_CAMERA_DISTORTION must contain comma-separated numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError("LOCALIZATION_CAMERA_DISTORTION must contain finite numbers")
    return result


def _from_mapping(data: dict[str, object]) -> CameraModel:
    matrix = data.get("camera_matrix") or data.get("cameraMatrix")
    if matrix is not None:
        rows = matrix  # type: ignore[assignment]
        fx = float(rows[0][0])  # type: ignore[index]
        fy = float(rows[1][1])  # type: ignore[index]
        cx = float(rows[0][2])  # type: ignore[index]
        cy = float(rows[1][2])  # type: ignore[index]
    else:
        fx, fy, cx, cy = (float(data[name]) for name in ("fx", "fy", "cx", "cy"))
    distortion = data.get("distortion_coefficients", data.get("distortion", []))
    return CameraModel(
        width=int(data.get("width", data.get("image_width", 0))),
        height=int(data.get("height", data.get("image_height", 0))),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion=tuple(float(item) for item in distortion),  # type: ignore[arg-type]
    )


def load_camera_calibration(path: Path) -> CameraModel:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        model = _from_mapping(json.loads(text))
    else:
        values: dict[str, float] = {}
        for name in ("fx", "fy", "cx", "cy", "width", "height"):
            match = re.search(rf"(?im)\b{name}\b\s*[:=]\s*([-+0-9.eE]+)", text)
            if match:
                values[name] = float(match.group(1))
        size = re.search(r"(?im)ImageSize\s*[:=]\s*\[\s*(\d+)\s+[, ]\s*(\d+)\s*\]", text)
        if size and ("width" not in values or "height" not in values):
            values["height"], values["width"] = float(size.group(1)), float(size.group(2))
        distortion_match = re.search(r"(?im)distortion(?:_coefficients)?\s*[:=]\s*\[([^\]]*)\]", text)
        distortion = tuple(
            float(item) for item in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", distortion_match.group(1))
        ) if distortion_match else ()
        model = CameraModel(
            width=int(values["width"]), height=int(values["height"]),
            fx=values["fx"], fy=values["fy"], cx=values["cx"], cy=values["cy"],
            distortion=distortion,
        )
    if not model.valid():
        raise ValueError("Camera calibration is incomplete or invalid")
    return model


class CameraModelProvider:
    def __init__(self, settings: Settings) -> None:
        self._model: CameraModel | None = None
        path = settings.localization_camera_calibration_path
        if path is not None:
            try:
                self._model = load_camera_calibration(path)
            except Exception:
                logger.warning("localization_camera_calibration_invalid", exc_info=True)
        if self._model is None:
            candidate = CameraModel(
                width=settings.localization_camera_width,
                height=settings.localization_camera_height,
                fx=settings.localization_camera_fx,
                fy=settings.localization_camera_fy,
                cx=settings.localization_camera_cx,
                cy=settings.localization_camera_cy,
                distortion=parse_distortion(settings.localization_camera_distortion),
            )
            if candidate.valid():
                self._model = candidate
            else:
                logger.warning("localization_camera_model_missing")

    def for_resolution(self, width: int, height: int) -> CameraModel | None:
        if self._model is None:
            return None
        if (width, height) != (self._model.width, self._model.height):
            logger.warning(
                "localization_camera_resolution_unknown",
                extra={"width": width, "height": height},
            )
            return None
        return self._model
