from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

ModelFactory = Callable[[str], object]


def _default_model_factory(model_path: str) -> object:
    # Ultralytics is intentionally imported only when the first inference needs it.
    from ultralytics import YOLO

    return YOLO(model_path)


class YoloRuntime:
    """Thread-safe lazy loader; one model instance is reused for every frame."""

    def __init__(
        self,
        model_path: Path | None,
        confidence: float,
        iou: float,
        *,
        model_factory: ModelFactory = _default_model_factory,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("DETECTION_CONFIDENCE 0 ile 1 arasında olmalıdır.")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("DETECTION_IOU 0 ile 1 arasında olmalıdır.")
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self._model_factory = model_factory
        self._model: object | None = None
        self._load_attempted = False
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load_once(self) -> object | None:
        if self._load_attempted:
            return self._model
        with self._load_lock:
            if self._load_attempted:
                return self._model
            self._load_attempted = True
            if self.model_path is None:
                logger.error("detection_model_path_not_configured")
                return None
            if not self.model_path.is_file():
                logger.error(
                    "detection_model_file_missing",
                    extra={"model_path": str(self.model_path)},
                )
                return None
            try:
                self._model = self._model_factory(str(self.model_path))
            except Exception:
                logger.error(
                    "detection_model_load_failed",
                    extra={"model_path": str(self.model_path)},
                    exc_info=True,
                )
                self._model = None
            return self._model

    def predict(self, image: object) -> list[object]:
        model = self._load_once()
        if model is None:
            return []
        try:
            with self._inference_lock:
                results = model.predict(
                    source=image,
                    conf=self.confidence,
                    iou=self.iou,
                    verbose=False,
                )
            return list(results or [])
        except Exception:
            logger.error("detection_inference_failed", exc_info=True)
            return []
