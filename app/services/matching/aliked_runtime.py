from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

from app.core.config import Settings
from app.services.matching.local_features import (
    LocalArtifactUnavailable,
    LocalFeatureError,
    LocalFeatureMetrics,
    LocalFeatureSet,
)


class AlikedRuntime:
    """Lazy, local-only ALIKED TorchScript extractor.

    Artifact contract: RGB float tensor ``[1,3,H,W]`` -> mapping containing
    ``keypoints`` (``[1,N,2]``), ``descriptors`` (``[1,N,D]``), and optional
    ``scores`` (``[1,N]``). Coordinates are interpreted in resized pixels and
    mapped back to the original image.
    """

    def __init__(self, settings: Settings, *, torch_module=None, model_loader=None) -> None:
        self._settings = settings
        self._torch = torch_module
        self._model_loader = model_loader or self._load_model
        self._model = None
        self._model_hash: str | None = None
        self._device: str | None = None
        self._load_lock = threading.Lock()
        self.inference_lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_hash(self) -> str:
        self._ensure_loaded()
        assert self._model_hash is not None
        return self._model_hash

    @property
    def device(self) -> str:
        self._ensure_loaded()
        assert self._device is not None
        return self._device

    def extract(self, image, source_hash: str) -> tuple[LocalFeatureSet, LocalFeatureMetrics]:
        import cv2
        import numpy as np

        self._ensure_loaded()
        torch = self._get_torch()
        height, width = image.shape[:2]
        max_edge = self._settings.matching_dinov2_max_long_edge
        scale = min(1.0, max_edge / max(width, height))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        started = time.perf_counter()
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None]
            .float().to(self.device).div(255.0)
        )
        preprocessing = time.perf_counter() - started
        started = time.perf_counter()
        try:
            with self.inference_lock, torch.inference_mode():
                output = self._model(tensor)
        except Exception as exc:
            raise LocalFeatureError("ALIKED forward basarisiz.") from exc
        forward = time.perf_counter() - started
        if not isinstance(output, dict) or "keypoints" not in output or "descriptors" not in output:
            raise LocalFeatureError("ALIKED artifact cikti sozlesmesi gecersiz.")
        keypoints = self._numpy(output["keypoints"])
        descriptors = self._numpy(output["descriptors"])
        scores = self._numpy(output.get("scores")) if output.get("scores") is not None else None
        if keypoints.ndim == 3 and keypoints.shape[0] == 1:
            keypoints = keypoints[0]
        if descriptors.ndim == 3 and descriptors.shape[0] == 1:
            descriptors = descriptors[0]
        if scores is not None and scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]
        if keypoints.ndim != 2 or keypoints.shape[1] != 2 or descriptors.ndim != 2:
            raise LocalFeatureError("ALIKED feature boyutlari gecersiz.")
        if len(keypoints) != len(descriptors):
            raise LocalFeatureError("ALIKED keypoint/descriptor sayisi uyusmuyor.")
        if scores is None:
            scores = np.ones((len(keypoints),), dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        keypoints = np.asarray(keypoints, dtype=np.float32)
        descriptors = np.asarray(descriptors, dtype=np.float32)
        if len(scores) != len(keypoints) or not all(
            np.isfinite(value).all() for value in (keypoints, descriptors, scores)
        ):
            raise LocalFeatureError("ALIKED feature degerleri gecersiz.")
        keypoints[:, 0] /= resized_width / width
        keypoints[:, 1] /= resized_height / height
        if len(keypoints) and (
            (keypoints[:, 0] < 0).any() or (keypoints[:, 0] > width).any()
            or (keypoints[:, 1] < 0).any() or (keypoints[:, 1] > height).any()
        ):
            raise LocalFeatureError("ALIKED keypoint goruntu siniri disinda.")
        features = LocalFeatureSet(
            keypoints, descriptors, scores, width, height, self.device, source_hash
        )
        metrics = LocalFeatureMetrics(
            preprocessing, forward, len(keypoints), descriptors.shape[1], (height, width), self.device
        )
        return features, metrics

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            path = self._settings.matching_aliked_model_path
            if path is None or not path.is_file():
                raise LocalArtifactUnavailable("Yerel ALIKED TorchScript artifact bulunamadi.")
            torch = self._get_torch()
            device = self._resolve_device(torch, self._settings.matching_aliked_device)
            model = self._model_loader(torch, path, device)
            if hasattr(model, "eval"):
                model = model.eval()
            self._model, self._device = model, device
            self._model_hash = self._sha256(path)

    def _get_torch(self):
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:
                raise LocalArtifactUnavailable("PyTorch kurulu degil.") from exc
            self._torch = torch
        return self._torch

    def _numpy(self, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return value

    @staticmethod
    def _resolve_device(torch, requested: str) -> str:
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)

    @staticmethod
    def _load_model(torch, path: Path, device: str):
        # The exported ALIKED graph contains torchvision's deformable-convolution
        # custom operator. Importing torchvision registers that operator before
        # TorchScript deserializes the otherwise self-contained local artifact.
        try:
            import torchvision  # noqa: F401
        except ImportError as exc:
            raise LocalArtifactUnavailable("Torchvision kurulu degil.") from exc
        return torch.jit.load(str(path), map_location=device)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


class AlikedRuntimeRegistry:
    _lock = threading.Lock()
    _instances: dict[tuple[str, str], AlikedRuntime] = {}

    @classmethod
    def get(cls, settings: Settings) -> AlikedRuntime:
        key = (str(settings.matching_aliked_model_path), settings.matching_aliked_device)
        with cls._lock:
            return cls._instances.setdefault(key, AlikedRuntime(settings))
