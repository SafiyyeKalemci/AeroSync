from __future__ import annotations

import threading
import time
from pathlib import Path

from app.core.config import Settings
from app.services.matching.local_features import (
    LocalArtifactUnavailable,
    LocalFeatureError,
    LocalFeatureSet,
    LocalMatchSet,
)


class LightGlueRuntime:
    """Lazy local TorchScript matcher; performs no network or model downloads."""

    def __init__(self, settings: Settings, *, torch_module=None, model_loader=None) -> None:
        self._settings = settings
        self._torch = torch_module
        self._model_loader = model_loader or self._load_model
        self._model = None
        self._device: str | None = None
        self._load_lock = threading.Lock()
        self.inference_lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        self._ensure_loaded()
        assert self._device is not None
        return self._device

    def match(self, reference: LocalFeatureSet, frame: LocalFeatureSet) -> tuple[LocalMatchSet, float]:
        import numpy as np

        self._ensure_loaded()
        torch = self._get_torch()
        features0 = self._torch_features(reference)
        features1 = self._torch_features(frame)
        started = time.perf_counter()
        try:
            with self.inference_lock, torch.inference_mode():
                output = self._model(features0, features1)
        except Exception as exc:
            raise LocalFeatureError("LightGlue forward basarisiz.") from exc
        elapsed = time.perf_counter() - started
        if not isinstance(output, dict):
            raise LocalFeatureError("LightGlue artifact cikti sozlesmesi gecersiz.")
        matches = output.get("matches")
        if matches is None:
            matches0 = output.get("matches0")
            if matches0 is None:
                raise LocalFeatureError("LightGlue match indices eksik.")
            values = self._numpy(matches0).reshape(-1)
            left = np.flatnonzero(values >= 0)
            matches = np.column_stack((left, values[left]))
        else:
            matches = self._numpy(matches)
            if matches.ndim == 3 and matches.shape[0] == 1:
                matches = matches[0]
        matches = np.asarray(matches, dtype=np.int64)
        if matches.size == 0:
            return self._failure("no_local_matches"), elapsed
        if matches.ndim != 2 or matches.shape[1] != 2:
            raise LocalFeatureError("LightGlue match index boyutu gecersiz.")
        scores_value = output.get("scores", output.get("matching_scores0"))
        if scores_value is None:
            scores = np.ones((len(matches),), dtype=np.float32)
        else:
            raw_scores = self._numpy(scores_value)
            if raw_scores.ndim == 2 and raw_scores.shape[0] == 1:
                raw_scores = raw_scores[0]
            raw_scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
            scores = raw_scores[matches[:, 0]] if len(raw_scores) == reference.keypoint_count else raw_scores
        if len(scores) != len(matches) or not np.isfinite(scores).all():
            raise LocalFeatureError("LightGlue score degerleri gecersiz.")
        if (
            (matches < 0).any() or (matches[:, 0] >= reference.keypoint_count).any()
            or (matches[:, 1] >= frame.keypoint_count).any()
        ):
            raise LocalFeatureError("LightGlue match index sinir disinda.")
        if len(set(matches[:, 0])) != len(matches) or len(set(matches[:, 1])) != len(matches):
            raise LocalFeatureError("LightGlue one-to-one esleme sozlesmesini ihlal etti.")
        ref_points = reference.keypoints[matches[:, 0]].astype(np.float32)
        frame_points = frame.keypoints[matches[:, 1]].astype(np.float32)
        return LocalMatchSet(
            ref_points, frame_points, scores.astype(np.float32), len(matches), float(scores.mean())
        ), elapsed

    def _torch_features(self, features: LocalFeatureSet):
        torch = self._get_torch()
        return {
            "keypoints": torch.from_numpy(features.keypoints)[None].to(self.device),
            "descriptors": torch.from_numpy(features.descriptors)[None].to(self.device),
            "scores": torch.from_numpy(features.scores)[None].to(self.device),
            "image_size": torch.tensor([[features.image_width, features.image_height]], device=self.device),
        }

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            path = self._settings.matching_lightglue_model_path
            if path is None or not path.is_file():
                raise LocalArtifactUnavailable("Yerel LightGlue TorchScript artifact bulunamadi.")
            torch = self._get_torch()
            device = self._resolve_device(torch, self._settings.matching_lightglue_device)
            model = self._model_loader(torch, path, device)
            if hasattr(model, "eval"):
                model = model.eval()
            self._model, self._device = model, device

    def _get_torch(self):
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:
                raise LocalArtifactUnavailable("PyTorch kurulu degil.") from exc
            self._torch = torch
        return self._torch

    @staticmethod
    def _resolve_device(torch, requested: str) -> str:
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)

    @staticmethod
    def _load_model(torch, path: Path, device: str):
        return torch.jit.load(str(path), map_location=device)

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else value

    @staticmethod
    def _failure(reason: str) -> LocalMatchSet:
        import numpy as np
        return LocalMatchSet(
            np.empty((0, 2), np.float32), np.empty((0, 2), np.float32),
            np.empty((0,), np.float32), 0, 0.0, reason
        )


class LightGlueRuntimeRegistry:
    _lock = threading.Lock()
    _instances: dict[tuple[str, str], LightGlueRuntime] = {}

    @classmethod
    def get(cls, settings: Settings) -> LightGlueRuntime:
        key = (str(settings.matching_lightglue_model_path), settings.matching_lightglue_device)
        with cls._lock:
            return cls._instances.setdefault(key, LightGlueRuntime(settings))
