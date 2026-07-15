from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.services.matching.lightglue_adapter import LocalAlikedLightGlueAdapter
from app.services.matching.xoftr_adapter import LocalXoFTRAdapter

logger = logging.getLogger(__name__)


class MatchingModelConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MatchingModelRuntime:
    torch: object
    device: str
    dinov2: object
    lightglue: object | None
    xoftr: object | None
    inference_lock: threading.RLock


class MatchingModelRuntimeRegistry:
    """Thread-safe lazy singleton registry keyed by local artifact configuration."""

    _lock = threading.Lock()
    _runtimes: dict[tuple[object, ...], MatchingModelRuntime] = {}

    @classmethod
    def get(cls, settings: Settings) -> MatchingModelRuntime:
        key = (
            settings.matching_device,
            settings.matching_allow_cpu_fallback,
            settings.matching_dinov2_repo_path,
            settings.matching_dinov2_weights_path,
            settings.matching_dinov2_model_name,
            settings.matching_aliked_weights_path,
            settings.matching_lightglue_weights_path,
            settings.matching_xoftr_model_path,
        )
        with cls._lock:
            runtime = cls._runtimes.get(key)
            if runtime is None:
                runtime = cls._build(settings)
                cls._runtimes[key] = runtime
            return runtime

    @classmethod
    def clear_for_tests(cls) -> None:
        with cls._lock:
            cls._runtimes.clear()

    @classmethod
    def _build(cls, settings: Settings) -> MatchingModelRuntime:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
            import torch
        except ImportError as exc:
            raise MatchingModelConfigurationError(
                "Görev 3 bağımlılıkları kurulu değil; `pip install .[matching]` gerekli."
            ) from exc

        device = cls._select_device(torch, settings)
        model = cls._load_dinov2(
            torch,
            settings.matching_dinov2_repo_path,
            settings.matching_dinov2_weights_path,
            settings.matching_dinov2_model_name,
            device,
        )
        lightglue = None
        if settings.matching_aliked_weights_path and settings.matching_lightglue_weights_path:
            try:
                lightglue = LocalAlikedLightGlueAdapter(
                    settings.matching_aliked_weights_path,
                    settings.matching_lightglue_weights_path,
                    device,
                )
            except Exception:
                logger.exception("matching_lightglue_load_failed")
        xoftr = None
        if settings.matching_xoftr_model_path:
            try:
                xoftr = LocalXoFTRAdapter(settings.matching_xoftr_model_path, device)
            except Exception:
                logger.exception("matching_xoftr_load_failed")
        return MatchingModelRuntime(
            torch=torch,
            device=device,
            dinov2=model,
            lightglue=lightglue,
            xoftr=xoftr,
            inference_lock=threading.RLock(),
        )

    @staticmethod
    def _select_device(torch, settings: Settings) -> str:
        requested = settings.matching_device.lower()
        cuda_available = bool(torch.cuda.is_available())
        if requested == "auto":
            if cuda_available:
                return "cuda"
            if not settings.matching_allow_cpu_fallback:
                raise MatchingModelConfigurationError(
                    "CUDA kullanılamıyor ve MATCHING_ALLOW_CPU_FALLBACK=false."
                )
            logger.warning("matching_cpu_fallback", extra={"event": "matching_cpu_fallback"})
            return "cpu"
        if requested == "cuda" and not cuda_available:
            if settings.matching_allow_cpu_fallback:
                logger.warning("matching_cpu_fallback", extra={"event": "matching_cpu_fallback"})
                return "cpu"
            raise MatchingModelConfigurationError("MATCHING_DEVICE=cuda fakat CUDA kullanılamıyor.")
        if requested not in {"cpu", "cuda"}:
            raise MatchingModelConfigurationError(f"Geçersiz MATCHING_DEVICE: {requested}")
        return requested

    @staticmethod
    def _load_dinov2(
        torch,
        repo_path: Path | None,
        weights_path: Path | None,
        model_name: str,
        device: str,
    ):
        if repo_path is None or not repo_path.is_dir():
            raise MatchingModelConfigurationError(
                "MATCHING_DINOV2_REPO_PATH yerel DINOv2 kaynak klasörünü göstermeli."
            )
        if weights_path is None or not weights_path.is_file():
            raise MatchingModelConfigurationError(
                "DINOV2_MODEL_PATH yerel DINOv2 ağırlık dosyasını göstermeli."
            )
        model = torch.hub.load(
            str(repo_path),
            model_name,
            source="local",
            pretrained=False,
            verbose=False,
        )
        checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        return model.eval().to(device)
