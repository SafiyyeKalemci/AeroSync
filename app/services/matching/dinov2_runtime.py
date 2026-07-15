from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from app.core.config import Settings
from app.services.matching.descriptor_types import (
    DenseDescriptorSet,
    DescriptorMetrics,
    PreprocessedImage,
)

logger = logging.getLogger(__name__)


class Dinov2ConfigurationError(RuntimeError):
    pass


class Dinov2DescriptorError(RuntimeError):
    pass


class Dinov2CudaOutOfMemory(Dinov2DescriptorError):
    pass


ModelLoader = Callable[[object, Path, Path, str, str], tuple[object, str]]


class Dinov2DescriptorRuntime:
    """Thread-safe lazy local-only DINOv2 dense descriptor runtime."""

    def __init__(
        self,
        settings: Settings,
        *,
        torch_module: object | None = None,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self._settings = settings
        self._torch = torch_module
        self._model_loader = model_loader or self._load_local_model
        self._load_lock = threading.Lock()
        self.inference_lock = threading.RLock()
        self._model: object | None = None
        self._model_hash: str | None = None
        self._device: str | None = None
        self._load_error: Exception | None = None
        self._validate_configuration_values()

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

    def extract(self, image, source_hash: str) -> tuple[DenseDescriptorSet, DescriptorMetrics]:
        self._ensure_loaded()
        assert self._model is not None and self._device is not None
        torch = self._get_torch()

        preprocess_started = time.perf_counter()
        prepared = self.preprocess(image)
        preprocessing_seconds = time.perf_counter() - preprocess_started
        forward_started = time.perf_counter()
        try:
            with self.inference_lock, torch.inference_mode():
                output = self._model.forward_features(prepared.tensor)
        except Exception as exc:
            if "out of memory" in str(exc).lower() and "cuda" in str(exc).lower():
                logger.error("matching_dinov2_cuda_oom", extra={"event": "matching_dinov2_cuda_oom"})
                raise Dinov2CudaOutOfMemory(str(exc)) from exc
            raise Dinov2DescriptorError("DINOv2 forward basarisiz.") from exc
        forward_seconds = time.perf_counter() - forward_started
        if forward_seconds > self._settings.matching_dinov2_timeout_seconds:
            logger.warning(
                "matching_dinov2_inference_lock_slow",
                extra={"event": "matching_dinov2_inference_lock_slow",
                       "forward_seconds": forward_seconds},
            )

        descriptors = self._extract_patch_tokens(output, prepared)
        descriptor_dtype = self._descriptor_dtype(torch)
        descriptors = descriptors.to(dtype=descriptor_dtype)
        if self._settings.matching_dinov2_normalize_descriptors:
            descriptors = torch.nn.functional.normalize(descriptors.float(), dim=-1)
            descriptors = descriptors.to(dtype=descriptor_dtype)
        if not bool(torch.isfinite(descriptors).all()):
            logger.error(
                "matching_dinov2_descriptor_invalid",
                extra={"event": "matching_dinov2_descriptor_invalid", "reason": "non_finite"},
            )
            raise Dinov2DescriptorError("Descriptor NaN veya Inf iceriyor.")

        cache_device = self._cache_device()
        descriptors = descriptors.detach().to(cache_device).contiguous()
        descriptor_set = DenseDescriptorSet(
            descriptors=descriptors,
            grid_width=prepared.grid_width,
            grid_height=prepared.grid_height,
            descriptor_dim=int(descriptors.shape[1]),
            image_width=prepared.original_width,
            image_height=prepared.original_height,
            resized_width=prepared.resized_width,
            resized_height=prepared.resized_height,
            patch_size=prepared.patch_size,
            scale_x=prepared.scale_x,
            scale_y=prepared.scale_y,
            device=str(descriptors.device),
            dtype=str(descriptors.dtype).removeprefix("torch."),
            source_hash=source_hash,
        )
        metrics = DescriptorMetrics(
            preprocessing_seconds=preprocessing_seconds,
            forward_seconds=forward_seconds,
            descriptor_count=descriptor_set.shape[0],
            descriptor_dimension=descriptor_set.descriptor_dim,
            descriptor_bytes=descriptor_set.nbytes,
        )
        return descriptor_set, metrics

    def preprocess(self, image) -> PreprocessedImage:
        import cv2
        import numpy as np

        torch = self._get_torch()
        if getattr(image, "ndim", 0) != 3 or image.shape[2] < 3:
            raise Dinov2DescriptorError("RGB/BGR uc kanalli goruntu bekleniyor.")
        original_height, original_width = (int(image.shape[0]), int(image.shape[1]))
        patch = self._settings.matching_dinov2_patch_size
        if original_width < patch or original_height < patch:
            raise Dinov2DescriptorError("Goruntu patch boyutundan kucuk.")
        scale = min(
            1.0,
            self._settings.matching_dinov2_max_long_edge
            / float(max(original_width, original_height)),
        )
        scaled_width = max(patch, int(original_width * scale))
        scaled_height = max(patch, int(original_height * scale))
        resized_width = max(patch, (scaled_width // patch) * patch)
        resized_height = max(patch, (scaled_height // patch) * patch)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        return PreprocessedImage(
            tensor=tensor,
            original_width=original_width,
            original_height=original_height,
            resized_width=resized_width,
            resized_height=resized_height,
            grid_width=resized_width // patch,
            grid_height=resized_height // patch,
            scale_x=resized_width / original_width,
            scale_y=resized_height / original_height,
            patch_size=patch,
        )

    def _extract_patch_tokens(self, output, prepared: PreprocessedImage):
        if not isinstance(output, dict) or "x_norm_patchtokens" not in output:
            raise Dinov2DescriptorError("Model x_norm_patchtokens dondurmedi.")
        descriptors = output["x_norm_patchtokens"]
        if getattr(descriptors, "ndim", 0) == 3 and descriptors.shape[0] == 1:
            descriptors = descriptors.squeeze(0)
        expected_count = prepared.grid_width * prepared.grid_height
        if getattr(descriptors, "ndim", 0) != 2:
            raise Dinov2DescriptorError("Descriptor shape [N,D] olmali.")
        if int(descriptors.shape[0]) != expected_count or int(descriptors.shape[1]) <= 0:
            raise Dinov2DescriptorError(
                f"Descriptor shape gecersiz: {tuple(descriptors.shape)}, beklenen N={expected_count}."
            )
        return descriptors

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._load_error is not None:
            raise Dinov2ConfigurationError(str(self._load_error)) from self._load_error
        with self._load_lock:
            if self._model is not None:
                return
            if self._load_error is not None:
                raise Dinov2ConfigurationError(str(self._load_error)) from self._load_error
            started = time.perf_counter()
            logger.info(
                "matching_dinov2_runtime_loading",
                extra={"event": "matching_dinov2_runtime_loading"},
            )
            try:
                torch = self._get_torch()
                repo = self._required_repo()
                weights = self._required_weights()
                device = self._select_device(torch)
                model, model_hash = self._model_loader(
                    torch,
                    repo,
                    weights,
                    self._settings.matching_dinov2_model_name,
                    device,
                )
                self._model = model
                self._model_hash = model_hash
                self._device = device
            except Exception as exc:
                self._load_error = exc
                logger.error(
                    "matching_dinov2_runtime_failed",
                    extra={"event": "matching_dinov2_runtime_failed", "reason": type(exc).__name__},
                    exc_info=True,
                )
                raise Dinov2ConfigurationError(str(exc)) from exc
            logger.info(
                "matching_dinov2_runtime_ready",
                extra={
                    "event": "matching_dinov2_runtime_ready",
                    "device": self._device,
                    "model_hash": self._model_hash[:12],
                    "load_seconds": time.perf_counter() - started,
                },
            )

    def _get_torch(self):
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:
                raise Dinov2ConfigurationError("PyTorch kurulu degil.") from exc
            self._torch = torch
        return self._torch

    def _required_repo(self) -> Path:
        path = self._settings.matching_dinov2_repo_path
        if path is None or not path.is_dir():
            raise Dinov2ConfigurationError("MATCHING_DINOV2_REPO_PATH mevcut yerel klasor olmali.")
        if not (path / "hubconf.py").is_file():
            raise Dinov2ConfigurationError("Yerel DINOv2 repository hubconf.py icermiyor.")
        return path

    def _required_weights(self) -> Path:
        path = self._settings.matching_dinov2_weights_path
        if path is None or not path.is_file():
            raise Dinov2ConfigurationError("MATCHING_DINOV2_WEIGHTS_PATH mevcut yerel dosya olmali.")
        return path

    def _select_device(self, torch) -> str:
        requested = self._settings.matching_dinov2_device.lower()
        cuda_available = bool(torch.cuda.is_available())
        if requested == "auto":
            if cuda_available:
                return "cuda"
            if not self._settings.matching_dinov2_allow_cpu_fallback:
                raise Dinov2ConfigurationError("CUDA yok ve CPU fallback kapali.")
            logger.warning(
                "matching_dinov2_cpu_fallback",
                extra={"event": "matching_dinov2_cpu_fallback"},
            )
            return "cpu"
        if requested == "cuda" and not cuda_available:
            if not self._settings.matching_dinov2_allow_cpu_fallback:
                raise Dinov2ConfigurationError("MATCHING_DINOV2_DEVICE=cuda ancak CUDA yok.")
            logger.warning(
                "matching_dinov2_cpu_fallback",
                extra={"event": "matching_dinov2_cpu_fallback"},
            )
            return "cpu"
        if requested not in {"cpu", "cuda"}:
            raise Dinov2ConfigurationError(f"Gecersiz DINOv2 device: {requested}")
        return requested

    def _descriptor_dtype(self, torch):
        configured = self._settings.matching_dinov2_descriptor_dtype
        if configured == "float16" and self.device == "cpu":
            logger.warning(
                "matching_dinov2_cpu_float16_fallback",
                extra={"event": "matching_dinov2_cpu_float16_fallback"},
            )
            return torch.float32
        return torch.float16 if configured == "float16" else torch.float32

    def _cache_device(self) -> str:
        configured = self._settings.matching_dinov2_cache_device
        if configured == "cuda" and self.device != "cuda":
            return "cpu"
        return configured

    def _validate_configuration_values(self) -> None:
        if self._settings.matching_dinov2_max_long_edge < 1:
            raise ValueError("MATCHING_DINOV2_MAX_LONG_EDGE pozitif olmali.")
        if self._settings.matching_dinov2_patch_size < 1:
            raise ValueError("MATCHING_DINOV2_PATCH_SIZE pozitif olmali.")
        if self._settings.matching_dinov2_descriptor_dtype not in {"float16", "float32"}:
            raise ValueError("MATCHING_DINOV2_DESCRIPTOR_DTYPE float16 veya float32 olmali.")
        if self._settings.matching_dinov2_cache_device not in {"cpu", "cuda"}:
            raise ValueError("MATCHING_DINOV2_CACHE_DEVICE cpu veya cuda olmali.")
        if self._settings.matching_dinov2_max_cached_references < 1:
            raise ValueError("MATCHING_DINOV2_MAX_CACHED_REFERENCES en az 1 olmali.")
        if self._settings.matching_dinov2_timeout_seconds <= 0:
            raise ValueError("MATCHING_DINOV2_TIMEOUT_SECONDS pozitif olmali.")
        if self._settings.matching_dinov2_max_cache_mb <= 0:
            raise ValueError("MATCHING_DINOV2_MAX_CACHE_MB pozitif olmali.")

    @staticmethod
    def _load_local_model(
        torch,
        repo_path: Path,
        weights_path: Path,
        model_name: str,
        device: str,
    ) -> tuple[object, str]:
        digest = hashlib.sha256()
        with weights_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        model_hash = digest.hexdigest()
        model = torch.hub.load(
            str(repo_path),
            model_name,
            source="local",
            pretrained=False,
            verbose=False,
        )
        try:
            checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        except TypeError:
            logger.warning(
                "matching_dinov2_weights_only_unsupported",
                extra={"event": "matching_dinov2_weights_only_unsupported"},
            )
            checkpoint = torch.load(str(weights_path), map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise Dinov2ConfigurationError("DINOv2 checkpoint state-dict degil.")
        clean_state = {str(key).removeprefix("module."): value for key, value in state.items()}
        incompatible = model.load_state_dict(clean_state, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing or unexpected:
            logger.error(
                "matching_dinov2_state_dict_incompatible",
                extra={
                    "event": "matching_dinov2_state_dict_incompatible",
                    "missing_key_count": len(missing),
                    "unexpected_key_count": len(unexpected),
                },
            )
            raise Dinov2ConfigurationError("DINOv2 state-dict model ile uyumsuz.")
        return model.eval().to(device), model_hash


class Dinov2RuntimeRegistry:
    _lock = threading.Lock()
    _runtimes: dict[tuple[object, ...], Dinov2DescriptorRuntime] = {}

    @classmethod
    def get(cls, settings: Settings) -> Dinov2DescriptorRuntime:
        key = (
            settings.matching_dinov2_repo_path,
            settings.matching_dinov2_weights_path,
            settings.matching_dinov2_model_name,
            settings.matching_dinov2_device,
            settings.matching_dinov2_allow_cpu_fallback,
            settings.matching_dinov2_max_long_edge,
            settings.matching_dinov2_patch_size,
            settings.matching_dinov2_descriptor_dtype,
            settings.matching_dinov2_normalize_descriptors,
            settings.matching_dinov2_cache_device,
        )
        with cls._lock:
            runtime = cls._runtimes.get(key)
            if runtime is None:
                runtime = Dinov2DescriptorRuntime(settings)
                cls._runtimes[key] = runtime
            return runtime

    @classmethod
    def clear_for_tests(cls) -> None:
        with cls._lock:
            cls._runtimes.clear()
