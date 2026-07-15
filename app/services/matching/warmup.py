from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass

from app.core.config import Settings
from app.services.matching.aliked_runtime import AlikedRuntimeRegistry
from app.services.matching.dinov2_runtime import Dinov2RuntimeRegistry
from app.services.matching.lightglue_runtime import LightGlueRuntimeRegistry


@dataclass(frozen=True, slots=True)
class MatchingWarmupDiagnostics:
    model_preloaded: bool = False
    warmup_completed: bool = False
    warmup_time_sec: float = 0.0


class MatchingWarmupService:
    """Thread-safe, local-only model preload and disposable dummy inference."""

    def __init__(
        self,
        settings: Settings,
        *,
        dinov2_factory=Dinov2RuntimeRegistry.get,
        aliked_factory=AlikedRuntimeRegistry.get,
        lightglue_factory=LightGlueRuntimeRegistry.get,
        clock=time.perf_counter,
    ) -> None:
        self._settings = settings
        self._dinov2_factory = dinov2_factory
        self._aliked_factory = aliked_factory
        self._lightglue_factory = lightglue_factory
        self._clock = clock
        self._lock = threading.Lock()
        self._completed = False
        self._diagnostics = MatchingWarmupDiagnostics()

    @property
    def diagnostics(self) -> MatchingWarmupDiagnostics:
        return self._diagnostics

    def warmup(self) -> MatchingWarmupDiagnostics:
        if self._completed:
            return self._diagnostics
        with self._lock:
            if self._completed:
                return self._diagnostics
            started = self._clock()
            import numpy as np

            edge = max(224, self._settings.matching_dinov2_patch_size * 4)
            # A flat image can legitimately yield zero ALIKED keypoints, which is
            # not a valid LightGlue warmup input. Deterministic texture keeps this
            # disposable input small while exercising the real feature contract.
            dummy = np.random.default_rng(0).integers(
                0, 256, size=(edge, edge, 3), dtype=np.uint8
            )
            dummy_hash = hashlib.sha256(dummy.tobytes()).hexdigest()
            dinov2 = self._dinov2_factory(self._settings)
            aliked = self._aliked_factory(self._settings)
            lightglue = self._lightglue_factory(self._settings)

            if self._settings.matching_preload_models:
                _ = dinov2.model_hash
                _ = aliked.model_hash
                _ = lightglue.device

            if self._settings.matching_warmup_enabled:
                dinov2.extract(dummy, f"warmup-dinov2-{dummy_hash}")
                local_features, _ = aliked.extract(dummy, f"warmup-aliked-{dummy_hash}")
                lightglue.match(local_features, local_features)

            elapsed = max(0.0, self._clock() - started)
            self._diagnostics = MatchingWarmupDiagnostics(
                model_preloaded=(
                    self._settings.matching_preload_models
                    or self._settings.matching_warmup_enabled
                ),
                warmup_completed=self._settings.matching_warmup_enabled,
                warmup_time_sec=elapsed,
            )
            self._completed = True
            return self._diagnostics
