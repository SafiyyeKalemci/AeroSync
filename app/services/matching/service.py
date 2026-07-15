from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field

from app.core.config import Settings
from app.schemas import ImageModality, MatchedReferenceObject
from app.services.common import FrameContext
from app.services.matching.interface import (
    ReferenceImage,
    ReferenceMatchingService,
    ReferenceStateInfo,
)
from app.services.matching.pipeline import ModalityDetector
from app.services.matching.coarse_matcher import CoarseMatchingPipeline
from app.services.matching.dinov2_runtime import (
    Dinov2ConfigurationError,
    Dinov2CudaOutOfMemory,
    Dinov2DescriptorError,
    Dinov2DescriptorRuntime,
    Dinov2RuntimeRegistry,
)
from app.services.matching.reference_store import DecodedReference, ReferenceStore
from app.services.matching.reference_state import ReferenceState
from app.services.matching.local_features import (
    LocalArtifactUnavailable,
    LocalFeatureError,
    LocalRefinementDiagnostics,
)
from app.services.matching.local_matcher import (
    LocalRefinementPipeline,
    ReferenceLocalFeatureCache,
)
from app.services.matching.warmup import MatchingWarmupDiagnostics, MatchingWarmupService
from app.utils.images import detect_image_format, read_image_bytes

logger = logging.getLogger(__name__)
Clock = Callable[[], float]
RuntimeFactory = Callable[[Settings], Dinov2DescriptorRuntime]
ImageReader = Callable[[str, float], object]


@dataclass(slots=True)
class _SessionState:
    session_id: str
    reference_store: ReferenceStore
    frame_modality: ImageModality | None = None
    local_feature_cache: ReferenceLocalFeatureCache = field(
        default_factory=ReferenceLocalFeatureCache
    )
    local_diagnostics: dict[int, LocalRefinementDiagnostics] = field(default_factory=dict)
    reference_prepare_epoch: int = 0
    model_preloaded: bool = False
    warmup_completed: bool = False
    warmup_time_sec: float = 0.0
    reference_cache_warmed: bool = False
    reference_prepare_time_sec: float = 0.0


class DinoReferenceMatchingService(ReferenceMatchingService):
    """Stage-2 dense descriptor producer; geometric matching remains disabled."""

    def __init__(
        self,
        settings: Settings,
        *,
        modality_detector: ModalityDetector | None = None,
        clock: Clock = time.monotonic,
        runtime_factory: RuntimeFactory = Dinov2RuntimeRegistry.get,
        image_reader: ImageReader = read_image_bytes,
        matching_pipeline: CoarseMatchingPipeline | None = None,
        local_pipeline: LocalRefinementPipeline | None = None,
        warmup_service: MatchingWarmupService | None = None,
    ) -> None:
        if settings.matching_reference_ttl_seconds <= 0:
            raise ValueError("MATCHING_REFERENCE_TTL_SECONDS pozitif olmalidir.")
        if settings.matching_max_reference_sessions < 1:
            raise ValueError("MATCHING_MAX_REFERENCE_SESSIONS en az 1 olmalidir.")
        if settings.matching_coarse_timeout_seconds <= 0 or settings.matching_reference_timeout_seconds <= 0:
            raise ValueError("Matching timeout degerleri pozitif olmalidir.")
        self._settings = settings
        self._modality_detector = modality_detector or ModalityDetector()
        self._clock = clock
        self._runtime_factory = runtime_factory
        self._image_reader = image_reader
        self._matching_pipeline = matching_pipeline or CoarseMatchingPipeline(settings)
        settings.validate_matching_local()
        self._local_pipeline = local_pipeline
        if local_pipeline is None and settings.matching_geometry_method != "dinov2":
            self._local_pipeline = LocalRefinementPipeline(settings)
        self._warmup_service = warmup_service or MatchingWarmupService(settings)
        self._sessions: dict[str, _SessionState] = {}
        self._last_match_diagnostics: dict[str, dict[int, dict[str, object]]] = {}
        self._registry_lock = asyncio.Lock()

    async def set_references(
        self,
        session_id: str,
        references: list[ReferenceImage],
        frame_modality: ImageModality | None = None,
    ) -> int:
        if not self._settings.matching_enabled:
            logger.info(
                "matching_disabled",
                extra={"event": "matching_disabled", "session_id": session_id},
            )
            return 0
        state = await self._get_session(session_id, create=True)
        assert state is not None
        try:
            loaded = await state.reference_store.replace(references)
        except Exception:
            logger.error(
                "matching_reference_set_rejected",
                extra={
                    "event": "matching_reference_set_rejected",
                    "session_id": session_id,
                    "reference_count": len(references),
                },
                exc_info=True,
            )
            return 0
        state.frame_modality = frame_modality
        state.local_feature_cache.clear()
        state.local_diagnostics.clear()
        state.reference_prepare_epoch += 1
        prepare_epoch = state.reference_prepare_epoch
        await self._prepare_matching_runtime_and_references(state, prepare_epoch)
        logger.info(
            "matching_references_loaded",
            extra={
                "event": "matching_references_loaded",
                "session_id": session_id,
                "reference_count": loaded,
                "descriptor_generation": (
                    "precomputed" if state.reference_cache_warmed else "lazy"
                ),
            },
        )
        return loaded

    async def process_frame(self, frame: FrameContext) -> list[MatchedReferenceObject]:
        if not self._settings.matching_enabled:
            return []
        state = await self._get_session(frame.session_id, create=False)
        if state is None:
            return []
        self._last_match_diagnostics[frame.session_id] = {}
        active = await state.reference_store.active_for_frame(frame.frame_index)
        active = tuple(reference for reference in active if reference.modality is ImageModality.RGB)
        logger.info(
            "matching_active_references_resolved",
            extra={
                "event": "matching_active_references_resolved",
                "session_id": frame.session_id,
                "frame_id": frame.frame_id,
                "frame_index": frame.frame_index,
                "active_reference_count": len(active),
                "descriptor_runtime_enabled": self._settings.matching_dinov2_enabled,
            },
        )
        if not active or not self._settings.matching_dinov2_enabled:
            return []
        if frame.image_modality is ImageModality.THERMAL:
            return []

        runtime = self._runtime_factory(self._settings)
        for reference in active:
            await self._ensure_reference_descriptor(state.reference_store, reference.object_id, runtime)

        try:
            content = await self._image_reader(
                frame.image_url,
                self._settings.matching_dinov2_timeout_seconds,
            )
            detect_image_format(content)
            image = await asyncio.to_thread(self._decode_image, content, "Frame")
            frame_modality = frame.image_modality or self._modality_detector.detect(image)
            if frame_modality is not ImageModality.RGB:
                logger.info(
                    "matching_frame_completed",
                    extra={"event": "matching_frame_completed", "session_id": frame.session_id,
                           "frame_id": frame.frame_id, "result_count": 0,
                           "reason": "rgb_only_stage"},
                )
                return []
            source_hash = hashlib.sha256(content).hexdigest()
            frame_descriptor, metrics = await asyncio.wait_for(
                asyncio.to_thread(runtime.extract, image, source_hash),
                timeout=self._settings.matching_dinov2_timeout_seconds,
            )
            logger.info(
                "matching_dinov2_frame_descriptor_ready",
                extra={
                    "event": "matching_dinov2_frame_descriptor_ready",
                    "session_id": frame.session_id,
                    "frame_id": frame.frame_id,
                    "descriptor_shape": frame_descriptor.shape,
                    "descriptor_bytes": metrics.descriptor_bytes,
                    "forward_seconds": metrics.forward_seconds,
                },
            )
        except asyncio.TimeoutError:
            logger.error(
                "matching_dinov2_timeout",
                extra={"event": "matching_dinov2_timeout", "source": "frame", "session_id": frame.session_id,
                       "frame_id": frame.frame_id},
            )
            return []
        except (Dinov2ConfigurationError, Dinov2DescriptorError, ValueError, OSError):
            logger.error(
                "matching_frame_descriptor_failed",
                extra={"event": "matching_frame_descriptor_failed", "session_id": frame.session_id,
                       "frame_id": frame.frame_id},
                exc_info=True,
            )
            return []

        matching_generation = await state.reference_store.generation_token()
        results: dict[int, MatchedReferenceObject] = {}
        timeout, timeout_stage = self._reference_match_timeout()
        for reference in active:
            current, generation = await state.reference_store.descriptor_snapshot(reference.object_id)
            if generation != matching_generation:
                logger.info(
                    "matching_reference_failed",
                    extra={"event": "matching_reference_failed", "session_id": frame.session_id,
                           "object_id": reference.object_id, "reason": "session_generation_changed"},
                )
                return []
            if current is None or not current.descriptor_ready or current.dense_descriptors is None:
                continue
            matching_started = self._clock()
            try:
                matched = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._match_selected_geometry,
                        runtime,
                        state,
                        reference.object_id,
                        current.dense_descriptors,
                        frame_descriptor,
                        current.image_content,
                        image,
                        source_hash,
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                elapsed = max(0.0, self._clock() - matching_started)
                self._record_match_diagnostic(
                    frame, reference.object_id, timeout_stage, timeout, elapsed, "timeout"
                )
                logger.error(
                    "matching_reference_failed",
                    extra={"event": "matching_reference_failed", "session_id": frame.session_id,
                           "object_id": reference.object_id, "reason": "timeout",
                           "timeout_stage": timeout_stage,
                           "timeout_limit_sec": timeout,
                           "elapsed_sec": elapsed},
                )
                continue
            except MemoryError:
                self._record_match_diagnostic(
                    frame,
                    reference.object_id,
                    timeout_stage,
                    timeout,
                    max(0.0, self._clock() - matching_started),
                    "memory_error",
                )
                logger.error(
                    "matching_reference_failed",
                    extra={"event": "matching_reference_failed", "session_id": frame.session_id,
                           "object_id": reference.object_id, "reason": "memory_error"},
                    exc_info=True,
                )
                continue
            except Exception as exc:
                message = str(exc).lower()
                reason = (
                    "cuda_out_of_memory" if "cuda" in message and "out of memory" in message
                    else "opencv_error" if type(exc).__module__.startswith("cv2")
                    else type(exc).__name__
                )
                self._record_match_diagnostic(
                    frame,
                    reference.object_id,
                    timeout_stage,
                    timeout,
                    max(0.0, self._clock() - matching_started),
                    reason,
                )
                if reason == "cuda_out_of_memory":
                    logger.error(
                        "matching_dinov2_cuda_oom",
                        extra={"event": "matching_dinov2_cuda_oom", "session_id": frame.session_id,
                               "object_id": reference.object_id},
                    )
                logger.error(
                    "matching_reference_failed",
                    extra={"event": "matching_reference_failed", "session_id": frame.session_id,
                           "object_id": reference.object_id, "reason": reason},
                    exc_info=True,
                )
                continue
            self._record_match_diagnostic(
                frame,
                reference.object_id,
                timeout_stage,
                timeout,
                max(0.0, self._clock() - matching_started),
                "matched" if matched is not None else "rejected",
            )
            if await state.reference_store.generation_token() != matching_generation:
                return []
            if matched is not None:
                previous = results.get(reference.object_id)
                if previous is None or (matched.confidence or 0.0) > (previous.confidence or 0.0):
                    results[reference.object_id] = matched
        if await state.reference_store.generation_token() != matching_generation:
            return []
        ordered = [results[reference.object_id] for reference in active if reference.object_id in results]
        logger.info(
            "matching_frame_completed",
            extra={"event": "matching_frame_completed", "session_id": frame.session_id,
                   "frame_id": frame.frame_id, "result_count": len(ordered)},
        )
        return ordered

    def _record_match_diagnostic(
        self,
        frame: FrameContext,
        object_id: int,
        stage: str,
        limit: float,
        elapsed: float,
        outcome: str,
    ) -> None:
        self._last_match_diagnostics.setdefault(frame.session_id, {})[object_id] = {
            "frame_id": frame.frame_id,
            "object_id": object_id,
            "timeout_stage": stage,
            "timeout_limit_sec": float(limit),
            "elapsed_sec": float(elapsed),
            "frame_local_refinement_time_sec": float(elapsed),
            "outcome": outcome,
            **self.get_startup_diagnostics(frame.session_id),
        }

    def get_last_match_diagnostics(
        self, session_id: str, frame_id: str
    ) -> dict[int, dict[str, object]]:
        return {
            object_id: dict(value)
            for object_id, value in self._last_match_diagnostics.get(session_id, {}).items()
            if value.get("frame_id") == frame_id
        }

    def get_startup_diagnostics(self, session_id: str) -> dict[str, object]:
        state = self._sessions.get(session_id)
        if state is None:
            return {
                "model_preloaded": False,
                "warmup_completed": False,
                "warmup_time_sec": 0.0,
                "reference_cache_warmed": False,
                "reference_prepare_time_sec": 0.0,
            }
        return {
            "model_preloaded": state.model_preloaded,
            "warmup_completed": state.warmup_completed,
            "warmup_time_sec": state.warmup_time_sec,
            "reference_cache_warmed": state.reference_cache_warmed,
            "reference_prepare_time_sec": state.reference_prepare_time_sec,
        }

    async def _prepare_matching_runtime_and_references(
        self, state: _SessionState, prepare_epoch: int
    ) -> None:
        if not (
            self._settings.matching_dinov2_enabled
            and self._settings.matching_geometry_method in {"hybrid", "aliked_lightglue"}
            and self._settings.matching_local_refinement_enabled
            and (self._settings.matching_preload_models or self._settings.matching_warmup_enabled)
        ):
            return
        reference_started: float | None = None
        try:
            warmup = await asyncio.to_thread(self._warmup_service.warmup)
            state.model_preloaded = warmup.model_preloaded
            state.warmup_completed = warmup.warmup_completed
            state.warmup_time_sec = warmup.warmup_time_sec
            reference_started = self._clock()
            runtime = self._runtime_factory(self._settings)
            references = await state.reference_store.list()
            for reference in references:
                if state.reference_prepare_epoch != prepare_epoch:
                    return
                await self._ensure_reference_descriptor(
                    state.reference_store,
                    reference.object_id,
                    runtime,
                    enforce_timeout=False,
                )
                if self._local_pipeline is None:
                    continue
                snapshot, generation = await state.reference_store.descriptor_snapshot(
                    reference.object_id
                )
                if snapshot is None:
                    continue
                reference_hash = hashlib.sha256(snapshot.image_content).hexdigest()
                image = await asyncio.to_thread(
                    self._decode_image, snapshot.image_content, "Referans"
                )
                model_hash, features, metrics = await asyncio.to_thread(
                    self._local_pipeline.prepare_reference, image, reference_hash
                )
                current, current_generation = await state.reference_store.descriptor_snapshot(
                    reference.object_id
                )
                if (
                    state.reference_prepare_epoch != prepare_epoch
                    or current_generation != generation
                    or current is None
                    or hashlib.sha256(current.image_content).hexdigest() != reference_hash
                ):
                    return
                state.local_feature_cache.put(reference_hash, model_hash, features, metrics)
            state.reference_cache_warmed = bool(references)
        except Exception:
            logger.warning(
                "matching_startup_warmup_failed",
                extra={
                    "event": "matching_startup_warmup_failed",
                    "session_id": state.session_id,
                },
                exc_info=True,
            )
        finally:
            if state.reference_prepare_epoch == prepare_epoch:
                state.reference_prepare_time_sec = (
                    max(0.0, self._clock() - reference_started)
                    if reference_started is not None
                    else 0.0
                )

    def _reference_match_timeout(self) -> tuple[float, str]:
        if self._settings.matching_geometry_method in {"hybrid", "aliked_lightglue"}:
            return self._settings.matching_local_refinement_timeout_sec, "local_refinement"
        return min(
            self._settings.matching_coarse_timeout_seconds,
            self._settings.matching_reference_timeout_seconds,
        ), "dinov2_coarse"

    def _match_with_runtime_lock(self, runtime, object_id, reference_descriptor, frame_descriptor):
        lock = getattr(runtime, "inference_lock", None)
        with lock if lock is not None else nullcontext():
            return self._matching_pipeline.match_reference(
                object_id, reference_descriptor, frame_descriptor
            )

    def _match_selected_geometry(
        self,
        runtime,
        state: _SessionState,
        object_id: int,
        reference_descriptor,
        frame_descriptor,
        reference_content: bytes,
        frame_image,
        frame_hash: str,
    ):
        method = self._settings.matching_geometry_method
        if method == "dinov2":
            return self._match_with_runtime_lock(
                runtime, object_id, reference_descriptor, frame_descriptor
            )
        if not self._settings.matching_local_refinement_enabled or self._local_pipeline is None:
            if self._settings.matching_local_fallback_to_dinov2:
                return self._match_with_runtime_lock(
                    runtime, object_id, reference_descriptor, frame_descriptor
                )
            return None
        try:
            reference_hash = hashlib.sha256(reference_content).hexdigest()
            reference_image = self._decode_image(reference_content, "Referans")
            local_result = self._local_pipeline.match_reference(
                method=method,
                object_id=object_id,
                reference_descriptor=reference_descriptor,
                frame_descriptor=frame_descriptor,
                reference_image=reference_image,
                frame_image=frame_image,
                reference_hash=reference_hash,
                frame_hash=frame_hash,
                cache=state.local_feature_cache,
            )
            state.local_diagnostics[object_id] = local_result.diagnostics
            return local_result.matched
        except LocalArtifactUnavailable:
            logger.warning(
                "matching_local_refinement_unavailable",
                extra={"event": "matching_local_refinement_unavailable", "object_id": object_id},
                exc_info=True,
            )
            if self._settings.matching_local_fallback_to_dinov2:
                return self._match_with_runtime_lock(
                    runtime, object_id, reference_descriptor, frame_descriptor
                )
            return None
        except LocalFeatureError:
            logger.error(
                "matching_local_refinement_failed",
                extra={"event": "matching_local_refinement_failed", "object_id": object_id},
                exc_info=True,
            )
            return None

    async def _ensure_reference_descriptor(
        self,
        store: ReferenceStore,
        object_id: int,
        runtime: Dinov2DescriptorRuntime,
        enforce_timeout: bool = True,
    ) -> None:
        descriptor_lock = await store.descriptor_lock_for(object_id)
        async with descriptor_lock:
            state, generation = await store.descriptor_snapshot(object_id)
            if state is None:
                return
            image_hash = hashlib.sha256(state.image_content).hexdigest()
            try:
                timeout = (
                    self._settings.matching_dinov2_timeout_seconds
                    if enforce_timeout
                    else None
                )
                model_hash_task = asyncio.to_thread(lambda: runtime.model_hash)
                model_hash = (
                    await model_hash_task
                    if timeout is None
                    else await asyncio.wait_for(model_hash_task, timeout=timeout)
                )
                if (
                    state.descriptor_ready
                    and state.descriptor_image_hash == image_hash
                    and state.descriptor_model_hash == model_hash
                ):
                    await store.touch_descriptor(object_id)
                    logger.info(
                        "matching_dinov2_reference_cache_hit",
                        extra={"event": "matching_dinov2_reference_cache_hit", "session_id": store.session_id,
                               "object_id": object_id},
                    )
                    return
                if state.descriptor_ready:
                    await store.invalidate_descriptor(object_id, "source_or_model_changed")
                    state, generation = await store.descriptor_snapshot(object_id)
                    if state is None:
                        return
                logger.info(
                    "matching_dinov2_reference_cache_miss",
                    extra={"event": "matching_dinov2_reference_cache_miss", "session_id": store.session_id,
                           "object_id": object_id},
                )
                image = await asyncio.to_thread(self._decode_image, state.image_content, "Referans")
                extraction_task = asyncio.to_thread(runtime.extract, image, image_hash)
                descriptor, metrics = (
                    await extraction_task
                    if timeout is None
                    else await asyncio.wait_for(extraction_task, timeout=timeout)
                )
                committed = await store.commit_descriptor(
                    object_id, generation, image_hash, model_hash, descriptor
                )
                if committed:
                    logger.info(
                        "matching_dinov2_reference_cached",
                        extra={
                            "event": "matching_dinov2_reference_cached",
                            "session_id": store.session_id,
                            "object_id": object_id,
                            "descriptor_shape": descriptor.shape,
                            "descriptor_bytes": metrics.descriptor_bytes,
                            "forward_seconds": metrics.forward_seconds,
                        },
                    )
                    await self._enforce_global_cache_limit()
            except asyncio.TimeoutError:
                await store.set_descriptor_error(object_id, generation, image_hash, "timeout")
                logger.error(
                    "matching_dinov2_timeout",
                    extra={"event": "matching_dinov2_timeout", "source": "reference", "session_id": store.session_id,
                           "object_id": object_id},
                )
            except Dinov2CudaOutOfMemory:
                await store.set_descriptor_error(object_id, generation, image_hash, "cuda_out_of_memory")
                logger.error(
                    "matching_reference_descriptor_oom",
                    extra={"event": "matching_reference_descriptor_oom", "session_id": store.session_id,
                           "object_id": object_id},
                )
            except (Dinov2ConfigurationError, Dinov2DescriptorError, ValueError, OSError) as exc:
                await store.set_descriptor_error(
                    object_id, generation, image_hash, type(exc).__name__
                )
                logger.error(
                    "matching_reference_descriptor_failed",
                    extra={"event": "matching_reference_descriptor_failed", "session_id": store.session_id,
                           "object_id": object_id, "reason": type(exc).__name__},
                    exc_info=True,
                )

    async def _enforce_global_cache_limit(self) -> None:
        async with self._registry_lock:
            stores = [state.reference_store for state in self._sessions.values()]
        entries: list[tuple[float, ReferenceStore, int, int]] = []
        for store in stores:
            for object_id, last_accessed, size in await store.descriptor_cache_summary():
                entries.append((last_accessed, store, object_id, size))
        max_bytes = int(self._settings.matching_dinov2_max_cache_mb * 1024 * 1024)
        total = sum(entry[3] for entry in entries)
        while entries and total > max_bytes:
            last_accessed, store, object_id, size = min(entries, key=lambda entry: entry[0])
            entries.remove((last_accessed, store, object_id, size))
            if await store.evict_descriptor(object_id, "global_memory_limit"):
                total -= size

    async def active_reference_ids(
        self,
        session_id: str,
        frame_index: int | None,
    ) -> tuple[int, ...]:
        state = await self._get_session(session_id, create=False)
        if state is None:
            return ()
        active = await state.reference_store.active_for_frame(frame_index)
        return tuple(reference.object_id for reference in active)

    async def list_references(
        self, session_id: str
    ) -> tuple[ImageModality | None, list[ReferenceStateInfo]]:
        state = await self._get_session(session_id, create=False)
        if state is None:
            return None, []
        references = await state.reference_store.list()
        return state.frame_modality, [
            ReferenceStateInfo(
                object_id=reference.object_id,
                active_from_frame=reference.frame_start,
                active_until_frame=reference.frame_end,
                modality=reference.modality,
                official_reference_url=reference.official_reference_url,
                order=reference.order,
                image_url=reference.image_url,
                video_name=reference.video_name,
                embedding_ready=reference.embedding_ready,
            )
            for reference in references
        ]

    async def get_reference_states(self, session_id: str) -> tuple[ReferenceState, ...]:
        """Return immutable state snapshots for diagnostics and unit tests."""
        state = await self._get_session(session_id, create=False)
        if state is None:
            return ()
        return await state.reference_store.list()

    async def remove_reference(self, session_id: str, object_id: int) -> bool:
        state = await self._get_session(session_id, create=False)
        if state is None:
            return False
        removed = await state.reference_store.remove(object_id)
        if removed:
            state.local_feature_cache.clear()
            state.local_diagnostics.pop(object_id, None)
            logger.info(
                "matching_reference_removed",
                extra={
                    "event": "matching_reference_removed",
                    "session_id": session_id,
                    "object_id": object_id,
                },
            )
        return removed

    async def clear_session(self, session_id: str) -> None:
        async with self._registry_lock:
            state = self._sessions.pop(session_id, None)
            self._last_match_diagnostics.pop(session_id, None)
        if state is not None:
            state.local_feature_cache.clear()
            state.local_diagnostics.clear()
            await state.reference_store.clear()
            logger.info(
                "matching_reference_session_cleared",
                extra={
                    "event": "matching_reference_session_cleared",
                    "session_id": session_id,
                },
            )

    async def purge_expired_sessions(self) -> int:
        expired = await self._take_expired_sessions()
        for state in expired:
            state.local_feature_cache.clear()
            state.local_diagnostics.clear()
            await state.reference_store.clear()
        return len(expired)

    async def _get_session(self, session_id: str, *, create: bool) -> _SessionState | None:
        if not session_id.strip():
            raise ValueError("session_id bos olamaz.")
        evicted: list[_SessionState] = []
        now = self._clock()
        async with self._registry_lock:
            evicted.extend(self._pop_expired_locked(now))
            state = self._sessions.get(session_id)
            if state is None and create:
                while len(self._sessions) >= self._settings.matching_max_reference_sessions:
                    oldest_id = min(
                        self._sessions,
                        key=lambda key: self._sessions[key].reference_store.last_access_time,
                    )
                    evicted.append(self._sessions.pop(oldest_id))
                store = ReferenceStore(
                    session_id,
                    ttl_seconds=self._settings.matching_reference_ttl_seconds,
                    hash_enabled=self._settings.matching_reference_hash_enabled,
                    cache_enabled=self._settings.matching_reference_cache_enabled,
                    decoder=self._decode_reference,
                    clock=self._clock,
                    max_cached_references=self._settings.matching_dinov2_max_cached_references,
                    max_cache_bytes=int(self._settings.matching_dinov2_max_cache_mb * 1024 * 1024),
                )
                state = _SessionState(session_id=session_id, reference_store=store)
                self._sessions[session_id] = state
        for old_state in evicted:
            old_state.local_feature_cache.clear()
            old_state.local_diagnostics.clear()
            await old_state.reference_store.clear()
            logger.info(
                "matching_reference_session_evicted",
                extra={
                    "event": "matching_reference_session_evicted",
                    "session_id": old_state.session_id,
                },
            )
        return state

    async def _take_expired_sessions(self) -> list[_SessionState]:
        async with self._registry_lock:
            return self._pop_expired_locked(self._clock())

    def _pop_expired_locked(self, now: float) -> list[_SessionState]:
        expired_ids = [
            session_id
            for session_id, state in self._sessions.items()
            if state.reference_store.is_expired(now)
        ]
        return [self._sessions.pop(session_id) for session_id in expired_ids]

    def _decode_reference(
        self,
        content: bytes,
        declared_modality: ImageModality | None,
    ) -> DecodedReference:
        image = self._decode_image(content, "Referans")
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Referans goruntu boyutu gecersiz.")
        modality = declared_modality or self._modality_detector.detect(image)
        return DecodedReference(image_size=(width, height), modality=modality)

    @staticmethod
    def _decode_image(content: bytes, label: str):
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"{label} goruntu decode edilemedi.")
        return image
