from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.schemas import ImageModality
from app.services.matching.interface import ReferenceImage
from app.services.matching.reference_catalog import ReferenceCatalog
from app.services.matching.descriptor_types import DenseDescriptorSet
from app.services.matching.reference_state import DecodeStatus, DownloadStatus, ReferenceState
from app.utils.images import detect_image_format

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DecodedReference:
    image_size: tuple[int, int]
    modality: ImageModality


DecodeReference = Callable[[bytes, ImageModality | None], DecodedReference]
Clock = Callable[[], float]


class ReferenceStore:
    """Session-owned, lock-protected reference metadata and decode cache."""

    def __init__(
        self,
        session_id: str,
        *,
        ttl_seconds: float,
        hash_enabled: bool,
        cache_enabled: bool,
        decoder: DecodeReference,
        clock: Clock = time.monotonic,
        max_cached_references: int = 32,
        max_cache_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("MATCHING_REFERENCE_TTL_SECONDS pozitif olmalidir.")
        self.session_id = session_id
        self.lock = asyncio.Lock()
        self._ttl_seconds = ttl_seconds
        self._hash_enabled = hash_enabled
        self._cache_enabled = cache_enabled
        self._decoder = decoder
        self._clock = clock
        self._max_cached_references = max_cached_references
        self._max_cache_bytes = max_cache_bytes
        self._references: dict[int, ReferenceState] = {}
        self._decode_cache: dict[str, DecodedReference] = {}
        self._descriptor_locks: dict[int, asyncio.Lock] = {}
        self._generation = 0
        self._last_access_time = clock()

    @property
    def last_access_time(self) -> float:
        return self._last_access_time

    @property
    def generation(self) -> int:
        return self._generation

    def is_expired(self, now: float | None = None) -> bool:
        current = self._clock() if now is None else now
        return current - self._last_access_time >= self._ttl_seconds

    async def replace(self, references: list[ReferenceImage]) -> int:
        async with self.lock:
            now = self._touch()
            if not references:
                self._references.clear()
                self._decode_cache.clear()
                self._descriptor_locks.clear()
                self._generation += 1
                return 0

            incoming_ids = [reference.object_id for reference in references]
            if len(incoming_ids) != len(set(incoming_ids)):
                raise ValueError("Ayni object_id bir reference set icinde tekrar edemez.")

            updated: dict[int, ReferenceState] = {}
            cache_before_update = dict(self._decode_cache)
            try:
                for reference in references:
                    state = await self._build_state(reference, now)
                    previous = self._references.get(reference.object_id)
                    if (
                        previous is not None
                        and previous.descriptor_ready
                        and self._content_hash(previous) == self._content_hash(state)
                    ):
                        state = self._copy_descriptor(previous, state)
                    elif previous is not None and previous.descriptor_ready:
                        logger.info(
                            "matching_dinov2_reference_cache_invalidated",
                            extra={"event": "matching_dinov2_reference_cache_invalidated",
                                   "session_id": self.session_id,
                                   "object_id": reference.object_id,
                                   "reason": "image_changed"},
                        )
                    updated[reference.object_id] = state
            except Exception:
                self._decode_cache = cache_before_update
                raise
            self._references = updated
            self._descriptor_locks = {
                object_id: self._descriptor_locks.get(object_id, asyncio.Lock())
                for object_id in updated
            }
            self._generation += 1

            logger.info(
                "matching_reference_store_updated",
                extra={
                    "event": "matching_reference_store_updated",
                    "session_id": self.session_id,
                    "reference_count": len(self._references),
                },
            )
            return len(self._references)

    async def list(self) -> tuple[ReferenceState, ...]:
        async with self.lock:
            now = self._touch()
            self._references = {
                object_id: state.accessed(now)
                for object_id, state in self._references.items()
            }
            return self._ordered_states()

    async def active_for_frame(self, frame_index: int | None) -> tuple[ReferenceState, ...]:
        async with self.lock:
            now = self._touch()
            active = ReferenceCatalog(self._ordered_states()).active_for_frame(frame_index)
            for state in active:
                self._references[state.object_id] = state.accessed(now)
            return tuple(self._references[state.object_id] for state in active)

    async def remove(self, object_id: int) -> bool:
        async with self.lock:
            self._touch()
            removed = self._references.pop(object_id, None) is not None
            if removed:
                self._descriptor_locks.pop(object_id, None)
                self._generation += 1
            return removed

    async def clear(self) -> None:
        async with self.lock:
            self._references.clear()
            self._decode_cache.clear()
            self._descriptor_locks.clear()
            self._generation += 1
            self._touch()

    async def descriptor_lock_for(self, object_id: int) -> asyncio.Lock:
        async with self.lock:
            return self._descriptor_locks.setdefault(object_id, asyncio.Lock())

    async def descriptor_snapshot(self, object_id: int) -> tuple[ReferenceState | None, int]:
        async with self.lock:
            return self._references.get(object_id), self._generation

    async def generation_token(self) -> int:
        async with self.lock:
            return self._generation

    async def touch_descriptor(self, object_id: int) -> ReferenceState | None:
        async with self.lock:
            state = self._references.get(object_id)
            if state is None or not state.descriptor_ready:
                return None
            state = state.descriptor_accessed(self._touch())
            self._references[object_id] = state
            return state

    async def set_descriptor_error(
        self,
        object_id: int,
        generation: int,
        image_hash: str | None,
        error: str,
    ) -> bool:
        async with self.lock:
            state = self._references.get(object_id)
            if (
                state is None
                or generation != self._generation
                or self._content_hash(state) != image_hash
            ):
                return False
            self._references[object_id] = state.without_descriptor(error)
            return True

    async def invalidate_descriptor(self, object_id: int, reason: str) -> bool:
        async with self.lock:
            state = self._references.get(object_id)
            if state is None or not state.descriptor_ready:
                return False
            self._references[object_id] = state.without_descriptor(reason)
            logger.info(
                "matching_dinov2_reference_cache_invalidated",
                extra={"event": "matching_dinov2_reference_cache_invalidated", "session_id": self.session_id,
                       "object_id": object_id, "reason": reason},
            )
            return True

    async def commit_descriptor(
        self,
        object_id: int,
        generation: int,
        image_hash: str | None,
        model_hash: str,
        descriptor: DenseDescriptorSet,
    ) -> bool:
        async with self.lock:
            state = self._references.get(object_id)
            if (
                state is None
                or generation != self._generation
                or self._content_hash(state) != image_hash
            ):
                logger.info(
                    "matching_dinov2_stale_worker_discarded",
                    extra={"event": "matching_dinov2_stale_worker_discarded", "session_id": self.session_id,
                           "object_id": object_id},
                )
                return False
            self._references[object_id] = state.with_descriptor(
                descriptor, model_hash=model_hash, image_hash=image_hash, now=self._touch()
            )
            self._evict_lru_locked()
            return self._references[object_id].descriptor_ready

    async def descriptor_cache_summary(self) -> list[tuple[int, float, int]]:
        async with self.lock:
            return [
                (state.object_id, state.descriptor_last_accessed or 0.0, state.dense_descriptors.nbytes)
                for state in self._references.values()
                if state.descriptor_ready and state.dense_descriptors is not None
            ]

    async def evict_descriptor(self, object_id: int, reason: str = "lru") -> bool:
        async with self.lock:
            state = self._references.get(object_id)
            if state is None or not state.descriptor_ready:
                return False
            self._references[object_id] = state.without_descriptor()
            logger.info(
                "matching_descriptor_cache_evicted",
                extra={"event": "matching_descriptor_cache_evicted", "session_id": self.session_id,
                       "object_id": object_id, "reason": reason},
            )
            return True

    async def _build_state(self, reference: ReferenceImage, now: float) -> ReferenceState:
        self._validate_reference(reference)
        detect_image_format(reference.content)
        reference_hash = (
            hashlib.sha256(reference.content).hexdigest() if self._hash_enabled else None
        )

        decoded = self._decode_cache.get(reference_hash) if (
            self._cache_enabled and reference_hash is not None
        ) else None
        if decoded is None:
            decoded = await asyncio.to_thread(
                self._decoder,
                reference.content,
                reference.modality,
            )
            if self._cache_enabled and reference_hash is not None:
                self._decode_cache[reference_hash] = decoded

        return ReferenceState(
            official_reference_url=reference.official_reference_url,
            object_id=reference.object_id,
            order=reference.order if reference.order is not None else reference.object_id,
            frame_start=reference.active_from_frame,
            frame_end=reference.active_until_frame,
            image_url=reference.image_url,
            video_name=reference.video_name,
            download_status=DownloadStatus.DOWNLOADED,
            decode_status=DecodeStatus.DECODED,
            embedding_ready=False,
            descriptor_ready=False,
            descriptor_error=None,
            descriptor_shape=None,
            descriptor_device=None,
            descriptor_dtype=None,
            descriptor_model_hash=None,
            descriptor_image_hash=None,
            descriptor_created_at=None,
            descriptor_last_accessed=None,
            dense_descriptors=None,
            last_access_time=now,
            reference_hash=reference_hash,
            image_size=decoded.image_size,
            modality=decoded.modality,
            image_content=reference.content,
        )

    @staticmethod
    def _validate_reference(reference: ReferenceImage) -> None:
        if reference.object_id <= 0:
            raise ValueError("object_id pozitif olmalidir.")
        if not reference.content:
            raise ValueError("Referans goruntu icerigi bos olamaz.")
        if reference.order is not None and reference.order < 0:
            raise ValueError("Reference order negatif olamaz.")
        if reference.active_from_frame is not None and reference.active_from_frame < 0:
            raise ValueError("frame_start negatif olamaz.")
        if reference.active_until_frame is not None and reference.active_until_frame < 0:
            raise ValueError("frame_end negatif olamaz.")
        if (
            reference.active_from_frame is not None
            and reference.active_until_frame is not None
            and reference.active_until_frame < reference.active_from_frame
        ):
            raise ValueError("frame_end, frame_start degerinden kucuk olamaz.")
        official_values = (
            reference.official_reference_url,
            reference.image_url,
            reference.video_name,
            reference.order,
        )
        if any(value is not None for value in official_values):
            if not all(value is not None for value in official_values):
                raise ValueError("Resmi referans metadata alanlari birlikte verilmelidir.")
            for field_name, value in (
                ("official_reference_url", reference.official_reference_url),
                ("image_url", reference.image_url),
                ("video_name", reference.video_name),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{field_name} bos olamaz.")

    def _ordered_states(self) -> tuple[ReferenceState, ...]:
        return tuple(
            sorted(
                self._references.values(),
                key=lambda state: (state.order, state.object_id),
            )
        )

    @staticmethod
    def _copy_descriptor(source: ReferenceState, target: ReferenceState) -> ReferenceState:
        return target.with_descriptor(
            source.dense_descriptors,
            model_hash=source.descriptor_model_hash or "",
            image_hash=source.descriptor_image_hash or "",
            now=source.descriptor_created_at or target.last_access_time,
        ) if source.dense_descriptors is not None else target

    @staticmethod
    def _content_hash(state: ReferenceState) -> str:
        return state.reference_hash or hashlib.sha256(state.image_content).hexdigest()

    def _evict_lru_locked(self) -> None:
        ready = [state for state in self._references.values() if state.descriptor_ready]
        total_bytes = sum(
            state.dense_descriptors.nbytes for state in ready if state.dense_descriptors is not None
        )
        while ready and (
            len(ready) > self._max_cached_references or total_bytes > self._max_cache_bytes
        ):
            victim = min(ready, key=lambda state: state.descriptor_last_accessed or 0.0)
            victim_bytes = victim.dense_descriptors.nbytes if victim.dense_descriptors else 0
            self._references[victim.object_id] = victim.without_descriptor()
            ready.remove(victim)
            total_bytes -= victim_bytes
            logger.info(
                "matching_descriptor_cache_evicted",
                extra={"event": "matching_descriptor_cache_evicted", "session_id": self.session_id,
                       "object_id": victim.object_id, "reason": "session_limit"},
            )

    def _touch(self) -> float:
        self._last_access_time = self._clock()
        return self._last_access_time
