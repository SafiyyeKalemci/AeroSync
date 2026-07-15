from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from app.schemas import ImageModality
from app.services.matching.descriptor_types import DenseDescriptorSet


class DownloadStatus(StrEnum):
    DOWNLOADED = "downloaded"
    FAILED = "failed"


class DecodeStatus(StrEnum):
    DECODED = "decoded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ReferenceState:
    official_reference_url: str | None
    object_id: int
    order: int
    frame_start: int | None
    frame_end: int | None
    image_url: str | None
    video_name: str | None
    download_status: DownloadStatus
    decode_status: DecodeStatus
    embedding_ready: bool
    descriptor_ready: bool
    descriptor_error: str | None
    descriptor_shape: tuple[int, int] | None
    descriptor_device: str | None
    descriptor_dtype: str | None
    descriptor_model_hash: str | None
    descriptor_image_hash: str | None
    descriptor_created_at: float | None
    descriptor_last_accessed: float | None
    dense_descriptors: DenseDescriptorSet | None
    last_access_time: float
    reference_hash: str | None
    image_size: tuple[int, int]
    modality: ImageModality
    image_content: bytes

    def is_active(self, frame_index: int | None) -> bool:
        if self.frame_start is None and self.frame_end is None:
            return True
        if frame_index is None:
            return False
        if self.frame_start is not None and frame_index < self.frame_start:
            return False
        if self.frame_end is not None and frame_index > self.frame_end:
            return False
        return True

    def accessed(self, now: float) -> "ReferenceState":
        return replace(self, last_access_time=now)

    def with_descriptor(
        self,
        descriptor: DenseDescriptorSet,
        *,
        model_hash: str,
        image_hash: str,
        now: float,
    ) -> "ReferenceState":
        return replace(
            self,
            embedding_ready=True,
            descriptor_ready=True,
            descriptor_error=None,
            descriptor_shape=descriptor.shape,
            descriptor_device=descriptor.device,
            descriptor_dtype=descriptor.dtype,
            descriptor_model_hash=model_hash,
            descriptor_image_hash=image_hash,
            descriptor_created_at=now,
            descriptor_last_accessed=now,
            dense_descriptors=descriptor,
        )

    def descriptor_accessed(self, now: float) -> "ReferenceState":
        return replace(self, descriptor_last_accessed=now, last_access_time=now)

    def without_descriptor(self, error: str | None = None) -> "ReferenceState":
        return replace(
            self,
            embedding_ready=False,
            descriptor_ready=False,
            descriptor_error=error,
            descriptor_shape=None,
            descriptor_device=None,
            descriptor_dtype=None,
            descriptor_model_hash=None,
            descriptor_image_hash=None,
            descriptor_created_at=None,
            descriptor_last_accessed=None,
            dense_descriptors=None,
        )
