from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.schemas import ImageModality, MatchedReferenceObject
from app.services.common import FrameContext


@dataclass(frozen=True)
class ReferenceImage:
    object_id: int
    content: bytes
    active_from_frame: int | None = None
    active_until_frame: int | None = None
    modality: ImageModality | None = None
    official_reference_url: str | None = None
    order: int | None = None
    image_url: str | None = None
    video_name: str | None = None

    def is_active(self, frame_index: int | None) -> bool:
        if self.active_from_frame is None and self.active_until_frame is None:
            return True
        if frame_index is None:
            return False
        if self.active_from_frame is not None and frame_index < self.active_from_frame:
            return False
        if self.active_until_frame is not None and frame_index > self.active_until_frame:
            return False
        return True


@dataclass(frozen=True)
class ReferenceStateInfo:
    object_id: int
    active_from_frame: int | None
    active_until_frame: int | None
    modality: ImageModality
    official_reference_url: str | None = None
    order: int | None = None
    image_url: str | None = None
    video_name: str | None = None
    embedding_ready: bool = False


class ReferenceMatchingService(ABC):
    @abstractmethod
    async def set_references(
        self,
        session_id: str,
        references: list[ReferenceImage],
        frame_modality: ImageModality | None = None,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    async def process_frame(self, frame: FrameContext) -> list[MatchedReferenceObject]:
        """Return an empty list when no reference or no verified match exists."""
        raise NotImplementedError

    @abstractmethod
    async def clear_session(self, session_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_references(
        self, session_id: str
    ) -> tuple[ImageModality | None, list[ReferenceStateInfo]]:
        raise NotImplementedError

    @abstractmethod
    async def remove_reference(self, session_id: str, object_id: int) -> bool:
        raise NotImplementedError
