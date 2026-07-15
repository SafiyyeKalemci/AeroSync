from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas import DetectedObject
from app.services.common import FrameContext


class DetectionService(ABC):
    @abstractmethod
    async def process_frame(self, frame: FrameContext) -> list[DetectedObject]:
        """Return only genuine model detections; never synthesize fallback boxes."""
        raise NotImplementedError

    @abstractmethod
    async def reset_session(self, session_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def reset_all_sessions(self) -> None:
        raise NotImplementedError
