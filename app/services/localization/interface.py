from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.schemas import DetectedTranslation, GPSHealthStatus
from app.services.common import FrameContext
from app.services.localization.state import VisualOdometryState


@dataclass
class LocalizationSessionState:
    session_id: str
    frame_count: int = 0
    last_frame_id: str | None = None
    last_gps_health_status: GPSHealthStatus | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    vo_state: VisualOdometryState | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LocalizationService(ABC):
    @abstractmethod
    async def process_frame(
        self,
        frame: FrameContext,
        state: LocalizationSessionState,
    ) -> DetectedTranslation | None:
        """Consume every frame and return None when no genuine estimate exists."""
        raise NotImplementedError

    async def reset_session(self, session_id: str) -> None:
        return None
