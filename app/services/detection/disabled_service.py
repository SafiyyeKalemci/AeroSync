import logging

from app.schemas import DetectedObject
from app.services.common import FrameContext
from app.services.detection.interface import DetectionService

logger = logging.getLogger(__name__)


class DisabledDetectionService(DetectionService):
    async def process_frame(self, frame: FrameContext) -> list[DetectedObject]:
        logger.info(
            "Görev 1 devre dışı; frame=%s için nesne tespiti üretilmedi.",
            frame.frame_id,
        )
        return []

    async def reset_session(self, session_id: str) -> None:
        return None

    async def reset_all_sessions(self) -> None:
        return None
