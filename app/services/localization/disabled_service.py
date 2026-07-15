import logging
from datetime import datetime, timezone

from app.schemas import DetectedTranslation
from app.services.common import FrameContext
from app.services.localization.interface import LocalizationService, LocalizationSessionState

logger = logging.getLogger(__name__)


class DisabledLocalizationService(LocalizationService):
    async def process_frame(
        self,
        frame: FrameContext,
        state: LocalizationSessionState,
    ) -> DetectedTranslation | None:
        state.frame_count += 1
        state.last_frame_id = frame.frame_id
        state.last_gps_health_status = frame.gps_health_status
        state.updated_at = datetime.now(timezone.utc)
        logger.info(
            "Görev 2 devre dışı; session=%s frame=%s için pozisyon üretilmedi.",
            frame.session_id,
            frame.frame_id,
        )
        return None
