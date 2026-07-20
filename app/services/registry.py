from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config import Settings
from app.services.detection import (
    DetectionService,
    DisabledDetectionService,
    YoloDetectionService,
)
from app.services.localization import AffineLocalizationService, DisabledLocalizationService, LocalizationService
from app.services.localization.session_store import LocalizationSessionStore
from app.services.matching import DinoReferenceMatchingService, ReferenceMatchingService

logger = logging.getLogger(__name__)


@dataclass
class ServiceRegistry:
    detection: DetectionService
    localization: LocalizationService
    localization_sessions: LocalizationSessionStore
    matching: ReferenceMatchingService


def build_services(settings: Settings) -> ServiceRegistry:
    try:
        settings.validate_detection_motion()
        settings.validate_detection_landing()
        motion_config_valid = True
    except ValueError:
        logger.error("detection_configuration_invalid", exc_info=True)
        motion_config_valid = False
    if settings.detection_enabled and motion_config_valid:
        try:
            detection: DetectionService = YoloDetectionService(settings)
        except ValueError:
            logger.error("detection_configuration_invalid", exc_info=True)
            detection = DisabledDetectionService()
    else:
        detection = DisabledDetectionService()
    try:
        settings.validate_localization_vo()
        localization_config_valid = True
    except ValueError:
        logger.error("localization_configuration_invalid", exc_info=True)
        localization_config_valid = False
    if settings.localization_enabled and settings.localization_vo_enabled and localization_config_valid:
        localization: LocalizationService = AffineLocalizationService(settings)
    else:
        localization = DisabledLocalizationService()
    if settings.matching_engine == "gorev3":
        from app.services.matching.gorev3_service import Gorev3MatchingService

        matching: ReferenceMatchingService = Gorev3MatchingService(settings)
        logger.info("matching_engine_selected", extra={"engine": "gorev3"})
    else:
        matching = DinoReferenceMatchingService(settings)
    return ServiceRegistry(
        detection=detection,
        localization=localization,
        localization_sessions=LocalizationSessionStore(
            ttl_seconds=settings.localization_session_ttl_seconds,
            max_sessions=settings.localization_max_sessions,
        ),
        matching=matching,
    )
