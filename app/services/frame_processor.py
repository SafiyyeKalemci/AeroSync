from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import TYPE_CHECKING, TypeVar

from pydantic import TypeAdapter

from app.schemas import DetectedObject, DetectedTranslation, MatchedReferenceObject
from app.services.common import FrameContext, FrameTaskResults
from app.services.localization.interface import LocalizationSessionState

if TYPE_CHECKING:
    from app.services.registry import ServiceRegistry

logger = logging.getLogger(__name__)
T = TypeVar("T")

_detections_adapter = TypeAdapter(list[DetectedObject])
_translation_adapter = TypeAdapter(DetectedTranslation | None)
_matches_adapter = TypeAdapter(list[MatchedReferenceObject])


async def _isolated_task(
    *,
    task_name: str,
    frame: FrameContext,
    operation: Awaitable[object],
    adapter: TypeAdapter[T],
    fallback: T,
) -> T:
    try:
        raw_result = await operation
        return adapter.validate_python(raw_result)
    except Exception:
        logger.error(
            "frame_task_failed",
            extra={
                "event": "frame_task_failed",
                "task": task_name,
                "session_id": frame.session_id,
                "frame_id": frame.frame_id,
            },
            exc_info=True,
        )
        return fallback


class FrameProcessor:
    """Runs task services independently and combines only validated results."""

    def __init__(self, services: "ServiceRegistry") -> None:
        self._services = services

    async def process(
        self,
        frame: FrameContext,
        localization_state: LocalizationSessionState,
    ) -> FrameTaskResults:
        detections, translation, matches = await asyncio.gather(
            _isolated_task(
                task_name="detection",
                frame=frame,
                operation=self._services.detection.process_frame(frame),
                adapter=_detections_adapter,
                fallback=[],
            ),
            _isolated_task(
                task_name="localization",
                frame=frame,
                operation=self._services.localization.process_frame(frame, localization_state),
                adapter=_translation_adapter,
                fallback=None,
            ),
            _isolated_task(
                task_name="matching",
                frame=frame,
                operation=self._services.matching.process_frame(frame),
                adapter=_matches_adapter,
                fallback=[],
            ),
        )
        return FrameTaskResults(
            detected_objects=detections,
            detected_translation=translation,
            matched_reference_objects=matches,
        )
