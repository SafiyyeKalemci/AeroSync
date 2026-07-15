from __future__ import annotations

import itertools
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from app.api.dependencies import verify_api_key
from app.schemas import (
    CompetitionResponse,
    FrameRequest,
    ReferenceInfo,
    ReferenceListResponse,
    ReferenceSetRequest,
)
from app.services.common import FrameContext
from app.services.detection.disabled_service import DisabledDetectionService
from app.services.localization.disabled_service import DisabledLocalizationService
from app.services.frame_processor import FrameProcessor
from app.services.matching.interface import ReferenceImage
from app.utils.images import read_image_bytes

logger = logging.getLogger(__name__)
router = APIRouter()
_response_ids = itertools.count(1)


class ReferenceLoadResponse(BaseModel):
    session: str
    loaded: int


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    services = request.app.state.services
    return {
        "status": "ok",
        "task1_detection": (
            "disabled" if isinstance(services.detection, DisabledDetectionService) else "enabled"
        ),
        "task2_localization": (
            "disabled" if isinstance(services.localization, DisabledLocalizationService) else "enabled"
        ),
        "task3_matching": "enabled" if settings.matching_enabled else "disabled",
    }


@router.post(
    "/sessions/{session_id}/references",
    response_model=ReferenceLoadResponse,
    dependencies=[Depends(verify_api_key)],
)
async def set_references(session_id: str, payload: ReferenceSetRequest, request: Request):
    settings = request.app.state.settings
    references: list[ReferenceImage] = []
    try:
        for item in payload.references:
            content = await read_image_bytes(item.image_url, settings.http_timeout_seconds)
            references.append(
                ReferenceImage(
                    object_id=item.object_id,
                    content=content,
                    active_from_frame=item.active_from_frame,
                    active_until_frame=item.active_until_frame,
                    modality=item.modality,
                )
            )
    except Exception as exc:
        logger.exception("Referans görüntüsü alınamadı.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Referans görüntüsü alınamadı: {exc}",
        ) from exc
    loaded = await request.app.state.services.matching.set_references(
        session_id,
        references,
        frame_modality=payload.frame_modality,
    )
    return ReferenceLoadResponse(session=session_id, loaded=loaded)


@router.get(
    "/sessions/{session_id}/references",
    response_model=ReferenceListResponse,
    dependencies=[Depends(verify_api_key)],
)
async def list_references(session_id: str, request: Request) -> ReferenceListResponse:
    frame_modality, references = await request.app.state.services.matching.list_references(
        session_id
    )
    return ReferenceListResponse(
        session=session_id,
        frame_modality=frame_modality,
        references=[
            ReferenceInfo(
                object_id=item.object_id,
                active_from_frame=item.active_from_frame,
                active_until_frame=item.active_until_frame,
                modality=item.modality,
            )
            for item in references
        ],
    )


@router.delete(
    "/sessions/{session_id}/references/{object_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_api_key)],
)
async def remove_reference(session_id: str, object_id: int, request: Request) -> None:
    removed = await request.app.state.services.matching.remove_reference(session_id, object_id)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Referans bulunamadı.",
        )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_api_key)],
)
async def clear_session(session_id: str, request: Request) -> None:
    services = request.app.state.services
    await services.detection.reset_session(session_id)
    await services.matching.clear_session(session_id)
    await services.localization.reset_session(session_id)
    await services.localization_sessions.reset(session_id)


@router.post(
    "/process_frame",
    response_model=CompetitionResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(verify_api_key)],
)
async def process_frame(payload: FrameRequest, request: Request) -> CompetitionResponse:
    services = request.app.state.services
    settings = request.app.state.settings
    frame = FrameContext.from_request(payload)
    processor = FrameProcessor(services)

    async with services.localization_sessions.locked(payload.session) as state:
        results = await processor.process(frame, state)

    return CompetitionResponse.from_task_results(
        response_id=next(_response_ids),
        user=settings.team_user_url,
        frame=payload.url,
        detected_objects=results.detected_objects,
        detected_translation=results.detected_translation,
        matched_reference_objects=results.matched_reference_objects,
    )
