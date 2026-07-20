"""teknofest_gorev3 ObjectMatcher'ini resmi ReferenceMatchingService arayuzune baglar.

MATCHING_ENGINE=gorev3 ile secilir. Kanitlanmis algoritma (dense DINOv2 +
heatmap yedegi + detect-then-track + XoFTR cross-modal) gorev3/ paketinde
birebir korunur; bu modul yalnizca oturum/referans yasam dongusunu ve
sema donusumunu ustlenir.

Not: Yarisma sozlesmesinde frame'ler daima RGB gelir, referanslar RGB veya
termal olabilir. Bu serviste frame modalitesine gore atlama YAPILMAZ;
termal referans <-> RGB frame durumu matcher'in cross-modal yolunda islenir.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field

from app.core.config import Settings
from app.schemas import ImageModality, MatchedReferenceObject
from app.services.common import FrameContext
from app.services.matching.interface import (
    ReferenceImage,
    ReferenceMatchingService,
    ReferenceStateInfo,
)
from app.utils.images import read_image_bytes

logger = logging.getLogger(__name__)


@dataclass
class _PreparedReference:
    reference: ReferenceImage
    matcher: object | None          # gorev3 ObjectMatcher; None -> hazirlik basarisiz
    modality: ImageModality


@dataclass
class _Session:
    session_id: str
    frame_modality: ImageModality | None = None
    references: list[_PreparedReference] = field(default_factory=list)


class Gorev3MatchingService(ReferenceMatchingService):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: dict[str, _Session] = {}
        self._registry_lock = asyncio.Lock()
        # Modeller modul-seviyesi tekil nesneler (GPU) -> cikarim tek is akisindan
        self._infer_lock = threading.Lock()

    async def set_references(
        self,
        session_id: str,
        references: list[ReferenceImage],
        frame_modality: ImageModality | None = None,
    ) -> int:
        if not self._settings.matching_enabled:
            logger.info(
                "matching_disabled",
                extra={"event": "matching_disabled", "session_id": session_id},
            )
            return 0
        prepared: list[_PreparedReference] = []
        for reference in references:
            prepared.append(await asyncio.to_thread(self._prepare_reference, reference))
        session = _Session(
            session_id=session_id,
            frame_modality=frame_modality,
            references=prepared,
        )
        async with self._registry_lock:
            self._sessions[session_id] = session
        loaded = sum(1 for item in prepared if item.matcher is not None)
        logger.info(
            "gorev3_references_loaded",
            extra={
                "event": "gorev3_references_loaded",
                "session_id": session_id,
                "reference_count": loaded,
                "failed_count": len(prepared) - loaded,
            },
        )
        return loaded

    def _prepare_reference(self, reference: ReferenceImage) -> _PreparedReference:
        import cv2
        import numpy as np

        try:
            image = cv2.imdecode(
                np.frombuffer(reference.content, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise ValueError("Referans goruntu decode edilemedi.")
            # Agir modeller ilk kullanimda yuklenir (modul importu)
            from app.services.matching.gorev3.dinov2_matcher import ObjectMatcher, _is_thermal

            with self._infer_lock:
                matcher = ObjectMatcher()
                matcher.set_reference(image)
            modality = reference.modality or (
                ImageModality.THERMAL if _is_thermal(image) else ImageModality.RGB
            )
            return _PreparedReference(reference, matcher, modality)
        except Exception:
            logger.error(
                "gorev3_reference_prepare_failed",
                extra={
                    "event": "gorev3_reference_prepare_failed",
                    "object_id": reference.object_id,
                },
                exc_info=True,
            )
            return _PreparedReference(
                reference, None, reference.modality or ImageModality.UNKNOWN
            )

    async def process_frame(self, frame: FrameContext) -> list[MatchedReferenceObject]:
        if not self._settings.matching_enabled:
            return []
        async with self._registry_lock:
            session = self._sessions.get(frame.session_id)
        if session is None:
            return []
        active = [
            item
            for item in session.references
            if item.matcher is not None and item.reference.is_active(frame.frame_index)
        ]
        if not active:
            return []
        try:
            content = await read_image_bytes(
                frame.image_url, self._settings.matching_timeout_seconds
            )
            frame_bgr = await asyncio.to_thread(self._decode_image, content)
        except Exception:
            logger.error(
                "gorev3_frame_decode_failed",
                extra={
                    "event": "gorev3_frame_decode_failed",
                    "session_id": frame.session_id,
                    "frame_id": frame.frame_id,
                },
                exc_info=True,
            )
            return []

        results: list[MatchedReferenceObject] = []
        for item in active:
            try:
                boxes = await asyncio.to_thread(self._find, item.matcher, frame_bgr)
            except Exception:
                logger.error(
                    "gorev3_match_failed",
                    extra={
                        "event": "gorev3_match_failed",
                        "session_id": frame.session_id,
                        "frame_id": frame.frame_id,
                        "object_id": item.reference.object_id,
                    },
                    exc_info=True,
                )
                continue
            matched = self._to_matched(
                item.reference.object_id, boxes, frame_bgr.shape
            )
            if matched is not None:
                results.append(matched)
        logger.info(
            "gorev3_frame_completed",
            extra={
                "event": "gorev3_frame_completed",
                "session_id": frame.session_id,
                "frame_id": frame.frame_id,
                "result_count": len(results),
            },
        )
        return results

    def _find(self, matcher, frame_bgr):
        with self._infer_lock:
            boxes = matcher.find(frame_bgr)
            if boxes or not self._settings.matching_gorev3_window_fallback:
                return boxes
            # Son care: referans yalnizca kendi penceresinde aktif geldigi icin
            # nesne sahnede; asiri bakis acisi farkinda politika hicbir kutu
            # birakmadiysa dusuk esikli en iyi heatmap adayini gonder.
            from app.services.matching.gorev3.dinov2_matcher import (
                fallback_best_candidate,
            )

            best = fallback_best_candidate(matcher._ref_feat, frame_bgr)
            return [best] if best is not None else []

    @staticmethod
    def _to_matched(object_id: int, boxes: list, frame_shape) -> MatchedReferenceObject | None:
        if not boxes:
            return None
        # mAP'te fazla kutu ceza yer (sartname Ornek 4) -> yalnizca en iyi kutu
        best = max(boxes, key=lambda b: b.get("conf", 0.0))
        frame_height, frame_width = frame_shape[:2]
        x1 = max(0.0, float(best["top_left_x"]))
        y1 = max(0.0, float(best["top_left_y"]))
        x2 = min(float(frame_width), float(best["bottom_right_x"]))
        y2 = min(float(frame_height), float(best["bottom_right_y"]))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            return None
        raw_conf = best.get("conf")
        confidence = (
            None if raw_conf is None else max(0.0, min(1.0, float(raw_conf)))
        )
        return MatchedReferenceObject(
            object_id=object_id,
            top_left_x=x1,
            top_left_y=y1,
            bottom_right_x=x2,
            bottom_right_y=y2,
            confidence=confidence,
        )

    @staticmethod
    def _decode_image(content: bytes):
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Frame goruntu decode edilemedi.")
        return image

    async def clear_session(self, session_id: str) -> None:
        async with self._registry_lock:
            removed = self._sessions.pop(session_id, None)
        if removed is not None:
            logger.info(
                "gorev3_session_cleared",
                extra={"event": "gorev3_session_cleared", "session_id": session_id},
            )

    async def list_references(
        self, session_id: str
    ) -> tuple[ImageModality | None, list[ReferenceStateInfo]]:
        async with self._registry_lock:
            session = self._sessions.get(session_id)
        if session is None:
            return None, []
        infos = [
            ReferenceStateInfo(
                object_id=item.reference.object_id,
                active_from_frame=item.reference.active_from_frame,
                active_until_frame=item.reference.active_until_frame,
                modality=item.modality,
                official_reference_url=item.reference.official_reference_url,
                order=item.reference.order,
                image_url=item.reference.image_url,
                video_name=item.reference.video_name,
                embedding_ready=item.matcher is not None,
            )
            for item in session.references
        ]
        return session.frame_modality, infos

    async def remove_reference(self, session_id: str, object_id: int) -> bool:
        async with self._registry_lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            before = len(session.references)
            session.references = [
                item
                for item in session.references
                if item.reference.object_id != object_id
            ]
            removed = len(session.references) != before
        if removed:
            logger.info(
                "gorev3_reference_removed",
                extra={
                    "event": "gorev3_reference_removed",
                    "session_id": session_id,
                    "object_id": object_id,
                },
            )
        return removed
