from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from app.schemas import FrameRequest

logger = logging.getLogger(__name__)


def _finite_float_or_none(value: Any) -> float | None:
    """Sunucu, sağlıksız GPS durumunda translation_x/y/z için "NaN" string'i
    gönderebilir (bkz. şartname Şekil 16 örneği). Bu değer, ground truth'un
    mevcut olmadığı anlamına gelir ve None'a eşlenir; FrameRequest alanı
    (FiniteFloat | None) None kabul eder ama NaN/Inf'i reddeder."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


class OfficialFrameMappingError(ValueError):
    pass


_FRAME_FIELDS = {"url", "image_url", "video_name"}
_TRANSLATION_FIELDS = {
    "url",
    "image_url",
    "health_status",
    "translation_x",
    "translation_y",
    "translation_z",
}


def _log_unknown(source: str, payload: dict[str, Any], known: set[str]) -> None:
    unknown = sorted(set(payload) - known)
    if unknown:
        logger.warning(
            "official_payload_unknown_fields",
            extra={"event": "official_payload_unknown_fields", "source": source, "fields": unknown},
        )


def map_official_frame(
    frame: dict[str, Any],
    translation: dict[str, Any] | None,
    *,
    session_id: str,
    frame_index: int,
    local_image_path: Path,
) -> FrameRequest:
    _log_unknown("frame", frame, _FRAME_FIELDS)
    missing = sorted(_FRAME_FIELDS - set(frame))
    if missing:
        raise OfficialFrameMappingError(
            "Resmî frame alanları eksik: " + ", ".join(missing)
        )

    health = None
    translation_x = translation_y = translation_z = None
    if translation is not None:
        _log_unknown("translation", translation, _TRANSLATION_FIELDS)
        health = translation.get("health_status")
        if health is not None:
            try:
                health = int(health)
            except (TypeError, ValueError) as exc:
                raise OfficialFrameMappingError(
                    "health_status 0, 1 veya null olmalıdır."
                ) from exc
        translation_x = _finite_float_or_none(translation.get("translation_x"))
        translation_y = _finite_float_or_none(translation.get("translation_y"))
        translation_z = _finite_float_or_none(translation.get("translation_z"))
        if health == 0 and any(
            v is None for v in (translation_x, translation_y, translation_z)
        ):
            logger.info(
                "official_translation_unhealthy_ground_truth_absent",
                extra={
                    "event": "official_translation_unhealthy_ground_truth_absent",
                    "session_id": session_id,
                    "frame_index": frame_index,
                },
            )

    return FrameRequest(
        url=str(frame["url"]),
        image_url=str(local_image_path.resolve()),
        video_name=str(frame["video_name"]),
        session=session_id,
        frame_index=frame_index,
        translation_x=translation_x,
        translation_y=translation_y,
        translation_z=translation_z,
        gps_health_status=health,
    )

