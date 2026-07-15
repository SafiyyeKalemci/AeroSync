from __future__ import annotations

from typing import Any

from app.schemas import CompetitionResponse, LandingStatus, MotionStatus, ObjectClass
from competition.official_interface_adapter import OfficialBindings
from competition.reference_mapper import ReferenceCatalog

_CLASS_IDS = {
    ObjectClass.TASIT: 0,
    ObjectClass.INSAN: 1,
    ObjectClass.UAP: 2,
    ObjectClass.UAI: 3,
}
_LANDING_CODES = {
    LandingStatus.SUITABLE: "1",
    LandingStatus.UNSUITABLE: "0",
    LandingStatus.NOT_APPLICABLE: "-1",
}
_MOTION_CODES = {
    MotionStatus.MOVING: "1",
    MotionStatus.STATIONARY: "0",
    MotionStatus.UNKNOWN: "-1",
}


def map_aerosync_result(
    response: CompetitionResponse,
    *,
    official_frame: dict[str, Any],
    official_translation: dict[str, Any] | None,
    catalog: ReferenceCatalog,
    bindings: OfficialBindings,
):
    translation = official_translation or {}
    prediction = bindings.frame_predictions(
        response.frame,
        official_frame["image_url"],
        official_frame["video_name"],
        translation.get("translation_x"),
        translation.get("translation_y"),
        translation.get("translation_z"),
    )
    for item in response.detected_objects:
        prediction.add_detected_object(
            bindings.detected_object(
                (_CLASS_IDS[item.cls],),
                _LANDING_CODES[item.landing_status],
                _MOTION_CODES[item.motion_status],
                item.top_left_x,
                item.top_left_y,
                item.bottom_right_x,
                item.bottom_right_y,
            )
        )
    for item in response.detected_translations:
        prediction.add_translation_object(
            bindings.detected_translation(
                item.translation_x,
                item.translation_y,
                item.translation_z,
            )
        )
    for item in response.detected_undefined_objects:
        prediction.add_reference_prediction(
            bindings.reference_prediction(
                catalog.official_url_for(item.object_id),
                response.frame,
                item.top_left_x,
                item.top_left_y,
                item.bottom_right_x,
                item.bottom_right_y,
            )
        )
    return prediction

