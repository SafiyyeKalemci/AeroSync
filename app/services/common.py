from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas import (
    DetectedObject,
    DetectedTranslation,
    FrameRequest,
    GPSHealthStatus,
    ImageModality,
    MatchedReferenceObject,
)


@dataclass(frozen=True)
class FrameContext:
    frame_id: str
    image_url: str
    video_name: str
    session_id: str
    gps_health_status: GPSHealthStatus | None
    gps_x: float | None
    gps_y: float | None
    gps_z: float | None
    frame_index: int | None = None
    image_modality: ImageModality | None = None

    @classmethod
    def from_request(cls, request: FrameRequest) -> "FrameContext":
        return cls(
            frame_id=request.url,
            image_url=request.image_url,
            video_name=request.video_name,
            session_id=request.session,
            frame_index=request.frame_index,
            image_modality=request.image_modality,
            gps_health_status=request.gps_health_status,
            gps_x=request.translation_x,
            gps_y=request.translation_y,
            gps_z=request.translation_z,
        )


@dataclass(frozen=True)
class FrameTaskResults:
    detected_objects: list[DetectedObject] = field(default_factory=list)
    detected_translation: DetectedTranslation | None = None
    matched_reference_objects: list[MatchedReferenceObject] = field(default_factory=list)
