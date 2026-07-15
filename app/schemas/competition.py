from __future__ import annotations

from enum import IntEnum, StrEnum
from pathlib import PureWindowsPath
from typing import Annotated
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

Coordinate = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
NonEmptyString = Annotated[str, Field(min_length=1)]


class GPSHealthStatus(IntEnum):
    UNHEALTHY = 0
    HEALTHY = 1


class ObjectClass(StrEnum):
    TASIT = "tasit"
    INSAN = "insan"
    UAP = "uap"
    UAI = "uai"


class MotionStatus(StrEnum):
    UNKNOWN = "unknown"
    STATIONARY = "stationary"
    MOVING = "moving"


class LandingStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    UNSUITABLE = "unsuitable"
    SUITABLE = "suitable"


class ImageModality(StrEnum):
    RGB = "rgb"
    THERMAL = "thermal"
    UNKNOWN = "unknown"


def _validate_image_source(value: str) -> str:
    source = value.strip()
    if not source:
        raise ValueError("Görüntü kaynağı boş olamaz.")
    parsed = urlparse(source)
    is_windows_path = bool(PureWindowsPath(source).drive)
    if parsed.scheme not in {"", "file", "http", "https"} and not is_windows_path:
        raise ValueError("Görüntü kaynağı yerel yol, file, http veya https olmalıdır.")
    return source


class FrameRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    url: NonEmptyString
    image_url: NonEmptyString
    video_name: NonEmptyString
    session: NonEmptyString
    frame_index: int | None = Field(default=None, ge=0)
    image_modality: ImageModality | None = None
    translation_x: FiniteFloat | None = None
    translation_y: FiniteFloat | None = None
    translation_z: FiniteFloat | None = None
    gps_health_status: GPSHealthStatus | None = None

    _image_source = field_validator("image_url")(_validate_image_source)


class BoundingBoxCoordinates(BaseModel):
    """Canonical flattened bbox order used by every task result."""

    top_left_x: Coordinate
    top_left_y: Coordinate
    bottom_right_x: Coordinate
    bottom_right_y: Coordinate

    @model_validator(mode="after")
    def validate_coordinate_order(self) -> "BoundingBoxCoordinates":
        if self.bottom_right_x <= self.top_left_x:
            raise ValueError("bottom_right_x, top_left_x değerinden büyük olmalıdır.")
        if self.bottom_right_y <= self.top_left_y:
            raise ValueError("bottom_right_y, top_left_y değerinden büyük olmalıdır.")
        return self


class DetectedObject(BoundingBoxCoordinates):
    cls: ObjectClass
    landing_status: LandingStatus
    motion_status: MotionStatus
    confidence: Confidence | None = None


class DetectedTranslation(BaseModel):
    translation_x: FiniteFloat
    translation_y: FiniteFloat
    translation_z: FiniteFloat


class MatchedReferenceObject(BoundingBoxCoordinates):
    object_id: int = Field(gt=0)
    confidence: Confidence | None = None


class CompetitionResponse(BaseModel):
    id: int = Field(gt=0)
    user: str
    frame: NonEmptyString
    detected_objects: list[DetectedObject] = Field(default_factory=list)
    detected_translations: list[DetectedTranslation] = Field(default_factory=list)
    detected_undefined_objects: list[MatchedReferenceObject] = Field(default_factory=list)

    @classmethod
    def from_task_results(
        cls,
        *,
        response_id: int,
        user: str,
        frame: str,
        detected_objects: list[DetectedObject],
        detected_translation: DetectedTranslation | None,
        matched_reference_objects: list[MatchedReferenceObject],
    ) -> "CompetitionResponse":
        return cls(
            id=response_id,
            user=user,
            frame=frame,
            detected_objects=detected_objects,
            detected_translations=(
                [detected_translation] if detected_translation is not None else []
            ),
            detected_undefined_objects=matched_reference_objects,
        )


class ReferenceImageRequest(BaseModel):
    object_id: int = Field(gt=0)
    image_url: NonEmptyString
    active_from_frame: int | None = Field(default=None, ge=0)
    active_until_frame: int | None = Field(default=None, ge=0)
    modality: ImageModality | None = None

    _image_source = field_validator("image_url")(_validate_image_source)

    @model_validator(mode="after")
    def validate_active_range(self) -> "ReferenceImageRequest":
        if (
            self.active_from_frame is not None
            and self.active_until_frame is not None
            and self.active_until_frame < self.active_from_frame
        ):
            raise ValueError("active_until_frame, active_from_frame değerinden küçük olamaz.")
        return self


class ReferenceSetRequest(BaseModel):
    references: list[ReferenceImageRequest] = Field(default_factory=list)
    frame_modality: ImageModality | None = None


class ReferenceInfo(BaseModel):
    object_id: int = Field(gt=0)
    active_from_frame: int | None = Field(default=None, ge=0)
    active_until_frame: int | None = Field(default=None, ge=0)
    modality: ImageModality


class ReferenceListResponse(BaseModel):
    session: NonEmptyString
    frame_modality: ImageModality | None = None
    references: list[ReferenceInfo] = Field(default_factory=list)
