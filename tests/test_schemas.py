import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import (
    CompetitionResponse,
    DetectedObject,
    FrameRequest,
    GPSHealthStatus,
    LandingStatus,
    MatchedReferenceObject,
    MotionStatus,
    ObjectClass,
)


def test_gps_coordinates_may_be_null():
    request = FrameRequest(
        url="frame-1",
        image_url="frame.jpg",
        video_name="video",
        session="session",
        translation_x=None,
        translation_y=None,
        translation_z=None,
        gps_health_status=0,
    )
    assert request.translation_x is None
    assert request.translation_y is None
    assert request.translation_z is None
    assert request.gps_health_status is GPSHealthStatus.UNHEALTHY


def test_central_enums_serialize_to_stable_values():
    detected = DetectedObject(
        cls=ObjectClass.TASIT,
        landing_status=LandingStatus.NOT_APPLICABLE,
        motion_status=MotionStatus.MOVING,
        top_left_x=1,
        top_left_y=2,
        bottom_right_x=10,
        bottom_right_y=20,
        confidence=0.8,
    )
    data = detected.model_dump(mode="json")
    assert data["cls"] == "tasit"
    assert data["motion_status"] == "moving"
    assert data["landing_status"] == "not_applicable"


def test_bbox_coordinate_order_is_validated_for_every_result_type():
    with pytest.raises(ValidationError):
        MatchedReferenceObject(
            object_id=1,
            top_left_x=10,
            top_left_y=2,
            bottom_right_x=5,
            bottom_right_y=20,
        )


def test_invalid_gps_health_status_is_rejected():
    with pytest.raises(ValidationError):
        FrameRequest(
            url="frame-1",
            image_url="frame.jpg",
            video_name="video",
            session="session",
            gps_health_status=2,
        )


def test_disabled_response_example_matches_pydantic_serialization():
    response = CompetitionResponse.from_task_results(
        response_id=1,
        user="",
        frame="frame-000001",
        detected_objects=[],
        detected_translation=None,
        matched_reference_objects=[],
    )
    example_path = (
        Path(__file__).parents[1]
        / "docs"
        / "examples"
        / "process_frame_response_disabled.json"
    )
    expected = json.loads(example_path.read_text(encoding="utf-8"))
    assert response.model_dump(mode="json", exclude_none=True) == expected
    assert json.loads(response.model_dump_json(exclude_none=True)) == expected
