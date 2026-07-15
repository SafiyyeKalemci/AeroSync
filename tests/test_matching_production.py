from dataclasses import replace

import pytest

from app.core.config import get_settings
from app.schemas import ImageModality
from app.services.matching.pipeline import BoundingBoxValidator, PipelineKind, PipelineSelector
from app.utils.images import detect_image_format


def settings():
    return replace(
        get_settings(),
        matching_min_confidence=0.35,
        matching_min_bbox_area=64.0,
        matching_max_bbox_area_ratio=0.50,
    )


@pytest.mark.parametrize(
    ("reference_modality", "frame_modality", "expected"),
    [
        (ImageModality.RGB, ImageModality.RGB, PipelineKind.SAME_MODAL),
        (ImageModality.THERMAL, ImageModality.THERMAL, PipelineKind.SAME_MODAL),
        (ImageModality.RGB, ImageModality.THERMAL, PipelineKind.CROSS_MODAL),
        (ImageModality.THERMAL, ImageModality.RGB, PipelineKind.CROSS_MODAL),
    ],
)
def test_pipeline_selection(reference_modality, frame_modality, expected):
    assert PipelineSelector().select(reference_modality, frame_modality) is expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"\xff\xd8\xffjpeg", "jpeg"),
        (b"\x89PNG\r\n\x1a\npng", "png"),
        (b"RIFF\x04\x00\x00\x00WEBPdata", "webp"),
    ],
)
def test_supported_image_formats(content, expected):
    assert detect_image_format(content) == expected


def test_invalid_boxes_are_rejected():
    validator = BoundingBoxValidator(settings())
    common = {"object_id": 1, "image_width": 100, "image_height": 100}
    assert validator.validate(
        raw_box={
            "top_left_x": 10,
            "top_left_y": 10,
            "bottom_right_x": 5,
            "bottom_right_y": 20,
            "confidence": 0.9,
        },
        **common,
    ) is None
    assert validator.validate(
        raw_box={
            "top_left_x": 0,
            "top_left_y": 0,
            "bottom_right_x": 100,
            "bottom_right_y": 100,
            "confidence": 0.9,
        },
        **common,
    ) is None
