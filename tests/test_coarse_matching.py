from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch

from app.core.config import get_settings
from app.schemas import ImageModality, MatchedReferenceObject
from app.services.common import FrameContext
from app.services.matching.bbox_validator import ProjectedBoundingBoxValidator
from app.services.matching.coarse_matcher import CoarseMatcher, CoarseMatchingPipeline
from app.services.matching.descriptor_types import (
    CoarseMatchSet,
    DenseDescriptorSet,
    DescriptorMetrics,
    HomographyResult,
    ProjectedPolygon,
)
from app.services.matching.geometry import ConfidenceScorer, HomographyEstimator, ProjectedPolygonValidator
from app.services.matching.interface import ReferenceImage
from app.services.matching.service import DinoReferenceMatchingService


def settings(**changes):
    values = {
        "matching_enabled": True,
        "matching_dinov2_enabled": True,
        "matching_geometry_method": "dinov2",
        "matching_coarse_min_similarity": 0.4,
        "matching_coarse_min_correspondences": 4,
        "matching_coarse_max_correspondences": 64,
        "matching_coarse_spatial_dedup_radius_px": 0.0,
        "matching_homography_min_inliers": 4,
        "matching_homography_min_inlier_ratio": 0.5,
        "matching_homography_max_rms_reprojection_error": 2.0,
        "matching_geometry_min_projected_area_px": 16.0,
        "matching_geometry_max_frame_area_ratio": 0.9,
        "matching_geometry_min_visible_ratio": 0.25,
        "matching_geometry_min_edge_length_px": 2.0,
        "matching_geometry_max_aspect_ratio": 10.0,
        "matching_geometry_max_perspective_distortion": 8.0,
        "matching_bbox_min_width_px": 2.0,
        "matching_bbox_min_height_px": 2.0,
        "matching_bbox_min_area_px": 16.0,
        "matching_bbox_max_frame_area_ratio": 0.9,
        "matching_min_confidence": 0.2,
        "matching_coarse_timeout_seconds": 1.0,
        "matching_reference_timeout_seconds": 1.0,
        "matching_dinov2_timeout_seconds": 1.0,
        "matching_preload_models": False,
        "matching_warmup_enabled": False,
    }
    values.update(changes)
    return replace(get_settings(), **values)


def descriptor(
    tensor=None,
    *,
    grid_width=4,
    grid_height=4,
    image_width=56,
    image_height=56,
    resized_width=56,
    resized_height=56,
    scale_x=1.0,
    scale_y=1.0,
):
    count = grid_width * grid_height
    tensor = torch.eye(count, dtype=torch.float32) if tensor is None else tensor
    return DenseDescriptorSet(
        descriptors=tensor,
        grid_width=grid_width,
        grid_height=grid_height,
        descriptor_dim=int(tensor.shape[1]),
        image_width=image_width,
        image_height=image_height,
        resized_width=resized_width,
        resized_height=resized_height,
        patch_size=14,
        scale_x=scale_x,
        scale_y=scale_y,
        device="cpu",
        dtype="float32",
        source_hash="hash",
    )


def match_set(source, target, similarities=None):
    similarities = np.ones(len(source), dtype=np.float32) if similarities is None else similarities
    return CoarseMatchSet(
        np.asarray(source, np.float32), np.asarray(target, np.float32), similarities,
        len(source), float(np.mean(similarities)), float(np.median(similarities)),
        float(np.min(similarities)), float(np.max(similarities)), 0.25,
    )


def grid_points():
    return np.asarray(
        [[10, 10], [30, 10], [50, 10], [10, 30], [30, 30], [50, 30],
         [10, 50], [30, 50], [50, 50]], dtype=np.float32
    )


def test_perfect_mutual_match_and_patch_coordinate_mapping():
    result = CoarseMatcher(settings()).match(descriptor(), descriptor())
    assert result.failure_reason is None
    assert result.correspondence_count == 16
    assert np.allclose(result.reference_points_px[0], [7, 7])
    assert result.mean_similarity == pytest.approx(1.0)


def test_patch_coordinates_map_back_to_original_scale():
    scaled = descriptor(image_width=112, image_height=112, scale_x=0.5, scale_y=0.5)
    result = CoarseMatcher(settings()).match(scaled, scaled)
    assert np.allclose(result.reference_points_px[0], [14, 14])


def test_non_mutual_and_similarity_threshold_are_filtered():
    ref = descriptor()
    duplicated = torch.zeros_like(ref.descriptors)
    duplicated[:, 0] = 1.0
    result = CoarseMatcher(settings()).match(ref, descriptor(duplicated))
    assert result.failure_reason is not None
    high_threshold = CoarseMatcher(settings(matching_coarse_min_similarity=0.99))
    noisy = torch.eye(16) * 0.5 + torch.roll(torch.eye(16), 1, 1) * 0.6
    assert high_threshold.match(ref, descriptor(noisy)).failure_reason is not None


def test_descriptor_dimension_mismatch_and_nonfinite_are_rejected():
    mismatch = descriptor(torch.ones((16, 8)))
    assert CoarseMatcher(settings()).match(descriptor(), mismatch).failure_reason == "descriptor_dimension_mismatch"
    invalid = torch.eye(16)
    invalid[0, 0] = float("nan")
    assert CoarseMatcher(settings()).match(descriptor(invalid), descriptor()).failure_reason == "descriptor_non_finite"


def test_minimum_maximum_and_spatial_dedup_filters():
    capped = CoarseMatcher(settings(matching_coarse_max_correspondences=5)).match(descriptor(), descriptor())
    assert capped.correspondence_count == 5
    deduped = CoarseMatcher(
        settings(matching_coarse_spatial_dedup_radius_px=20.0)
    ).match(descriptor(), descriptor())
    assert deduped.correspondence_count < 16


@pytest.mark.parametrize(
    "matrix",
    [
        np.asarray([[1, 0, 12], [0, 1, 7], [0, 0, 1]], np.float64),
        np.asarray([[1.4, 0, 3], [0, 1.4, 5], [0, 0, 1]], np.float64),
        np.asarray([[1, 0.05, 4], [0.03, 1, 2], [0.0005, 0.0002, 1]], np.float64),
    ],
)
def test_known_translation_scale_and_perspective_homographies(matrix):
    source = grid_points()
    target = cv2.perspectiveTransform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    result = HomographyEstimator(settings()).estimate(match_set(source, target))
    assert result.valid is True
    assert result.inlier_count == len(source)
    assert result.rms_reprojection_error < 0.01


def test_homography_none_is_rejected(monkeypatch):
    monkeypatch.setattr(cv2, "findHomography", lambda *args, **kwargs: (None, None))
    points = grid_points()
    assert HomographyEstimator(settings()).estimate(match_set(points, points)).failure_reason == "homography_not_found"


@pytest.mark.parametrize(
    ("matrix", "reason"),
    [
        (np.eye(2), "homography_matrix_invalid"),
        (np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, 1]], float), "homography_singular"),
    ],
)
def test_invalid_or_singular_homography_is_rejected(monkeypatch, matrix, reason):
    points = grid_points()
    monkeypatch.setattr(cv2, "findHomography", lambda *args, **kwargs: (matrix, np.ones((len(points), 1))))
    assert HomographyEstimator(settings()).estimate(match_set(points, points)).failure_reason == reason


def test_low_inlier_count_and_ratio_are_rejected(monkeypatch):
    points = grid_points()
    mask = np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0], np.uint8)[:, None]
    monkeypatch.setattr(cv2, "findHomography", lambda *args, **kwargs: (np.eye(3), mask))
    assert HomographyEstimator(settings()).estimate(match_set(points, points)).failure_reason == "low_inliers"


def test_high_reprojection_error_is_rejected(monkeypatch):
    points = grid_points()
    monkeypatch.setattr(
        cv2, "findHomography", lambda *args, **kwargs: (np.eye(3), np.ones((len(points), 1)))
    )
    result = HomographyEstimator(settings(matching_homography_max_rms_reprojection_error=0.1)).estimate(
        match_set(points, points + np.asarray([5, 0], np.float32))
    )
    assert result.failure_reason == "high_reprojection_error"


def test_convex_and_partially_visible_polygons_are_valid():
    validator = ProjectedPolygonValidator(settings())
    assert validator.validate_points(
        [[10, 10], [40, 10], [40, 40], [10, 40]], frame_width=100, frame_height=100
    ).valid
    partial = validator.validate_points(
        [[-10, 10], [30, 10], [30, 40], [-10, 40]], frame_width=100, frame_height=100
    )
    assert partial.valid and partial.visible_ratio == pytest.approx(0.75)


@pytest.mark.parametrize(
    "points",
    [
        [[10, 10], [40, 40], [10, 40], [40, 10]],  # self-intersecting
        [[10, 10], [40, 10], [20, 20], [10, 40]],  # non-convex
        [[-50, 10], [-20, 10], [-20, 40], [-50, 40]],  # outside
        [[10, 10], [12, 10], [12, 12], [10, 12]],  # too small
        [[0, 0], [99, 0], [99, 99], [0, 99]],  # too large
        [[10, 10], [90, 10], [90, 13], [10, 13]],  # extreme aspect
    ],
)
def test_invalid_polygon_shapes_are_rejected(points):
    result = ProjectedPolygonValidator(settings()).validate_points(
        points, frame_width=100, frame_height=100
    )
    assert result.valid is False


def test_low_visible_ratio_is_rejected():
    result = ProjectedPolygonValidator(settings(matching_geometry_min_visible_ratio=0.8)).validate_points(
        [[-30, 10], [10, 10], [10, 40], [-30, 40]], frame_width=100, frame_height=100
    )
    assert result.valid is False


def test_bbox_clipping_and_invalid_bbox():
    validator = ProjectedBoundingBoxValidator(settings())
    partial = ProjectedPolygon(
        np.asarray([[-10, 10], [30, 10], [30, 40], [-10, 40]], np.float32),
        1200, 900, 0.75, True,
    )
    result = validator.validate(partial, frame_width=100, frame_height=100)
    assert result is not None and result.clipped_box == (0.0, 10.0, 30.0, 40.0)
    tiny = ProjectedPolygon(np.asarray([[1, 1], [2, 1], [2, 2], [1, 2]], np.float32), 1, 1, 1, True)
    assert validator.validate(tiny, frame_width=100, frame_height=100) is None


def test_confidence_is_derived_bounded_and_thresholded():
    matches = match_set(grid_points(), grid_points(), np.full(9, 0.8, np.float32))
    homography = HomographyResult(np.eye(3), np.ones(9, bool), 8, 8 / 9, 0.5, True)
    polygon = ProjectedPolygon(np.zeros((4, 2)), 100, 80, 0.8, True)
    quality = ConfidenceScorer(settings()).score(matches, homography, polygon)
    assert quality is not None and 0 <= quality.confidence <= 1
    assert ConfidenceScorer(settings(matching_min_confidence=0.99)).score(
        matches, homography, polygon
    ) is None


def test_complete_pipeline_produces_geometry_derived_bbox():
    result = CoarseMatchingPipeline(settings(
        matching_geometry_max_frame_area_ratio=1.0,
        matching_bbox_max_frame_area_ratio=1.0,
    )).match_reference(7, descriptor(), descriptor())
    assert result is not None
    assert result.object_id == 7
    assert result.top_left_x == pytest.approx(0, abs=1e-3)
    assert result.bottom_right_x == pytest.approx(56, abs=1e-3)
    assert result.confidence != pytest.approx(0.5)


def png_bytes():
    image = np.zeros((56, 56, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


class FakeRuntime:
    model_hash = "model"

    def __init__(self):
        self.calls = 0

    def extract(self, image, source_hash):
        self.calls += 1
        result = descriptor()
        return result, DescriptorMetrics(0, 0, result.shape[0], result.descriptor_dim, result.nbytes)


class FakePipeline:
    def __init__(self, failures=(), delay=0.0):
        self.failures = set(failures)
        self.delay = delay
        self.started = threading.Event()

    def match_reference(self, object_id, reference, frame):
        self.started.set()
        if self.delay:
            time.sleep(self.delay)
        if object_id in self.failures:
            raise RuntimeError("isolated")
        return MatchedReferenceObject(
            object_id=object_id, top_left_x=1, top_left_y=2,
            bottom_right_x=10, bottom_right_y=12,
            confidence=0.6 + object_id / 100,
        )


def reference(object_id, order):
    return ReferenceImage(
        object_id=object_id, content=png_bytes(), active_from_frame=0, active_until_frame=10,
        modality=ImageModality.RGB, order=order,
        official_reference_url=f"/references/{object_id}/",
        image_url=f"/media/reference-{object_id}.png",
        video_name="v",
    )


def frame(index=1):
    return FrameContext(
        frame_id=f"f-{index}", image_url="frame.png", video_name="v", session_id="s",
        gps_health_status=0, gps_x=None, gps_y=None, gps_z=None,
        frame_index=index, image_modality=ImageModality.RGB,
    )


@pytest.mark.asyncio
async def test_service_isolates_failure_supports_multiple_references_and_order():
    runtime = FakeRuntime()
    pipeline = FakePipeline(failures={1})

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(), runtime_factory=lambda _: runtime, image_reader=reader,
        matching_pipeline=pipeline,
    )
    await service.set_references("s", [reference(1, 2), reference(2, 1), reference(3, 3)])
    results = await service.process_frame(frame())
    assert [item.object_id for item in results] == [2, 3]
    assert runtime.calls == 4  # three references, one shared frame descriptor
    assert len({item.object_id for item in results}) == len(results)
    await service.process_frame(frame(2))
    assert runtime.calls == 5  # cached references, one new frame forward


@pytest.mark.asyncio
async def test_no_active_reference_means_no_model_call():
    runtime = FakeRuntime()
    service = DinoReferenceMatchingService(settings(), runtime_factory=lambda _: runtime)
    await service.set_references("s", [replace(reference(1, 1), active_from_frame=10, active_until_frame=20)])
    assert await service.process_frame(frame(1)) == []
    assert runtime.calls == 0


@pytest.mark.asyncio
async def test_reference_timeout_is_isolated():
    runtime = FakeRuntime()
    pipeline = FakePipeline(delay=0.08)

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(matching_coarse_timeout_seconds=0.01, matching_reference_timeout_seconds=0.01),
        runtime_factory=lambda _: runtime, image_reader=reader, matching_pipeline=pipeline,
    )
    await service.set_references("s", [reference(1, 1)])
    assert await service.process_frame(frame()) == []


@pytest.mark.asyncio
async def test_session_reset_during_matching_discards_late_result():
    runtime = FakeRuntime()
    pipeline = FakePipeline(delay=0.08)

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(), runtime_factory=lambda _: runtime, image_reader=reader,
        matching_pipeline=pipeline,
    )
    await service.set_references("s", [reference(1, 1)])
    task = asyncio.create_task(service.process_frame(frame()))
    await asyncio.to_thread(pipeline.started.wait, 0.5)
    await service.clear_session("s")
    assert await task == []
