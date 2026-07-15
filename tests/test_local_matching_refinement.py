from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from app.core.config import get_settings
from app.schemas import MatchedReferenceObject
from app.services.matching.aliked_runtime import AlikedRuntime
from app.services.matching.descriptor_types import DenseDescriptorSet, DescriptorMetrics
from app.services.matching.lightglue_runtime import LightGlueRuntime
from app.services.matching.local_features import (
    LocalArtifactUnavailable,
    LocalFeatureMetrics,
    LocalFeatureSet,
    LocalMatchSet,
)
from app.services.matching.local_matcher import LocalRefinementPipeline, ReferenceLocalFeatureCache
from app.services.matching.service import DinoReferenceMatchingService, _SessionState


def settings(**changes):
    values = {
        "matching_enabled": True,
        "matching_dinov2_enabled": True,
        "matching_geometry_method": "hybrid",
        "matching_local_refinement_enabled": True,
        "matching_preload_models": False,
        "matching_warmup_enabled": False,
        "matching_local_fallback_to_dinov2": True,
        "matching_local_min_keypoints": 4,
        "matching_local_min_matches": 4,
        "matching_local_min_inliers": 4,
        "matching_local_min_inlier_ratio": 0.5,
        "matching_local_max_reprojection_error": 3.0,
        "matching_coarse_min_similarity": 0.3,
        "matching_coarse_min_correspondences": 4,
        "matching_coarse_max_correspondences": 64,
        "matching_coarse_spatial_dedup_radius_px": 0.0,
        "matching_homography_min_inliers": 4,
        "matching_homography_min_inlier_ratio": 0.5,
        "matching_geometry_min_projected_area_px": 4.0,
        "matching_geometry_max_frame_area_ratio": 1.0,
        "matching_geometry_min_visible_ratio": 0.2,
        "matching_geometry_min_edge_length_px": 1.0,
        "matching_bbox_min_width_px": 1.0,
        "matching_bbox_min_height_px": 1.0,
        "matching_bbox_min_area_px": 4.0,
        "matching_bbox_max_frame_area_ratio": 1.0,
        "matching_min_confidence": 0.0,
    }
    values.update(changes)
    return replace(get_settings(), **values)


def dense(source_hash="dense"):
    tensor = torch.eye(16)
    return DenseDescriptorSet(
        tensor, 4, 4, 16, 100, 100, 56, 56, 14, 0.56, 0.56,
        "cpu", "float32", source_hash,
    )


def features(source_hash: str, offset=(0.0, 0.0)):
    base = np.asarray(
        [[10, 10], [90, 10], [90, 90], [10, 90], [50, 20], [70, 60]],
        dtype=np.float32,
    )
    points = base + np.asarray(offset, dtype=np.float32)
    return LocalFeatureSet(
        points, np.eye(len(points), dtype=np.float32), np.full(len(points), 0.95, np.float32),
        200 if offset != (0.0, 0.0) else 100,
        200 if offset != (0.0, 0.0) else 100,
        "cpu", source_hash,
    )


class FakeAliked:
    def __init__(self):
        self.calls: list[str] = []
        self.model_hash = "aliked-hash"
        self.device = "cpu"

    def extract(self, image, source_hash):
        self.calls.append(source_hash)
        item = features(source_hash, (20.0, 30.0) if source_hash == "frame" else (0.0, 0.0))
        metrics = LocalFeatureMetrics(0.01, 0.02, item.keypoint_count, item.descriptor_dimension, image.shape[:2], "cpu")
        return item, metrics


class FakeLightGlue:
    def __init__(self, count=6):
        self.calls = 0
        self.count = count
        self.device = "cpu"

    def match(self, reference, frame):
        self.calls += 1
        count = min(self.count, reference.keypoint_count, frame.keypoint_count)
        scores = np.full(count, 0.95, np.float32)
        return LocalMatchSet(
            reference.keypoints[:count], frame.keypoints[:count], scores,
            count, float(scores.mean()) if count else 0.0,
            None if count else "no_local_matches",
        ), 0.03


def pipeline(aliked=None, lightglue=None, **changes):
    aliked = aliked or FakeAliked()
    lightglue = lightglue or FakeLightGlue()
    return (
        LocalRefinementPipeline(
            settings(**changes),
            aliked_factory=lambda _: aliked,
            lightglue_factory=lambda _: lightglue,
        ),
        aliked,
        lightglue,
    )


def run_local(target, *, method="hybrid", cache=None, ref_dense=None, frame_dense=None):
    return target.match_reference(
        method=method,
        object_id=7,
        reference_descriptor=ref_dense or dense("ref-dense"),
        frame_descriptor=frame_dense or dense("frame-dense"),
        reference_image=np.zeros((100, 100, 3), np.uint8),
        frame_image=np.zeros((200, 200, 3), np.uint8),
        reference_hash="ref",
        frame_hash="frame",
        cache=cache if cache is not None else ReferenceLocalFeatureCache(),
    )


def test_aliked_is_lazy_and_maps_keypoints_to_original_pixels(tmp_path):
    artifact = tmp_path / "aliked.ts"
    artifact.write_bytes(b"local")

    class Model:
        def eval(self): return self
        def __call__(self, _tensor):
            return {
                "keypoints": torch.tensor([[[5.0, 10.0], [20.0, 20.0], [40.0, 30.0], [60.0, 40.0]]]),
                "descriptors": torch.ones((1, 4, 8)),
                "scores": torch.ones((1, 4)),
            }

    runtime = AlikedRuntime(
        settings(matching_aliked_model_path=artifact, matching_dinov2_max_long_edge=100),
        torch_module=torch, model_loader=lambda *_: Model(),
    )
    assert not runtime.is_loaded
    result, metrics = runtime.extract(np.zeros((100, 200, 3), np.uint8), "image")
    assert runtime.is_loaded
    assert result.keypoint_count == 4
    assert result.keypoints[0].tolist() == pytest.approx([10.0, 20.0])
    assert metrics.runtime == "ALIKED TorchScript"


def test_lightglue_is_lazy_and_validates_one_to_one_matches(tmp_path):
    artifact = tmp_path / "lightglue.ts"
    artifact.write_bytes(b"local")

    class Model:
        def eval(self): return self
        def __call__(self, _left, _right):
            return {"matches": torch.tensor([[[0, 0], [1, 1], [2, 2], [3, 3]]]), "scores": torch.ones((1, 4))}

    runtime = LightGlueRuntime(
        settings(matching_lightglue_model_path=artifact),
        torch_module=torch, model_loader=lambda *_: Model(),
    )
    assert not runtime.is_loaded
    result, _ = runtime.match(features("left"), features("right"))
    assert runtime.is_loaded
    assert result.match_count == 4


def test_missing_artifact_is_offline_safe(tmp_path):
    runtime = AlikedRuntime(settings(matching_aliked_model_path=tmp_path / "missing.ts"))
    with pytest.raises(LocalArtifactUnavailable):
        _ = runtime.model_hash


def test_hybrid_success_and_reference_cache_without_frame_cache():
    target, aliked, lightglue = pipeline()
    cache = ReferenceLocalFeatureCache()
    first = run_local(target, cache=cache)
    second = run_local(target, cache=cache)
    assert first.matched is not None and second.matched is not None
    assert first.diagnostics.cache_status == "MISS"
    assert second.diagnostics.cache_status == "HIT"
    assert aliked.calls.count("ref") == 1
    assert aliked.calls.count("frame") == 2
    assert lightglue.calls == 2


def test_coarse_rejection_does_not_call_local_models():
    target, aliked, lightglue = pipeline()
    bad = dense("bad")
    bad = replace(bad, descriptors=torch.zeros_like(bad.descriptors))
    result = run_local(target, frame_dense=bad)
    assert result.matched is None
    assert aliked.calls == []
    assert lightglue.calls == 0


def test_local_only_skips_failed_coarse_gate_and_can_match():
    target, aliked, lightglue = pipeline()
    bad = replace(dense("bad"), descriptors=torch.zeros((16, 16)))
    result = run_local(target, method="aliked_lightglue", frame_dense=bad)
    assert result.matched is not None
    assert lightglue.calls == 1


def test_low_local_match_rejects_safely():
    target, _, _ = pipeline(lightglue=FakeLightGlue(count=3))
    result = run_local(target)
    assert result.matched is None
    assert result.diagnostics.reason == "low_local_matches"


def test_low_local_inlier_ratio_rejects_safely():
    class OutlierLightGlue(FakeLightGlue):
        def match(self, reference, frame):
            result, elapsed = super().match(reference, frame)
            bad = np.asarray(result.frame_points_px).copy()
            bad[2:] = np.asarray([[180, 5], [5, 180], [170, 170], [130, 15]], np.float32)
            return replace(result, frame_points_px=bad), elapsed

    target, _, _ = pipeline(
        lightglue=OutlierLightGlue(), matching_local_min_inlier_ratio=0.9
    )
    result = run_local(target)
    assert result.matched is None
    assert result.diagnostics.reason in {"low_inliers", "homography_not_found"}


def test_session_cache_clear_prevents_stale_reference_features():
    cache = ReferenceLocalFeatureCache()
    target, aliked, _ = pipeline()
    assert run_local(target, cache=cache).matched is not None
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0
    assert run_local(target, cache=cache).matched is not None
    assert aliked.calls.count("ref") == 2


def test_service_falls_back_only_when_local_artifact_is_unavailable():
    import cv2
    expected = MatchedReferenceObject(
        object_id=7, top_left_x=1, top_left_y=2, bottom_right_x=3, bottom_right_y=4,
        confidence=0.8,
    )

    class DinoPipeline:
        def match_reference(self, *_): return expected

    class MissingLocal:
        def match_reference(self, **_): raise LocalArtifactUnavailable("missing")

    service = DinoReferenceMatchingService(
        settings(), matching_pipeline=DinoPipeline(), local_pipeline=MissingLocal()
    )
    state = _SessionState("s", object())
    ok, encoded = cv2.imencode(".png", np.zeros((10, 10, 3), np.uint8))
    assert ok
    result = service._match_selected_geometry(
        object(), state, 7, dense(), dense(), encoded.tobytes(), np.zeros((10, 10, 3), np.uint8), "frame"
    )
    assert result == expected


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_local_refinement_timeout_must_be_finite_and_positive(value):
    configured = settings(matching_local_refinement_timeout_sec=value)
    with pytest.raises(ValueError, match="MATCHING_LOCAL_REFINEMENT_TIMEOUT_SEC"):
        configured.validate_matching_local()


def test_local_refinement_timeout_is_independent_from_coarse_and_reference_limits():
    service = DinoReferenceMatchingService(
        settings(
            matching_geometry_method="hybrid",
            matching_local_refinement_timeout_sec=5.0,
            matching_coarse_timeout_seconds=0.01,
            matching_reference_timeout_seconds=0.02,
        )
    )
    assert service._reference_match_timeout() == (5.0, "local_refinement")


def test_dinov2_timeout_policy_remains_unchanged():
    service = DinoReferenceMatchingService(
        settings(
            matching_geometry_method="dinov2",
            matching_local_refinement_timeout_sec=5.0,
            matching_coarse_timeout_seconds=0.25,
            matching_reference_timeout_seconds=0.5,
        )
    )
    assert service._reference_match_timeout() == (0.25, "dinov2_coarse")


def test_source_contains_no_network_or_prediction_submission_calls():
    root = __import__("pathlib").Path(__file__).parents[1]
    sources = "\n".join(
        (root / "app/services/matching" / name).read_text(encoding="utf-8")
        for name in ("aliked_runtime.py", "lightglue_runtime.py", "local_matcher.py")
    )
    for forbidden in ("torch.hub", "requests.post", "httpx.post", "send_prediction", "prediction/"):
        assert forbidden not in sources
