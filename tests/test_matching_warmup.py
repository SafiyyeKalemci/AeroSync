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
from app.services.matching.descriptor_types import DenseDescriptorSet, DescriptorMetrics
from app.services.matching.interface import ReferenceImage
from app.services.matching.local_features import (
    LocalFeatureMetrics,
    LocalFeatureSet,
    LocalMatchSet,
    LocalRefinementDiagnostics,
)
from app.services.matching.local_matcher import LocalPipelineResult
from app.services.matching.service import DinoReferenceMatchingService
from app.services.matching.warmup import MatchingWarmupDiagnostics, MatchingWarmupService


def settings(**changes):
    values = {
        "matching_enabled": True,
        "matching_dinov2_enabled": True,
        "matching_geometry_method": "hybrid",
        "matching_local_refinement_enabled": True,
        "matching_preload_models": True,
        "matching_warmup_enabled": True,
        "matching_local_refinement_timeout_sec": 0.05,
        "matching_dinov2_timeout_seconds": 1.0,
        "matching_local_min_keypoints": 4,
        "matching_local_min_matches": 4,
        "matching_local_min_inliers": 4,
    }
    values.update(changes)
    return replace(get_settings(), **values)


def png_bytes(value: int = 0) -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((56, 56, 3), value, np.uint8))
    assert ok
    return encoded.tobytes()


def local_features(source_hash: str) -> LocalFeatureSet:
    points = np.asarray([[1, 1], [10, 1], [10, 10], [1, 10]], np.float32)
    return LocalFeatureSet(
        points,
        np.eye(4, dtype=np.float32),
        np.ones(4, np.float32),
        56,
        56,
        "cpu",
        source_hash,
    )


def dense(source_hash: str) -> DenseDescriptorSet:
    return DenseDescriptorSet(
        torch.eye(16), 4, 4, 16, 56, 56, 56, 56, 14, 1.0, 1.0,
        "cpu", "float32", source_hash,
    )


class FakeDino:
    def __init__(self):
        self.model_hash_calls = 0
        self.extract_calls = 0

    @property
    def model_hash(self):
        self.model_hash_calls += 1
        return "dino-model"

    def extract(self, image, source_hash):
        self.extract_calls += 1
        return dense(source_hash), DescriptorMetrics(0.0, 0.0, 16, 16, 1024)


class FakeAliked:
    def __init__(self):
        self.model_hash_calls = 0
        self.extract_calls = 0

    @property
    def model_hash(self):
        self.model_hash_calls += 1
        return "aliked-model"

    def extract(self, image, source_hash):
        self.extract_calls += 1
        return local_features(source_hash), LocalFeatureMetrics(0, 0, 4, 4, (56, 56), "cpu")


class FakeLightGlue:
    def __init__(self):
        self.device_calls = 0
        self.match_calls = 0

    @property
    def device(self):
        self.device_calls += 1
        return "cpu"

    def match(self, reference, frame):
        self.match_calls += 1
        points = reference.keypoints.copy()
        return LocalMatchSet(points, points, np.ones(4, np.float32), 4, 1.0), 0.0


def test_model_preload_and_dummy_warmup_run_once_thread_safely():
    dino, aliked, lightglue = FakeDino(), FakeAliked(), FakeLightGlue()
    warmup = MatchingWarmupService(
        settings(),
        dinov2_factory=lambda _: dino,
        aliked_factory=lambda _: aliked,
        lightglue_factory=lambda _: lightglue,
    )
    outputs = []
    threads = [threading.Thread(target=lambda: outputs.append(warmup.warmup())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(outputs) == 8
    assert all(item.model_preloaded and item.warmup_completed for item in outputs)
    assert dino.extract_calls == aliked.extract_calls == lightglue.match_calls == 1
    assert dino.model_hash_calls == aliked.model_hash_calls == lightglue.device_calls == 1


class FakeWarmup:
    def __init__(self, delay=0.0):
        self.calls = 0
        self.delay = delay

    def warmup(self):
        self.calls += 1
        time.sleep(self.delay)
        return MatchingWarmupDiagnostics(True, True, self.delay)


class FakeLocalPipeline:
    def __init__(self, prepare_delay=0.0, *, produce_match=False):
        self.prepare_delay = prepare_delay
        self.produce_match = produce_match
        self.prepared_hashes = []
        self.match_calls = 0

    def prepare_reference(self, image, reference_hash):
        time.sleep(self.prepare_delay)
        self.prepared_hashes.append(reference_hash)
        metrics = LocalFeatureMetrics(0, 0, 4, 4, image.shape[:2], "cpu")
        return "aliked-model", local_features(reference_hash), metrics

    def match_reference(self, **kwargs):
        self.match_calls += 1
        return LocalPipelineResult(
            (
                MatchedReferenceObject(
                    object_id=kwargs["object_id"], top_left_x=1, top_left_y=1,
                    bottom_right_x=10, bottom_right_y=10, confidence=0.9,
                )
                if self.produce_match else None
            ),
            LocalRefinementDiagnostics(
                method="hybrid", accepted=self.produce_match,
                reason="accepted" if self.produce_match else "controlled_rejection",
            ),
        )


@pytest.mark.asyncio
async def test_reference_cache_warmup_is_generation_safe_and_has_no_result_leakage():
    runtime, local, warmup = FakeDino(), FakeLocalPipeline(), FakeWarmup()
    service = DinoReferenceMatchingService(
        settings(), runtime_factory=lambda _: runtime, local_pipeline=local,
        warmup_service=warmup,
    )
    first = png_bytes(1)
    second = png_bytes(2)
    await service.set_references("s", [ReferenceImage(1, first, modality=ImageModality.RGB)])
    await service.set_references("s", [ReferenceImage(1, second, modality=ImageModality.RGB)])

    state = service._sessions["s"]
    second_hash = hashlib.sha256(second).hexdigest()
    assert len(state.local_feature_cache) == 1
    assert state.local_feature_cache.get(second_hash, "aliked-model") is not None
    assert state.local_feature_cache.get(hashlib.sha256(first).hexdigest(), "aliked-model") is None
    assert service.get_last_match_diagnostics("s", "any") == {}
    diagnostics = service.get_startup_diagnostics("s")
    assert diagnostics["model_preloaded"] is True
    assert diagnostics["warmup_completed"] is True
    assert diagnostics["reference_cache_warmed"] is True
    assert warmup.calls == 2


@pytest.mark.asyncio
async def test_frame_timeout_excludes_model_and_reference_preparation():
    content = png_bytes(3)
    runtime, local = FakeDino(), FakeLocalPipeline(prepare_delay=0.04, produce_match=True)
    service = DinoReferenceMatchingService(
        settings(matching_local_refinement_timeout_sec=0.05),
        runtime_factory=lambda _: runtime,
        local_pipeline=local,
        warmup_service=FakeWarmup(delay=0.04),
        image_reader=lambda *_: asyncio.sleep(0, result=content),
    )
    started = time.perf_counter()
    await service.set_references("s", [ReferenceImage(1, content, modality=ImageModality.RGB)])
    assert time.perf_counter() - started >= 0.07

    frame = FrameContext(
        frame_id="f1", image_url="local-only", video_name="v", session_id="s",
        gps_health_status=1, gps_x=None, gps_y=None, gps_z=None, frame_index=1,
        image_modality=ImageModality.RGB,
    )
    results = await service.process_frame(frame)
    assert [result.object_id for result in results] == [1]
    diagnostic = service.get_last_match_diagnostics("s", "f1")[1]
    assert diagnostic["outcome"] == "matched"
    assert diagnostic["frame_local_refinement_time_sec"] < 0.05
    assert diagnostic["warmup_time_sec"] == pytest.approx(0.04)
    assert diagnostic["reference_prepare_time_sec"] > 0.02
    assert (
        diagnostic["warmup_time_sec"] + diagnostic["reference_prepare_time_sec"]
        > 0.05
    )


@pytest.mark.asyncio
async def test_session_reset_discards_reference_warmup_cache():
    service = DinoReferenceMatchingService(
        settings(), runtime_factory=lambda _: FakeDino(),
        local_pipeline=FakeLocalPipeline(), warmup_service=FakeWarmup(),
    )
    await service.set_references("s", [ReferenceImage(1, png_bytes(), modality=ImageModality.RGB)])
    assert service.get_startup_diagnostics("s")["reference_cache_warmed"] is True
    await service.clear_session("s")
    assert service.get_startup_diagnostics("s")["reference_cache_warmed"] is False


def test_warmup_module_has_no_prediction_or_network_calls():
    source = (__import__("pathlib").Path(__file__).parents[1] / "app/services/matching/warmup.py").read_text()
    assert "send_prediction" not in source
    assert "prediction/" not in source
    assert "requests." not in source
