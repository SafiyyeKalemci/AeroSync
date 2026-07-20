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
from app.schemas import ImageModality
from app.services.common import FrameContext
from app.services.matching.descriptor_types import DenseDescriptorSet, DescriptorMetrics
from app.services.matching.dinov2_runtime import (
    Dinov2ConfigurationError,
    Dinov2CudaOutOfMemory,
    Dinov2DescriptorError,
    Dinov2DescriptorRuntime,
    Dinov2RuntimeRegistry,
)
from app.services.matching.interface import ReferenceImage
from app.services.matching.service import DinoReferenceMatchingService
from competition.preflight_check import Report, Status, _check_tasks


def settings(tmp_path, **changes):
    repo = tmp_path / "dinov2"
    repo.mkdir(exist_ok=True)
    (repo / "hubconf.py").write_text("def dinov2_vitb14():\n    return None\n", encoding="utf-8")
    weights = tmp_path / "dinov2.pth"
    weights.write_bytes(b"local-weights")
    values = {
        "matching_enabled": True,
        "matching_dinov2_enabled": True,
        "matching_dinov2_repo_path": repo,
        "matching_dinov2_weights_path": weights,
        "matching_dinov2_device": "cpu",
        "matching_dinov2_max_long_edge": 42,
        "matching_dinov2_patch_size": 14,
        "matching_dinov2_timeout_seconds": 1.0,
        "matching_reference_hash_enabled": True,
        "matching_preload_models": False,
        "matching_warmup_enabled": False,
    }
    values.update(changes)
    return replace(get_settings(), **values)


class FakeModel:
    def __init__(self, *, invalid_shape=False, non_finite=False):
        self.calls = 0
        self.invalid_shape = invalid_shape
        self.non_finite = non_finite

    def forward_features(self, tensor):
        self.calls += 1
        count = (tensor.shape[2] // 14) * (tensor.shape[3] // 14)
        if self.invalid_shape:
            count += 1
        data = torch.arange(count * 8, dtype=torch.float32).reshape(1, count, 8)
        if self.non_finite:
            data[0, 0, 0] = float("nan")
        return {"x_norm_patchtokens": data}


def image(width=42, height=28, color=(10, 20, 30)):
    value = np.zeros((height, width, 3), dtype=np.uint8)
    value[:, :] = color
    return value


def png_bytes(value=None):
    ok, encoded = cv2.imencode(".png", image() if value is None else value)
    assert ok
    return encoded.tobytes()


def frame(session="s", index=5):
    return FrameContext(
        frame_id=f"frame-{index}", image_url="frame.png", video_name="v",
        session_id=session, gps_health_status=0, gps_x=None, gps_y=None, gps_z=None,
        frame_index=index, image_modality=ImageModality.RGB,
    )


def reference(object_id=1, content=None, start=0, end=10):
    return ReferenceImage(
        object_id=object_id, content=content or png_bytes(), active_from_frame=start,
        active_until_frame=end, modality=ImageModality.RGB,
    )


def runtime(tmp_path, model=None, **changes):
    loaded = []
    fake_model = model or FakeModel()

    def loader(torch_module, repo, weights, model_name, device):
        loaded.append((repo, weights, model_name, device))
        return fake_model, hashlib.sha256(b"local-weights").hexdigest()

    return Dinov2DescriptorRuntime(settings(tmp_path, **changes), model_loader=loader), fake_model, loaded


def test_runtime_is_lazy_and_loads_model_only_once(tmp_path):
    subject, model, loads = runtime(tmp_path)
    assert subject.is_loaded is False
    first, _ = subject.extract(image(), "a")
    second, _ = subject.extract(image(), "b")
    assert len(loads) == 1
    assert model.calls == 2
    assert first.shape == second.shape == (6, 8)


def test_preprocessing_preserves_aspect_and_snaps_to_patch_grid(tmp_path):
    subject, _, _ = runtime(tmp_path)
    subject.model_hash
    prepared = subject.preprocess(image(width=100, height=50))
    assert (prepared.resized_width, prepared.resized_height) == (42, 14)
    assert (prepared.grid_width, prepared.grid_height) == (3, 1)
    assert prepared.tensor.shape == (1, 3, 14, 42)


def test_bgr_is_converted_to_rgb_before_imagenet_normalization(tmp_path):
    subject, _, _ = runtime(tmp_path)
    subject.model_hash
    prepared = subject.preprocess(image(color=(0, 0, 255)))
    pixel = prepared.tensor[0, :, 0, 0]
    assert pixel[0] > pixel[2]


def test_dense_descriptors_are_l2_normalized_and_finite(tmp_path):
    subject, _, _ = runtime(tmp_path)
    descriptor, metrics = subject.extract(image(), "hash")
    norms = torch.linalg.vector_norm(descriptor.descriptors, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert metrics.descriptor_count == 6
    assert descriptor.source_hash == "hash"


def test_cpu_float16_request_falls_back_to_float32(tmp_path):
    subject, _, _ = runtime(tmp_path, matching_dinov2_descriptor_dtype="float16")
    descriptor, _ = subject.extract(image(), "hash")
    assert descriptor.dtype == "float32"


def test_runtime_registry_shares_instance_for_same_configuration(tmp_path):
    configured = settings(tmp_path)
    Dinov2RuntimeRegistry.clear_for_tests()
    try:
        assert Dinov2RuntimeRegistry.get(configured) is Dinov2RuntimeRegistry.get(configured)
    finally:
        Dinov2RuntimeRegistry.clear_for_tests()


@pytest.mark.parametrize("model", [FakeModel(invalid_shape=True), FakeModel(non_finite=True)])
def test_invalid_descriptor_output_is_rejected(tmp_path, model):
    subject, _, _ = runtime(tmp_path, model=model)
    with pytest.raises(Dinov2DescriptorError):
        subject.extract(image(), "hash")


def test_missing_local_artifact_fails_without_calling_loader(tmp_path):
    subject, _, loads = runtime(tmp_path)
    subject._settings = replace(subject._settings, matching_dinov2_weights_path=tmp_path / "missing.pth")
    with pytest.raises(Dinov2ConfigurationError):
        subject.extract(image(), "hash")
    assert loads == []


def test_missing_local_repository_fails_without_calling_loader(tmp_path):
    subject, _, loads = runtime(tmp_path)
    subject._settings = replace(subject._settings, matching_dinov2_repo_path=tmp_path / "missing")
    with pytest.raises(Dinov2ConfigurationError):
        subject.extract(image(), "hash")
    assert loads == []


def test_cuda_request_falls_back_to_cpu_when_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    subject, _, loads = runtime(tmp_path, matching_dinov2_device="cuda",
                                matching_dinov2_allow_cpu_fallback=True)
    subject.extract(image(), "hash")
    assert loads[0][3] == "cpu"


def test_cuda_request_fails_when_cpu_fallback_is_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    subject, _, loads = runtime(tmp_path, matching_dinov2_device="cuda",
                                matching_dinov2_allow_cpu_fallback=False)
    with pytest.raises(Dinov2ConfigurationError):
        subject.extract(image(), "hash")
    assert loads == []


def test_image_smaller_than_patch_is_rejected(tmp_path):
    subject, _, _ = runtime(tmp_path)
    with pytest.raises(Dinov2DescriptorError):
        subject.extract(image(width=13, height=13), "hash")


class FakeRuntime:
    def __init__(self, *, delay=0.0):
        self.model_hash = "model-hash"
        self.calls = 0
        self.delay = delay
        self.started = threading.Event()

    def extract(self, value, source_hash):
        self.calls += 1
        self.started.set()
        if self.delay:
            time.sleep(self.delay)
        descriptors = torch.ones((6, 4), dtype=torch.float32)
        result = DenseDescriptorSet(
            descriptors=descriptors, grid_width=3, grid_height=2, descriptor_dim=4,
            image_width=42, image_height=28, resized_width=42, resized_height=28,
            patch_size=14, scale_x=1.0, scale_y=1.0, device="cpu", dtype="float32",
            source_hash=source_hash,
        )
        return result, DescriptorMetrics(0.001, 0.002, 6, 4, result.nbytes)


class OomRuntime(FakeRuntime):
    def extract(self, value, source_hash):
        raise Dinov2CudaOutOfMemory("CUDA out of memory")


@pytest.mark.asyncio
async def test_service_caches_reference_and_runs_frame_once_per_request(tmp_path):
    fake = FakeRuntime()

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(tmp_path), runtime_factory=lambda _: fake, image_reader=reader
    )
    await service.set_references("s", [reference()])
    assert await service.process_frame(frame()) == []
    assert await service.process_frame(frame(index=6)) == []
    assert fake.calls == 3  # one cached reference + one forward for each frame
    state = (await service.get_reference_states("s"))[0]
    assert state.descriptor_ready is True
    assert state.embedding_ready is True
    assert state.descriptor_shape == (6, 4)


@pytest.mark.asyncio
async def test_inactive_reference_causes_no_runtime_or_frame_read(tmp_path):
    calls = []

    async def reader(source, timeout):
        calls.append(source)
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(tmp_path), runtime_factory=lambda _: (_ for _ in ()).throw(AssertionError()),
        image_reader=reader,
    )
    await service.set_references("s", [reference(start=10, end=20)])
    assert await service.process_frame(frame(index=1)) == []
    assert calls == []


@pytest.mark.asyncio
async def test_concurrent_frames_share_same_reference_descriptor(tmp_path):
    fake = FakeRuntime(delay=0.02)

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(tmp_path), runtime_factory=lambda _: fake, image_reader=reader
    )
    await service.set_references("s", [reference()])
    assert await asyncio.gather(service.process_frame(frame()), service.process_frame(frame(index=6))) == [[], []]
    assert fake.calls == 3


@pytest.mark.asyncio
async def test_session_reset_during_forward_discards_stale_descriptor(tmp_path):
    fake = FakeRuntime(delay=0.08)

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(tmp_path), runtime_factory=lambda _: fake, image_reader=reader
    )
    await service.set_references("s", [reference()])
    task = asyncio.create_task(service.process_frame(frame()))
    await asyncio.to_thread(fake.started.wait, 0.5)
    await service.clear_session("s")
    assert await task == []
    assert await service.get_reference_states("s") == ()


@pytest.mark.asyncio
async def test_descriptor_timeout_is_safe_and_records_error(tmp_path):
    fake = FakeRuntime(delay=0.08)

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(tmp_path, matching_dinov2_timeout_seconds=0.01),
        runtime_factory=lambda _: fake, image_reader=reader,
    )
    await service.set_references("s", [reference()])
    assert await service.process_frame(frame()) == []
    state = (await service.get_reference_states("s"))[0]
    assert state.descriptor_ready is False
    assert state.descriptor_error == "timeout"


@pytest.mark.asyncio
async def test_reference_cache_lru_limit_evicts_old_descriptor(tmp_path):
    fake = FakeRuntime()

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(tmp_path, matching_dinov2_max_cached_references=1),
        runtime_factory=lambda _: fake, image_reader=reader,
    )
    await service.set_references("s", [reference(1), reference(2)])
    assert await service.process_frame(frame()) == []
    states = await service.get_reference_states("s")
    assert sum(item.descriptor_ready for item in states) == 1


@pytest.mark.asyncio
async def test_multiple_active_references_still_decode_and_forward_frame_once(tmp_path):
    fake = FakeRuntime()
    reads = 0

    async def reader(source, timeout):
        nonlocal reads
        reads += 1
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(tmp_path), runtime_factory=lambda _: fake, image_reader=reader
    )
    await service.set_references("s", [reference(1), reference(2), reference(3)])
    assert await service.process_frame(frame()) == []
    assert fake.calls == 4  # three reference forwards and exactly one frame forward
    assert reads == 1


@pytest.mark.asyncio
async def test_changed_reference_image_invalidates_descriptor(tmp_path):
    fake = FakeRuntime()

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(tmp_path), runtime_factory=lambda _: fake, image_reader=reader
    )
    await service.set_references("s", [reference(content=png_bytes(image(color=(1, 2, 3))))])
    await service.process_frame(frame())
    await service.set_references("s", [reference(content=png_bytes(image(color=(4, 5, 6))))])
    await service.process_frame(frame(index=6))
    assert fake.calls == 4  # reference and frame are regenerated


@pytest.mark.asyncio
async def test_changed_model_hash_invalidates_reference_descriptor(tmp_path):
    fake = FakeRuntime()

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(tmp_path), runtime_factory=lambda _: fake, image_reader=reader
    )
    await service.set_references("s", [reference()])
    await service.process_frame(frame())
    fake.model_hash = "different-model"
    await service.process_frame(frame(index=6))
    assert fake.calls == 4


@pytest.mark.asyncio
async def test_incremental_reference_add_only_generates_new_reference(tmp_path):
    fake = FakeRuntime()

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(tmp_path), runtime_factory=lambda _: fake, image_reader=reader
    )
    first = reference(1)
    await service.set_references("s", [first])
    await service.process_frame(frame())
    await service.set_references("s", [first, reference(2)])
    await service.process_frame(frame(index=6))
    assert fake.calls == 4  # ref1 + frame, then only ref2 + frame


@pytest.mark.asyncio
async def test_removal_and_session_isolation_clear_only_owned_descriptors(tmp_path):
    fake = FakeRuntime()

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(tmp_path), runtime_factory=lambda _: fake, image_reader=reader
    )
    await service.set_references("a", [reference(1), reference(2)])
    await service.set_references("b", [reference(3)])
    await service.process_frame(frame("a"))
    await service.process_frame(frame("b"))
    assert await service.remove_reference("a", 1) is True
    assert [item.object_id for item in await service.get_reference_states("a")] == [2]
    assert (await service.get_reference_states("b"))[0].descriptor_ready is True


@pytest.mark.asyncio
async def test_cache_byte_limit_evicts_descriptor(tmp_path):
    fake = FakeRuntime()

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(tmp_path, matching_dinov2_max_cache_mb=0.00001),
        runtime_factory=lambda _: fake, image_reader=reader,
    )
    await service.set_references("s", [reference()])
    assert await service.process_frame(frame()) == []
    assert (await service.get_reference_states("s"))[0].descriptor_ready is False


@pytest.mark.asyncio
async def test_cuda_oom_returns_safe_empty_result(tmp_path):
    fake = OomRuntime()

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(
        settings(tmp_path), runtime_factory=lambda _: fake, image_reader=reader
    )
    await service.set_references("s", [reference()])
    assert await service.process_frame(frame()) == []
    state = (await service.get_reference_states("s"))[0]
    assert state.descriptor_ready is False
    assert state.descriptor_error == "cuda_out_of_memory"


@pytest.mark.asyncio
async def test_missing_model_artifact_never_creates_match_or_bbox(tmp_path):
    broken = settings(tmp_path, matching_dinov2_weights_path=tmp_path / "missing.pth")

    async def reader(source, timeout):
        return png_bytes()

    service = DinoReferenceMatchingService(broken, image_reader=reader)
    await service.set_references("s", [reference()])
    assert await service.process_frame(frame()) == []
    assert (await service.get_reference_states("s"))[0].descriptor_ready is False


def test_preflight_treats_dinov2_as_required_and_other_matchers_as_optional(tmp_path):
    configured = settings(
        tmp_path,
        matching_aliked_weights_path=None,
        matching_lightglue_weights_path=None,
        matching_aliked_model_path=None,
        matching_lightglue_model_path=None,
        matching_xoftr_model_path=None,
        matching_geometry_method="dinov2",
    )
    report = Report(tmp_path)
    _check_tasks(report, configured, skip_models=False)
    task3 = {check.name: check for check in report.checks if check.section == "Task 3 Matching"}
    assert task3["DINOv2 local repository"].status is Status.OK
    assert task3["DINOv2 weights"].status is Status.OK
    assert task3["DINOv2 weights"].details["size_bytes"] > 0
    assert len(task3["DINOv2 weights"].details["sha256_short"]) == 12
    assert task3["ALIKED"].status is Status.WARNING
    assert task3["LightGlue"].status is Status.WARNING
    assert task3["XoFTR TorchScript (legacy)"].status is Status.WARNING


def test_preflight_fails_when_required_dinov2_artifacts_are_missing(tmp_path):
    configured = settings(
        tmp_path,
        matching_dinov2_repo_path=tmp_path / "missing-repo",
        matching_dinov2_weights_path=tmp_path / "missing.pth",
    )
    report = Report(tmp_path)
    _check_tasks(report, configured, skip_models=False)
    failures = [
        check.name for check in report.checks
        if check.section == "Task 3 Matching" and check.status is Status.FAIL
    ]
    assert "DINOv2 local repository" in failures
    assert "DINOv2 weights" in failures
