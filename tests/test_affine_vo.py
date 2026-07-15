from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace

import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import GPSHealthStatus, ImageModality
from app.services.common import FrameContext
from app.services.localization import AffineLocalizationService
from app.services.localization.disabled_service import DisabledLocalizationService
from app.services.localization.affine_vo import AffineVOConfig, AffineVisualOdometry
from app.services.localization.camera_model import CameraModel, CameraModelProvider, load_camera_calibration
from app.services.localization.interface import LocalizationSessionState
from app.services.localization.session_store import LocalizationSessionStore
from app.services.localization.state import AffineMotionResult
from app.services.registry import build_services


def settings(**changes):
    values = {
        "localization_enabled": True,
        "localization_vo_enabled": True,
        "localization_min_features": 8,
        "localization_max_features": 100,
        "localization_feature_quality_level": 0.01,
        "localization_feature_min_distance": 2.0,
        "localization_lk_win_size": 15,
        "localization_lk_max_level": 2,
        "localization_lk_fb_error_threshold": 0.5,
        "localization_ransac_iterations": 200,
        "localization_ransac_residual_threshold": 0.5,
        "localization_min_inliers": 6,
        "localization_min_inlier_ratio": 0.5,
        "localization_max_frame_gap": 1,
        "localization_warmup_frames": 1,
        "localization_freeze_threshold": 0.0,
        "localization_session_ttl_seconds": 60.0,
        "localization_max_sessions": 8,
        "localization_camera_width": 100,
        "localization_camera_height": 100,
        "localization_camera_fx": 80.0,
        "localization_camera_fy": 80.0,
        "localization_camera_cx": 50.0,
        "localization_camera_cy": 50.0,
        "localization_camera_distortion": "0,0,0,0,0",
        "localization_camera_calibration_path": None,
        "matching_enabled": False,
    }
    values.update(changes)
    return replace(get_settings(), **values)


def camera(width=100, height=100):
    return CameraModel(width, height, 80.0, 80.0, width / 2, height / 2, ())


def points(count=20):
    values = [(10 + (i % 5) * 16, 12 + (i // 5) * 18) for i in range(count)]
    return np.asarray(values, np.float32).reshape(-1, 1, 2)


def transformed(source, tx=0.0, ty=0.0, yaw=0.0, cx=50.0, cy=50.0):
    previous = source.reshape(-1, 2).astype(np.float32)
    current = previous.copy()
    current[:, 0] += -tx + yaw * (previous[:, 1] - cy)
    current[:, 1] += -ty - yaw * (previous[:, 0] - cx)
    return current.reshape(-1, 1, 2)


class TrackingDouble:
    def __init__(self, forward, backward=None, forward_status=None, backward_status=None):
        self.forward = forward
        self.backward = backward if backward is not None else None
        self.forward_status = forward_status
        self.backward_status = backward_status
        self.calls = 0

    def __call__(self, _source, _target, source_points):
        self.calls += 1
        if self.calls == 1:
            result = self.forward
            status = self.forward_status
        else:
            result = self.backward if self.backward is not None else source_points
            status = self.backward_status
        if status is None:
            status = np.ones((len(result), 1), np.uint8)
        return result.copy(), status, np.zeros((len(result), 1), np.float32)


def vo_for(source, current, **changes):
    options = dict(
        min_features=8, max_features=100, feature_quality_level=0.01,
        feature_min_distance=2.0, lk_win_size=15, lk_max_level=2,
        lk_fb_error_threshold=0.5, ransac_iterations=200,
        ransac_residual_threshold=0.5, min_inliers=6,
        min_inlier_ratio=0.5, freeze_threshold=0.0,
    )
    options.update(changes)
    tracker = TrackingDouble(current, source)
    return AffineVisualOdometry(
        AffineVOConfig(**options),
        feature_detector=lambda _gray: source.copy(),
        tracker=tracker,
    ), tracker


@pytest.mark.parametrize(
    "tx,ty,yaw",
    [(4.0, -3.0, 0.0), (0.0, 0.0, 0.015), (3.0, 2.0, -0.01)],
)
def test_pure_translation_yaw_and_combined_motion(tx, ty, yaw):
    source = points()
    current = transformed(source, tx, ty, yaw)
    vo, _ = vo_for(source, current)
    result = vo.estimate(np.zeros((100, 100), np.uint8), np.ones((100, 100), np.uint8), camera())
    assert result.quality_valid
    assert result.delta_x_px == pytest.approx(tx, abs=1e-4)
    assert result.delta_y_px == pytest.approx(ty, abs=1e-4)
    assert result.delta_yaw_rad == pytest.approx(yaw, abs=1e-5)
    assert result.inlier_count == len(source)
    assert result.rms_residual == pytest.approx(0.0, abs=1e-5)


def test_forward_backward_filtering_discards_bad_tracks():
    source = points()
    current = transformed(source, 2, 1, 0)
    backward = source.copy()
    backward[:7, 0, 0] += 4
    tracker = TrackingDouble(current, backward)
    vo = AffineVisualOdometry(
        AffineVOConfig(8, 100, 0.01, 2, 15, 2, 0.5, 100, 0.5, 6, 0.5, 0),
        feature_detector=lambda _gray: source.copy(), tracker=tracker,
    )
    result = vo.estimate(np.zeros((100, 100), np.uint8), np.ones((100, 100), np.uint8), camera())
    assert result.quality_valid
    assert result.tracked_points == 13


def test_nan_out_of_bounds_and_insufficient_features_are_rejected():
    source = points(10)
    current = transformed(source, 1, 1, 0)
    current[0, 0] = np.nan
    current[1, 0] = [150, 150]
    vo, _ = vo_for(source, current, min_features=9)
    result = vo.estimate(np.zeros((100, 100), np.uint8), np.ones((100, 100), np.uint8), camera())
    assert not result.quality_valid
    assert result.failure_reason == "insufficient_features"


def test_low_inlier_ratio_is_invalid():
    source = points(30)
    current = transformed(source, 2, 1, 0)
    current[10:, 0, 0] += np.linspace(-15, 15, 20)
    current[10:, 0, 1] += np.linspace(14, -14, 20)
    vo, _ = vo_for(source, current, min_inlier_ratio=0.8)
    result = vo.estimate(np.zeros((100, 100), np.uint8), np.ones((100, 100), np.uint8), camera())
    assert not result.quality_valid
    assert result.failure_reason == "low_quality"


def test_high_residual_does_not_become_successful_zero_motion():
    source = points()
    rng = np.random.default_rng(7)
    current = source + rng.normal(0, 4, source.shape).astype(np.float32)
    vo, _ = vo_for(source, current, ransac_residual_threshold=0.01)
    result = vo.estimate(np.zeros((100, 100), np.uint8), np.ones((100, 100), np.uint8), camera())
    assert not result.quality_valid
    assert result.delta_x_px is None
    assert result.failure_reason in {"ransac_failed", "low_quality"}


def test_freeze_is_invalid_not_successful_zero_motion():
    source = points()
    vo, _ = vo_for(source, source)
    result = vo.estimate(np.zeros((100, 100), np.uint8), np.zeros((100, 100), np.uint8), camera())
    assert result.failure_reason == "freeze"
    assert not result.quality_valid


class AnyResolutionCamera:
    def for_resolution(self, width, height):
        return CameraModel(width, height, 80, 80, width / 2, height / 2, ())


class VOStub:
    def __init__(self, result=None):
        self.result = result or AffineMotionResult(2, 1, 0.01, 20, 20, 1, 0, True, None)
        self.calls = 0

    def estimate(self, _previous, _current, _camera):
        self.calls += 1
        return self.result


def frame(frame_id="f1", index=1, session="s", video="v", modality=ImageModality.RGB, health=GPSHealthStatus.UNHEALTHY, source="a"):
    return FrameContext(frame_id, source, video, session, health, 1.0, 2.0, 3.0, index, modality)


def service(images, vo=None, **changes):
    async def reader(source, _timeout):
        value = images[source]
        if isinstance(value, Exception):
            raise value
        return source.encode()

    return AffineLocalizationService(
        settings(**changes), vo=vo or VOStub(), camera_provider=AnyResolutionCamera(),
        image_reader=reader, image_decoder=lambda content: images[content.decode()].copy(),
    )


@pytest.mark.asyncio
async def test_first_frame_is_baseline_and_second_frame_updates_motion_but_returns_none():
    state = LocalizationSessionState("s")
    vo = VOStub()
    localization = service({"a": np.zeros((100, 100, 3), np.uint8), "b": np.ones((100, 100, 3), np.uint8)}, vo)
    assert await localization.process_frame(frame(), state) is None
    assert state.vo_state.previous_frame_id == "f1"
    assert state.vo_state.last_motion_result is None
    assert await localization.process_frame(frame("f2", 2, source="b"), state) is None
    assert state.vo_state.last_motion_result.quality_valid
    assert state.vo_state.cumulative_dx_px != 0
    assert vo.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second,event",
    [
        (frame("f2", 1, source="b"), "out_of_order"),
        (frame("f2", 5, source="b"), "gap"),
        (frame("f2", None, source="b"), "missing_index"),
        (frame("f2", 2, video="v2", source="b"), "video"),
    ],
)
async def test_discontinuities_reset_baseline(second, event):
    state = LocalizationSessionState("s")
    vo = VOStub()
    localization = service({"a": np.zeros((100, 100, 3), np.uint8), "b": np.ones((100, 100, 3), np.uint8)}, vo)
    await localization.process_frame(frame(), state)
    assert await localization.process_frame(second, state) is None
    assert state.vo_state.previous_frame_id == "f2"
    assert state.vo_state.last_motion_result is None
    assert vo.calls == 0


@pytest.mark.asyncio
async def test_resolution_change_resets_baseline():
    state = LocalizationSessionState("s")
    vo = VOStub()
    localization = service({"a": np.zeros((100, 100, 3), np.uint8), "b": np.ones((80, 120, 3), np.uint8)}, vo)
    await localization.process_frame(frame(), state)
    await localization.process_frame(frame("f2", 2, source="b"), state)
    assert state.vo_state.image_shape == (80, 120)
    assert vo.calls == 0


@pytest.mark.asyncio
async def test_unsupported_thermal_modality_resets_without_running_rgb_vo():
    state = LocalizationSessionState("s")
    vo = VOStub()
    localization = service({"a": np.zeros((100, 100, 3), np.uint8), "b": np.ones((100, 100, 3), np.uint8)}, vo)
    await localization.process_frame(frame(), state)
    await localization.process_frame(frame("f2", 2, modality=ImageModality.THERMAL, source="b"), state)
    assert state.vo_state.previous_gray is None
    assert vo.calls == 0


@pytest.mark.asyncio
async def test_duplicate_frame_does_not_advance_state():
    state = LocalizationSessionState("s")
    vo = VOStub()
    localization = service({"a": np.zeros((100, 100, 3), np.uint8)}, vo)
    await localization.process_frame(frame(), state)
    count = state.frame_count
    await localization.process_frame(frame(), state)
    assert state.frame_count == count
    assert vo.calls == 0


@pytest.mark.asyncio
async def test_same_image_with_new_id_is_freeze_not_valid_motion():
    state = LocalizationSessionState("s")
    vo = VOStub()
    localization = service({"a": np.zeros((100, 100, 3), np.uint8)}, vo)
    await localization.process_frame(frame(), state)
    await localization.process_frame(frame("f2", 2), state)
    assert state.vo_state.freeze_detected
    assert state.vo_state.last_motion_result.failure_reason == "freeze"
    assert vo.calls == 0


@pytest.mark.asyncio
async def test_corrupt_image_does_not_damage_previous_state():
    state = LocalizationSessionState("s")
    localization = service({"a": np.zeros((100, 100, 3), np.uint8), "bad": ValueError("corrupt")})
    await localization.process_frame(frame(), state)
    previous = state.vo_state
    assert await localization.process_frame(frame("f2", 2, source="bad"), state) is None
    assert state.vo_state is previous
    assert state.vo_state.previous_frame_id == "f1"


@pytest.mark.asyncio
@pytest.mark.parametrize("health", [GPSHealthStatus.HEALTHY, GPSHealthStatus.UNHEALTHY])
async def test_gps_health_does_not_stop_vo_and_only_healthy_ground_truth_is_returned(health):
    state = LocalizationSessionState("s")
    vo = VOStub()
    localization = service({"a": np.zeros((100, 100, 3), np.uint8), "b": np.ones((100, 100, 3), np.uint8)}, vo)
    first = await localization.process_frame(frame(health=health), state)
    second = await localization.process_frame(frame("f2", 2, health=health, source="b"), state)
    if health is GPSHealthStatus.HEALTHY:
        assert (first.translation_x, second.translation_y) == (1.0, 2.0)
    else:
        assert first is None and second is None
    assert state.last_gps_health_status is health
    assert state.vo_state.last_motion_result.quality_valid


@pytest.mark.asyncio
async def test_sessions_are_isolated_and_resettable():
    store = LocalizationSessionStore(ttl_seconds=60, max_sessions=4)
    async with store.locked("a") as first:
        first.frame_count = 10
    async with store.locked("b") as second:
        assert second.frame_count == 0
    await store.reset("a")
    async with store.locked("a") as reset:
        assert reset.frame_count == 0


@pytest.mark.asyncio
async def test_same_session_lock_serializes_and_different_sessions_overlap():
    store = LocalizationSessionStore(ttl_seconds=60, max_sessions=4)
    active = 0
    maximum = 0

    async def work(session):
        nonlocal active, maximum
        async with store.locked(session):
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            active -= 1

    await asyncio.gather(work("same"), work("same"))
    assert maximum == 1
    maximum = 0
    await asyncio.gather(work("a"), work("b"))
    assert maximum == 2


@pytest.mark.asyncio
async def test_cancelled_worker_cannot_commit_late_result():
    started = threading.Event()
    release = threading.Event()

    class SlowVO(VOStub):
        def estimate(self, *_args):
            started.set()
            release.wait(2)
            return self.result

    state = LocalizationSessionState("s")
    localization = service({"a": np.zeros((100, 100, 3), np.uint8), "b": np.ones((100, 100, 3), np.uint8)}, SlowVO())
    await localization.process_frame(frame(), state)
    task = asyncio.create_task(localization.process_frame(frame("f2", 2, source="b"), state))
    await asyncio.to_thread(started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.sleep(0.03)
    assert state.vo_state.previous_frame_id == "f1"
    assert state.vo_state.last_motion_result is None


def test_camera_json_loading_and_unknown_resolution(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({"width": 100, "height": 80, "fx": 70, "fy": 71, "cx": 50, "cy": 40, "distortion": [0.1, -0.1]}), encoding="utf-8")
    model = load_camera_calibration(path)
    assert model.width == 100 and model.distortion == (0.1, -0.1)
    provider = CameraModelProvider(settings(localization_camera_calibration_path=path))
    assert provider.for_resolution(100, 80) is not None
    assert provider.for_resolution(99, 80) is None


def test_registry_selects_real_vo_only_when_both_switches_are_enabled():
    enabled = build_services(settings())
    disabled = build_services(settings(localization_vo_enabled=False))
    assert isinstance(enabled.localization, AffineLocalizationService)
    assert isinstance(disabled.localization, DisabledLocalizationService)


@pytest.mark.parametrize("change", [
    {"localization_min_features": 2},
    {"localization_max_features": 5, "localization_min_features": 8},
    {"localization_feature_quality_level": 0},
    {"localization_lk_win_size": 4},
    {"localization_lk_fb_error_threshold": -1},
    {"localization_ransac_iterations": 0},
    {"localization_min_inlier_ratio": 0},
    {"localization_max_frame_gap": 0},
    {"localization_session_ttl_seconds": 0},
    {"localization_camera_fx": 0},
    {"localization_camera_distortion": "bad"},
])
def test_localization_config_validation(change):
    with pytest.raises(ValueError):
        settings(**change).validate_localization_vo()
