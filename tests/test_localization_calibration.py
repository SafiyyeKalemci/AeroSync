from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import GPSHealthStatus, ImageModality
from app.services.common import FrameContext
from app.services.localization.alignment import fit_calibration
from app.services.localization.calibration import (
    CalibrationPolicy,
    CalibrationResult,
    CalibrationSample,
    make_calibration_sample,
)
from app.services.localization.camera_model import CameraModel
from app.services.localization.estimator import estimate_translation
from app.services.localization.interface import LocalizationSessionState
from app.services.localization.service import AffineLocalizationService
from app.services.localization.session_store import LocalizationSessionStore
from app.services.localization.state import AffineMotionResult, VisualOdometryState


def policy(**changes):
    values = dict(
        min_samples=4,
        max_samples=20,
        min_camera_step_px=0.5,
        min_gps_step=0.01,
        max_rms_residual=0.05,
        min_inlier_ratio=0.7,
        min_directional_diversity=0.1,
        scale_min=0.001,
        scale_max=2.0,
        outlier_mad_factor=3.5,
        allow_reflection=False,
    )
    values.update(changes)
    return CalibrationPolicy(**values)


def sample(camera_delta, gps_delta, sequence=1):
    return CalibrationSample(
        frame_index=sequence,
        sequence=sequence,
        camera_delta_px_2d=tuple(camera_delta),
        gps_delta_xy=tuple(gps_delta),
        camera_motion_quality=0.95,
        gps_movement_magnitude=float(np.linalg.norm(gps_delta)),
        yaw_rad=0.0,
        timestamp=datetime.now(timezone.utc),
    )


CAMERA_STEPS = [(4, 0), (0, 5), (-3, 2), (2, 4), (-4, -1), (1, -3)]


def aligned_samples(angle=math.radians(30), scale=0.2, reflection=False):
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    if reflection:
        rotation = np.array([[-1.0, 0.0], [0.0, 1.0]]) @ rotation
    return [
        sample(step, scale * (rotation @ np.asarray(step)), index)
        for index, step in enumerate(CAMERA_STEPS, 1)
    ], rotation


def valid_motion(dx=2.0, dy=1.0, yaw=0.0):
    return AffineMotionResult(dx, dy, yaw, 20, 19, 0.95, 0.05, True, None)


def test_calibration_sample_acceptance_and_rejections():
    accepted, reason = make_calibration_sample(
        frame_index=2, sequence=2, camera_delta=(2, 0), gps_delta=(0.2, 0),
        motion=valid_motion(), yaw_rad=0, policy=policy(),
    )
    assert accepted is not None and reason is None
    rejected, reason = make_calibration_sample(
        frame_index=2, sequence=2, camera_delta=(0.1, 0), gps_delta=(0.2, 0),
        motion=valid_motion(), yaw_rad=0, policy=policy(),
    )
    assert rejected is None and reason == "camera_step_too_small"
    rejected, reason = make_calibration_sample(
        frame_index=2, sequence=2, camera_delta=(2, 0), gps_delta=(0.001, 0),
        motion=valid_motion(), yaw_rad=0, policy=policy(),
    )
    assert rejected is None and reason == "gps_step_too_small"
    rejected, reason = make_calibration_sample(
        frame_index=2, sequence=2, camera_delta=(2, 0), gps_delta=(0.2, 0),
        motion=AffineMotionResult(failure_reason="low_quality"), yaw_rad=0, policy=policy(),
    )
    assert rejected is None and reason == "motion_quality_invalid"


def test_not_ready_before_minimum_samples():
    samples, _ = aligned_samples()
    result = fit_calibration(samples[:3], policy())
    assert not result.ready
    assert result.failure_reason == "insufficient_samples"


def test_known_rotation_and_scale_are_recovered():
    samples, expected_rotation = aligned_samples()
    result = fit_calibration(samples, policy())
    assert result.ready
    assert result.scale == pytest.approx(0.2, rel=1e-6)
    assert np.asarray(result.rotation_matrix_2x2) == pytest.approx(expected_rotation, abs=1e-6)
    assert result.rms_residual == pytest.approx(0.0, abs=1e-7)
    assert result.directional_diversity >= 0.1


def test_outlier_scale_is_robustly_rejected():
    samples, _ = aligned_samples()
    samples.append(sample((3, 2), (15, 10), 99))
    result = fit_calibration(samples, policy())
    assert result.ready
    assert result.scale == pytest.approx(0.2, rel=1e-6)
    assert result.inlier_count == len(samples) - 1


def test_reflection_policy_is_explicit():
    samples, expected = aligned_samples(reflection=True)
    denied = fit_calibration(samples, policy(allow_reflection=False))
    allowed = fit_calibration(samples, policy(allow_reflection=True))
    assert not denied.ready and denied.failure_reason == "reflection_detected"
    assert allowed.ready
    assert np.linalg.det(np.asarray(allowed.rotation_matrix_2x2)) < 0
    assert np.asarray(allowed.rotation_matrix_2x2) == pytest.approx(expected, abs=1e-6)


def test_low_directional_diversity_is_not_ready():
    samples = [sample((i + 1, 0), (0.2 * (i + 1), 0), i) for i in range(6)]
    result = fit_calibration(samples, policy())
    assert not result.ready
    assert result.failure_reason == "insufficient_directional_diversity"


def test_high_residual_is_not_ready():
    samples, _ = aligned_samples()
    rotated_gps = [item.gps_delta_xy for item in samples]
    scrambled = [sample(CAMERA_STEPS[i], rotated_gps[(i + 2) % len(samples)], i) for i in range(len(samples))]
    result = fit_calibration(scrambled, policy(max_rms_residual=0.001, min_inlier_ratio=0.4))
    assert not result.ready
    assert result.failure_reason in {"high_residual", "low_inlier_ratio"}


def test_estimator_applies_transform_clamp_and_z_policy():
    calibration = CalibrationResult(
        ready=True, sample_count=6, inlier_count=6,
        rotation_matrix_2x2=((0.0, -1.0), (1.0, 0.0)), scale=0.5,
        rms_residual=0, scale_median=0.5, scale_mad=0,
        motion_span=10, directional_diversity=0.5, failure_reason=None,
    )
    result = estimate_translation(
        calibration=calibration, gps_anchor=(100, 200, 50), camera_anchor=(0, 0),
        camera_position=(4, 0), last_estimate=(100, 200, 50),
        max_delta_per_frame=10, z_policy="hold_last_valid_z",
    )
    assert (result.translation_x, result.translation_y, result.translation_z) == pytest.approx((100, 202, 50))
    clamped = estimate_translation(
        calibration=calibration, gps_anchor=(100, 200, 50), camera_anchor=(0, 0),
        camera_position=(40, 0), last_estimate=(100, 200, 50),
        max_delta_per_frame=2, z_policy="hold_last_valid_z",
    )
    assert (clamped.translation_x, clamped.translation_y) == pytest.approx((100, 202))
    assert estimate_translation(
        calibration=calibration, gps_anchor=(100, 200, 50), camera_anchor=(0, 0),
        camera_position=(4, 0), last_estimate=None, max_delta_per_frame=0,
        z_policy="return_none_if_schema_allows",
    ) is None
    assert estimate_translation(
        calibration=calibration, gps_anchor=(100, 200, 50), camera_anchor=(0, 0),
        camera_position=(float("nan"), 0), last_estimate=None,
        max_delta_per_frame=0, z_policy="hold_last_valid_z",
    ) is None


def app_settings(**changes):
    values = {
        "localization_enabled": True,
        "localization_vo_enabled": True,
        "localization_min_features": 8,
        "localization_max_features": 100,
        "localization_min_inliers": 6,
        "localization_camera_width": 100,
        "localization_camera_height": 100,
        "localization_camera_fx": 80.0,
        "localization_camera_fy": 80.0,
        "localization_camera_cx": 50.0,
        "localization_camera_cy": 50.0,
        "localization_camera_calibration_path": None,
        "localization_calibration_min_samples": 4,
        "localization_calibration_max_samples": 20,
        "localization_calibration_min_camera_step_px": 0.5,
        "localization_calibration_min_gps_step": 0.01,
        "localization_calibration_max_rms_residual": 0.05,
        "localization_calibration_min_inlier_ratio": 0.7,
        "localization_calibration_min_directional_diversity": 0.1,
        "localization_calibration_scale_min": 0.001,
        "localization_calibration_scale_max": 2.0,
        "localization_calibration_expected_max_frame": 450,
        "localization_max_delta_per_frame": 10.0,
        "localization_warmup_frames": 1,
        "matching_enabled": False,
    }
    values.update(changes)
    return replace(get_settings(), **values)


class AnyCamera:
    def for_resolution(self, width, height):
        return CameraModel(width, height, 80, 80, width / 2, height / 2, ())


class SequenceVO:
    def __init__(self, motions):
        self.motions = list(motions)
        self.calls = 0

    def estimate(self, *_args):
        motion = self.motions[min(self.calls, len(self.motions) - 1)]
        self.calls += 1
        return motion


def request(index, gps, health=GPSHealthStatus.HEALTHY, session="s", source=None):
    return FrameContext(
        frame_id=f"f{index}", image_url=source or f"i{index}", video_name="v",
        session_id=session, gps_health_status=health,
        gps_x=None if gps is None else gps[0], gps_y=None if gps is None else gps[1],
        gps_z=None if gps is None else gps[2], frame_index=index, image_modality=ImageModality.RGB,
    )


def localization(images, motions, **changes):
    async def reader(source, _timeout):
        value = images[source]
        if isinstance(value, Exception):
            raise value
        return source.encode()

    return AffineLocalizationService(
        app_settings(**changes), vo=SequenceVO(motions), camera_provider=AnyCamera(),
        image_reader=reader, image_decoder=lambda content: images[content.decode()].copy(),
    )


def trajectory(scale=0.2, angle=math.radians(30)):
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    gps = np.array([100.0, 200.0, 50.0])
    values = [tuple(gps)]
    motions = []
    for dx, dy in CAMERA_STEPS:
        motions.append(valid_motion(dx, dy, 0))
        gps[:2] += scale * (rotation @ np.array([dx, dy]))
        values.append(tuple(gps))
    return values, motions, rotation


@pytest.mark.asyncio
async def test_healthy_first_frame_returns_server_ground_truth():
    images = {"i1": np.zeros((100, 100, 3), np.uint8)}
    service = localization(images, [valid_motion()])
    result = await service.process_frame(request(1, (10, 20, 30)), LocalizationSessionState("s"))
    assert (result.translation_x, result.translation_y, result.translation_z) == (10, 20, 30)


@pytest.mark.asyncio
async def test_quality_samples_accumulate_and_alignment_becomes_ready():
    gps_values, motions, _ = trajectory()
    images = {f"i{i}": np.full((100, 100, 3), i, np.uint8) for i in range(1, len(gps_values) + 1)}
    service = localization(images, motions)
    state = LocalizationSessionState("s")
    for index, gps in enumerate(gps_values, 1):
        result = await service.process_frame(request(index, gps), state)
        assert result is not None
    assert state.vo_state.frame_count == len(gps_values)
    assert len(state.vo_state.calibration_samples) == len(motions)
    assert state.vo_state.calibration_result.ready
    assert state.vo_state.calibration_result.scale == pytest.approx(0.2, rel=1e-6)


@pytest.mark.asyncio
async def test_invalid_motion_and_small_steps_are_rejected():
    images = {"i1": np.zeros((100, 100, 3), np.uint8), "i2": np.ones((100, 100, 3), np.uint8)}
    bad = AffineMotionResult(failure_reason="low_quality")
    state = LocalizationSessionState("s")
    service = localization(images, [bad])
    await service.process_frame(request(1, (1, 2, 3)), state)
    await service.process_frame(request(2, (2, 3, 3)), state)
    assert state.vo_state.calibration_samples == []
    state = LocalizationSessionState("s")
    service = localization(images, [valid_motion(0.1, 0)], localization_calibration_min_camera_step_px=0.5)
    await service.process_frame(request(1, (1, 2, 3)), state)
    await service.process_frame(request(2, (1.001, 2, 3)), state)
    assert state.vo_state.calibration_samples == []


async def calibrated_service_state():
    gps_values, motions, rotation = trajectory()
    motions = motions + [valid_motion(2, 0), valid_motion(1, 0)]
    images = {f"i{i}": np.full((100, 100, 3), i, np.uint8) for i in range(1, len(gps_values) + 3)}
    service = localization(images, motions)
    state = LocalizationSessionState("s")
    for index, gps in enumerate(gps_values, 1):
        await service.process_frame(request(index, gps), state)
    return service, state, gps_values[-1], rotation


@pytest.mark.asyncio
async def test_gps_loss_creates_anchor_and_generates_xy_with_held_z():
    service, state, last_gps, rotation = await calibrated_service_state()
    index = state.vo_state.previous_frame_index + 1
    result = await service.process_frame(request(index, None, GPSHealthStatus.UNHEALTHY), state)
    expected_xy = np.asarray(last_gps[:2]) + 0.2 * (rotation @ np.array([2, 0]))
    assert (result.translation_x, result.translation_y) == pytest.approx(expected_xy)
    assert result.translation_z == last_gps[2]
    assert state.vo_state.calibration_frozen
    assert state.vo_state.gps_anchor == last_gps


@pytest.mark.asyncio
async def test_gps_loss_without_calibration_or_last_gps_returns_none():
    images = {"i1": np.zeros((100, 100, 3), np.uint8), "i2": np.ones((100, 100, 3), np.uint8)}
    service = localization(images, [valid_motion()])
    state = LocalizationSessionState("s")
    assert await service.process_frame(request(1, None, GPSHealthStatus.UNHEALTHY), state) is None
    assert await service.process_frame(request(2, None, GPSHealthStatus.UNHEALTHY), state) is None


@pytest.mark.asyncio
async def test_ready_calibration_still_returns_none_for_invalid_current_vo():
    service, state, _, _ = await calibrated_service_state()
    service._vo.motions[service._vo.calls] = AffineMotionResult(failure_reason="low_quality")
    index = state.vo_state.previous_frame_index + 1
    assert await service.process_frame(request(index, None, GPSHealthStatus.UNHEALTHY), state) is None
    assert state.vo_state.last_estimate is not None


@pytest.mark.asyncio
async def test_invalid_vo_duplicate_and_gap_do_not_predict():
    service, state, _, _ = await calibrated_service_state()
    state.vo_state.calibration_result = state.vo_state.calibration_result
    index = state.vo_state.previous_frame_index
    duplicate = FrameContext(f"f{index}", f"i{index}", "v", "s", GPSHealthStatus.UNHEALTHY, None, None, None, index, ImageModality.RGB)
    assert await service.process_frame(duplicate, state) is None
    gap_index = index + 5
    assert await service.process_frame(request(gap_index, None, GPSHealthStatus.UNHEALTHY, source=f"i{index + 1}"), state) is None


@pytest.mark.asyncio
async def test_gps_recovery_returns_truth_and_renews_anchor():
    service, state, _, _ = await calibrated_service_state()
    lost_index = state.vo_state.previous_frame_index + 1
    await service.process_frame(request(lost_index, None, GPSHealthStatus.UNHEALTHY), state)
    recovery = (500.0, 600.0, 70.0)
    result = await service.process_frame(request(lost_index + 1, recovery), state)
    assert (result.translation_x, result.translation_y, result.translation_z) == recovery
    assert state.vo_state.gps_health_transition == "recovered"
    assert state.vo_state.gps_anchor == recovery
    assert not state.vo_state.calibration_frozen


@pytest.mark.asyncio
async def test_expected_450_window_logs_critical_warning(caplog):
    images = {"i450": np.zeros((100, 100, 3), np.uint8)}
    service = localization(images, [valid_motion()])
    state = LocalizationSessionState("s")
    await service.process_frame(request(450, (1, 2, 3)), state)
    assert any(record.message == "localization_expected_window_exhausted" for record in caplog.records)


@pytest.mark.asyncio
async def test_sessions_keep_calibration_isolated_and_reset():
    store = LocalizationSessionStore(ttl_seconds=60, max_sessions=4)
    async with store.locked("a") as first:
        first.vo_state = VisualOdometryState()
        first.vo_state.calibration_samples.append(sample((1, 0), (0.2, 0)))
    async with store.locked("b") as second:
        assert second.vo_state is None
    await store.reset("a")
    async with store.locked("a") as reset:
        assert reset.vo_state is None


@pytest.mark.asyncio
async def test_null_nan_and_infinite_healthy_gps_return_none():
    images = {"i1": np.zeros((100, 100, 3), np.uint8)}
    for gps in (None, (float("nan"), 2, 3), (1, float("inf"), 3)):
        service = localization(images, [valid_motion()])
        assert await service.process_frame(request(1, gps), LocalizationSessionState("s")) is None


@pytest.mark.parametrize("change", [
    {"localization_calibration_min_samples": 2},
    {"localization_calibration_max_samples": 3, "localization_calibration_min_samples": 4},
    {"localization_calibration_min_camera_step_px": -1},
    {"localization_calibration_min_inlier_ratio": 0},
    {"localization_calibration_min_directional_diversity": 1.1},
    {"localization_calibration_scale_min": 0},
    {"localization_calibration_scale_min": 2, "localization_calibration_scale_max": 1},
    {"localization_calibration_outlier_mad_factor": 0},
    {"localization_calibration_expected_max_frame": 0},
    {"localization_z_policy": "invent_z"},
    {"localization_recovery_min_healthy_frames": 0},
])
def test_calibration_config_validation(change):
    with pytest.raises(ValueError):
        app_settings(**change).validate_localization_vo()
