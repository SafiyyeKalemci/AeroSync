from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

import cv2
import numpy as np

from app.core.config import Settings
from app.schemas import DetectedTranslation, GPSHealthStatus, ImageModality
from app.services.common import FrameContext
from app.services.localization.affine_vo import AffineVOConfig, AffineVisualOdometry
from app.services.localization.alignment import fit_calibration
from app.services.localization.calibration import CalibrationPolicy, CalibrationResult, make_calibration_sample
from app.services.localization.camera_model import CameraModelProvider
from app.services.localization.continuity import ContinuityAction, evaluate_continuity
from app.services.localization.estimator import estimate_translation
from app.services.localization.interface import LocalizationService, LocalizationSessionState
from app.services.localization.state import AffineMotionResult, VisualOdometryState
from app.utils.images import read_image_bytes

logger = logging.getLogger(__name__)

ImageReader = Callable[[str, float], Awaitable[bytes]]
ImageDecoder = Callable[[bytes], object]


def decode_opencv_image(content: bytes) -> np.ndarray:
    if not content:
        raise ValueError("Image content is empty")
    encoded = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("OpenCV could not decode image")
    return image


class AffineLocalizationService(LocalizationService):
    """Updates per-session visual-motion state without fabricating coordinates."""

    def __init__(
        self,
        settings: Settings,
        *,
        vo: AffineVisualOdometry | None = None,
        camera_provider: CameraModelProvider | None = None,
        image_reader: ImageReader = read_image_bytes,
        image_decoder: ImageDecoder = decode_opencv_image,
    ) -> None:
        settings.validate_localization_vo()
        self._settings = settings
        self._vo = vo or AffineVisualOdometry(
            AffineVOConfig(
                min_features=settings.localization_min_features,
                max_features=settings.localization_max_features,
                feature_quality_level=settings.localization_feature_quality_level,
                feature_min_distance=settings.localization_feature_min_distance,
                lk_win_size=settings.localization_lk_win_size,
                lk_max_level=settings.localization_lk_max_level,
                lk_fb_error_threshold=settings.localization_lk_fb_error_threshold,
                ransac_iterations=settings.localization_ransac_iterations,
                ransac_residual_threshold=settings.localization_ransac_residual_threshold,
                min_inliers=settings.localization_min_inliers,
                min_inlier_ratio=settings.localization_min_inlier_ratio,
                freeze_threshold=settings.localization_freeze_threshold,
            )
        )
        self._camera_provider = camera_provider or CameraModelProvider(settings)
        self._calibration_policy = CalibrationPolicy(
            min_samples=settings.localization_calibration_min_samples,
            max_samples=settings.localization_calibration_max_samples,
            min_camera_step_px=settings.localization_calibration_min_camera_step_px,
            min_gps_step=settings.localization_calibration_min_gps_step,
            max_rms_residual=settings.localization_calibration_max_rms_residual,
            min_inlier_ratio=settings.localization_calibration_min_inlier_ratio,
            min_directional_diversity=settings.localization_calibration_min_directional_diversity,
            scale_min=settings.localization_calibration_scale_min,
            scale_max=settings.localization_calibration_scale_max,
            outlier_mad_factor=settings.localization_calibration_outlier_mad_factor,
            allow_reflection=settings.localization_allow_reflection,
        )
        self._image_reader = image_reader
        self._image_decoder = image_decoder

    async def reset_session(self, session_id: str) -> None:
        logger.info(
            "localization_state_reset",
            extra={"event": "localization_state_reset", "session_id": session_id},
        )

    async def process_frame(
        self, frame: FrameContext, state: LocalizationSessionState
    ) -> DetectedTranslation | None:
        ground_truth = self._ground_truth(frame)
        previous_health = state.last_gps_health_status
        if frame.image_modality not in {None, ImageModality.RGB}:
            logger.info(
                "localization_modality_changed",
                extra={
                    "event": "localization_modality_changed",
                    "reason": "unsupported_modality",
                    "session_id": frame.session_id,
                    "frame_id": frame.frame_id,
                },
            )
            state.vo_state = VisualOdometryState()
            self._update_envelope(state, frame)
            return ground_truth
        try:
            content = await self._image_reader(frame.image_url, self._settings.http_timeout_seconds)
            image = await asyncio.to_thread(self._image_decoder, content)
            shape = getattr(image, "shape", ())
            if len(shape) < 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
                raise ValueError("Decoded image dimensions are invalid")
            height, width = int(shape[0]), int(shape[1])
            camera = self._camera_provider.for_resolution(width, height)
            if camera is None:
                logger.warning("localization_low_quality", extra={"reason": "invalid_camera_model"})
                return ground_truth
            gray = await asyncio.to_thread(self._prepare_gray, image, camera)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "localization_tracking_failed",
                extra={"reason": "image_decode_failed", "session_id": frame.session_id, "frame_id": frame.frame_id},
                exc_info=True,
            )
            return ground_truth

        fingerprint = hashlib.blake2b(gray.tobytes(), digest_size=16).digest()
        current = state.vo_state or VisualOdometryState()
        decision = evaluate_continuity(
            current,
            frame,
            shape=(height, width),
            modality=frame.image_modality,
            fingerprint=fingerprint,
            max_frame_gap=self._settings.localization_max_frame_gap,
        )
        if decision.event:
            logger.info(
                decision.event,
                extra={"event": decision.event, "session_id": frame.session_id, "frame_id": frame.frame_id},
            )
        if decision.action is ContinuityAction.DUPLICATE:
            current.last_access_time = datetime.now(timezone.utc)
            state.updated_at = current.last_access_time
            return ground_truth
        if decision.action in {ContinuityAction.FIRST, ContinuityAction.RESET}:
            self._commit_baseline(state, current, frame, gray, fingerprint)
            return self._produce_output(
                frame=frame,
                state=state,
                previous=current,
                previous_health=previous_health,
                motion=None,
                continuous=False,
                ground_truth=ground_truth,
            )
        if decision.action is ContinuityAction.REPEATED_IMAGE:
            result = AffineMotionResult(failure_reason="freeze")
        else:
            try:
                result = await asyncio.to_thread(self._vo.estimate, current.previous_gray, gray, camera)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "localization_tracking_failed",
                    extra={"reason": "vo_exception", "session_id": frame.session_id, "frame_id": frame.frame_id},
                    exc_info=True,
                )
                return ground_truth

        self._log_result(result, frame)
        self._commit_motion(state, current, frame, gray, fingerprint, result)
        return self._produce_output(
            frame=frame,
            state=state,
            previous=current,
            previous_health=previous_health,
            motion=result,
            continuous=decision.action is ContinuityAction.CONTINUE,
            ground_truth=ground_truth,
        )

    @staticmethod
    def _ground_truth(frame: FrameContext) -> DetectedTranslation | None:
        if frame.gps_health_status is not GPSHealthStatus.HEALTHY:
            return None
        values = (frame.gps_x, frame.gps_y, frame.gps_z)
        if any(value is None for value in values):
            return None
        coordinates = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in coordinates):
            return None
        return DetectedTranslation(
            translation_x=coordinates[0],
            translation_y=coordinates[1],
            translation_z=coordinates[2],
        )

    def _produce_output(
        self,
        *,
        frame: FrameContext,
        state: LocalizationSessionState,
        previous: VisualOdometryState,
        previous_health: GPSHealthStatus | None,
        motion: AffineMotionResult | None,
        continuous: bool,
        ground_truth: DetectedTranslation | None,
    ) -> DetectedTranslation | None:
        current = state.vo_state
        assert current is not None
        if frame.gps_health_status is GPSHealthStatus.HEALTHY:
            if ground_truth is None:
                current.previous_valid_gps = None
                current.previous_gps_frame_index = None
                logger.info(
                    "localization_calibration_sample_rejected",
                    extra={"event": "localization_calibration_sample_rejected", "reason": "invalid_ground_truth", "session_id": frame.session_id},
                )
                return None
            return self._handle_healthy(
                frame, current, previous, previous_health, motion, continuous, ground_truth
            )
        if frame.gps_health_status is GPSHealthStatus.UNHEALTHY:
            return self._handle_unhealthy(
                frame, current, previous, previous_health, motion, continuous
            )
        logger.info(
            "localization_prediction_unavailable",
            extra={"event": "localization_prediction_unavailable", "reason": "gps_health_unknown", "session_id": frame.session_id},
        )
        return None

    def _handle_healthy(
        self,
        frame: FrameContext,
        current: VisualOdometryState,
        previous: VisualOdometryState,
        previous_health: GPSHealthStatus | None,
        motion: AffineMotionResult | None,
        continuous: bool,
        ground_truth: DetectedTranslation,
    ) -> DetectedTranslation:
        gps = (
            ground_truth.translation_x,
            ground_truth.translation_y,
            ground_truth.translation_z,
        )
        if previous_health is GPSHealthStatus.UNHEALTHY:
            current.gps_health_transition = "recovered"
            current.recovery_healthy_count = 1
            current.gps_anchor = gps
            current.camera_anchor = (current.cumulative_dx_px, current.cumulative_dy_px)
            current.last_estimate = gps
            logger.info(
                "localization_gps_recovered",
                extra={"event": "localization_gps_recovered", "session_id": frame.session_id},
            )
        elif current.recovery_healthy_count:
            current.recovery_healthy_count += 1
        if current.recovery_healthy_count >= self._settings.localization_recovery_min_healthy_frames:
            current.calibration_frozen = False
            current.frozen_calibration_result = None

        can_pair = (
            self._settings.localization_calibration_enabled
            and continuous
            and motion is not None
            and previous_health is GPSHealthStatus.HEALTHY
            and previous.previous_valid_gps is not None
            and previous.previous_gps_frame_index is not None
            and frame.frame_index is not None
            and 0 < frame.frame_index - previous.previous_gps_frame_index <= self._settings.localization_max_frame_gap
        )
        if can_pair:
            camera_delta = (
                current.cumulative_dx_px - previous.cumulative_dx_px,
                current.cumulative_dy_px - previous.cumulative_dy_px,
            )
            gps_delta = (
                gps[0] - previous.previous_valid_gps[0],
                gps[1] - previous.previous_valid_gps[1],
            )
            sample, reason = make_calibration_sample(
                frame_index=frame.frame_index,
                sequence=current.frame_count,
                camera_delta=camera_delta,
                gps_delta=gps_delta,
                motion=motion,
                yaw_rad=current.cumulative_yaw,
                policy=self._calibration_policy,
            )
            if sample is None:
                logger.info(
                    "localization_calibration_sample_rejected",
                    extra={"event": "localization_calibration_sample_rejected", "reason": reason, "session_id": frame.session_id},
                )
            else:
                current.calibration_samples.append(sample)
                current.calibration_samples = current.calibration_samples[-self._calibration_policy.max_samples :]
                current.camera_step_samples.append(camera_delta)
                current.gps_step_samples.append(gps_delta)
                current.camera_step_samples = current.camera_step_samples[-self._calibration_policy.max_samples :]
                current.gps_step_samples = current.gps_step_samples[-self._calibration_policy.max_samples :]
                current.healthy_sample_count += 1
                logger.info(
                    "localization_calibration_sample_added",
                    extra={"event": "localization_calibration_sample_added", "sample_count": len(current.calibration_samples), "session_id": frame.session_id},
                )
                prior_ready = bool(current.calibration_result and current.calibration_result.ready)
                current.calibration_result = fit_calibration(
                    current.calibration_samples, self._calibration_policy
                )
                if current.calibration_result.ready:
                    if not prior_ready:
                        logger.info(
                            "localization_calibration_ready",
                            extra={"event": "localization_calibration_ready", "sample_count": current.calibration_result.sample_count, "inlier_count": current.calibration_result.inlier_count, "session_id": frame.session_id},
                        )
                else:
                    event = "localization_calibration_failed" if current.calibration_result.failure_reason in {"alignment_failed", "non_finite_samples"} else "localization_calibration_quality_low"
                    logger.info(
                        event,
                        extra={"event": event, "reason": current.calibration_result.failure_reason, "sample_count": current.calibration_result.sample_count, "session_id": frame.session_id},
                    )
        elif motion is not None and self._settings.localization_calibration_enabled:
            reason = "motion_quality_invalid" if not motion.quality_valid else "unmatched_gps_step"
            logger.info(
                "localization_calibration_sample_rejected",
                extra={"event": "localization_calibration_sample_rejected", "reason": reason, "session_id": frame.session_id},
            )

        current.previous_valid_gps = gps
        current.last_valid_gps = gps
        current.previous_gps_frame_index = frame.frame_index
        current.last_estimate = gps
        current.unhealthy_frame_count = 0
        self._check_expected_window(frame, current)
        logger.info(
            "localization_ground_truth_returned",
            extra={"event": "localization_ground_truth_returned", "source": "server_ground_truth", "session_id": frame.session_id},
        )
        return ground_truth

    def _handle_unhealthy(
        self,
        frame: FrameContext,
        current: VisualOdometryState,
        previous: VisualOdometryState,
        previous_health: GPSHealthStatus | None,
        motion: AffineMotionResult | None,
        continuous: bool,
    ) -> DetectedTranslation | None:
        current.unhealthy_frame_count += 1
        current.recovery_healthy_count = 0
        if previous_health is GPSHealthStatus.HEALTHY:
            current.gps_health_transition = "lost"
            logger.info(
                "localization_gps_lost",
                extra={"event": "localization_gps_lost", "session_id": frame.session_id},
            )
            has_adjacent_gps = (
                previous.previous_valid_gps is not None
                and previous.previous_gps_frame_index is not None
                and frame.frame_index is not None
                and 0 < frame.frame_index - previous.previous_gps_frame_index <= self._settings.localization_max_frame_gap
            )
            if has_adjacent_gps:
                current.gps_anchor = previous.previous_valid_gps
                current.camera_anchor = (
                    previous.cumulative_dx_px,
                    previous.cumulative_dy_px,
                )
                current.calibration_frozen = True
                current.frozen_calibration_result = previous.calibration_result
                logger.info(
                    "localization_anchor_created",
                    extra={"event": "localization_anchor_created", "session_id": frame.session_id},
                )
        if not continuous or motion is None or not motion.quality_valid:
            logger.info(
                "localization_prediction_unavailable",
                extra={"event": "localization_prediction_unavailable", "reason": "vo_quality_or_continuity", "session_id": frame.session_id},
            )
            return None
        calibration = (
            current.frozen_calibration_result
            if current.calibration_frozen
            else current.calibration_result
        )
        if calibration is None:
            calibration = CalibrationResult()
        estimate = estimate_translation(
            calibration=calibration,
            gps_anchor=current.gps_anchor,
            camera_anchor=current.camera_anchor,
            camera_position=(current.cumulative_dx_px, current.cumulative_dy_px),
            last_estimate=current.last_estimate,
            max_delta_per_frame=self._settings.localization_max_delta_per_frame,
            z_policy=self._settings.localization_z_policy,
        )
        if estimate is None:
            logger.info(
                "localization_prediction_unavailable",
                extra={"event": "localization_prediction_unavailable", "reason": "calibration_or_anchor_missing", "session_id": frame.session_id},
            )
            return None
        current.last_estimate = (
            estimate.translation_x, estimate.translation_y, estimate.translation_z
        )
        logger.info(
            "localization_prediction_generated",
            extra={"event": "localization_prediction_generated", "session_id": frame.session_id},
        )
        if self._settings.localization_z_policy in {"hold_last_valid_z", "zero_delta_from_anchor"}:
            logger.info(
                "localization_z_held",
                extra={"event": "localization_z_held", "policy": self._settings.localization_z_policy, "session_id": frame.session_id},
            )
        return estimate

    def _check_expected_window(
        self, frame: FrameContext, state: VisualOdometryState
    ) -> None:
        if (
            frame.frame_index is not None
            and frame.frame_index >= self._settings.localization_calibration_expected_max_frame
            and not (state.calibration_result and state.calibration_result.ready)
            and not state.expected_window_warned
        ):
            state.expected_window_warned = True
            logger.critical(
                "localization_expected_window_exhausted",
                extra={"event": "localization_expected_window_exhausted", "frame_index": frame.frame_index, "session_id": frame.session_id},
            )

    @staticmethod
    def _prepare_gray(image: object, camera) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 2:
            gray = array
        elif array.ndim == 3:
            gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unsupported decoded image shape")
        return camera.undistort(gray)

    def _commit_baseline(
        self,
        state: LocalizationSessionState,
        previous: VisualOdometryState,
        frame: FrameContext,
        gray: np.ndarray,
        fingerprint: bytes,
    ) -> None:
        state.vo_state = VisualOdometryState(
            previous_gray=gray,
            previous_frame_id=frame.frame_id,
            previous_frame_index=frame.frame_index,
            video_name=frame.video_name,
            image_shape=(int(gray.shape[0]), int(gray.shape[1])),
            modality=frame.image_modality,
            frame_count=previous.frame_count + 1,
            previous_fingerprint=fingerprint,
        )
        self._update_envelope(state, frame)

    def _commit_motion(
        self,
        state: LocalizationSessionState,
        previous: VisualOdometryState,
        frame: FrameContext,
        gray: np.ndarray,
        fingerprint: bytes,
        result: AffineMotionResult,
    ) -> None:
        warmup = previous.warmup_count + 1
        yaw = previous.cumulative_yaw
        dx = previous.cumulative_dx_px
        dy = previous.cumulative_dy_px
        if result.quality_valid and warmup >= self._settings.localization_warmup_frames:
            assert result.delta_x_px is not None and result.delta_y_px is not None and result.delta_yaw_rad is not None
            yaw += result.delta_yaw_rad
            cosine, sine = math.cos(-yaw), math.sin(-yaw)
            dx += cosine * result.delta_x_px - sine * result.delta_y_px
            dy += sine * result.delta_x_px + cosine * result.delta_y_px
        state.vo_state = VisualOdometryState(
            previous_gray=gray,
            previous_frame_id=frame.frame_id,
            previous_frame_index=frame.frame_index,
            video_name=frame.video_name,
            image_shape=(int(gray.shape[0]), int(gray.shape[1])),
            modality=frame.image_modality,
            frame_count=previous.frame_count + 1,
            warmup_count=warmup,
            cumulative_yaw=yaw,
            cumulative_dx_px=dx,
            cumulative_dy_px=dy,
            last_motion_result=result,
            duplicate_cache=result,
            previous_fingerprint=fingerprint,
            freeze_detected=result.failure_reason == "freeze",
        )
        self._carry_calibration_state(state.vo_state, previous)
        self._update_envelope(state, frame)

    @staticmethod
    def _carry_calibration_state(
        target: VisualOdometryState, source: VisualOdometryState
    ) -> None:
        target.previous_valid_gps = source.previous_valid_gps
        target.last_valid_gps = source.last_valid_gps
        target.previous_gps_frame_index = source.previous_gps_frame_index
        target.camera_step_samples = list(source.camera_step_samples)
        target.gps_step_samples = list(source.gps_step_samples)
        target.calibration_samples = list(source.calibration_samples)
        target.calibration_result = source.calibration_result
        target.frozen_calibration_result = source.frozen_calibration_result
        target.calibration_frozen = source.calibration_frozen
        target.gps_anchor = source.gps_anchor
        target.camera_anchor = source.camera_anchor
        target.last_estimate = source.last_estimate
        target.gps_health_transition = source.gps_health_transition
        target.healthy_sample_count = source.healthy_sample_count
        target.unhealthy_frame_count = source.unhealthy_frame_count
        target.recovery_healthy_count = source.recovery_healthy_count
        target.expected_window_warned = source.expected_window_warned

    @staticmethod
    def _update_envelope(state: LocalizationSessionState, frame: FrameContext) -> None:
        state.frame_count += 1
        state.last_frame_id = frame.frame_id
        state.last_gps_health_status = frame.gps_health_status
        state.updated_at = datetime.now(timezone.utc)

    @staticmethod
    def _log_result(result: AffineMotionResult, frame: FrameContext) -> None:
        events = {
            "insufficient_features": "localization_insufficient_features",
            "tracking_failed": "localization_tracking_failed",
            "ransac_failed": "localization_ransac_failed",
            "low_quality": "localization_low_quality",
            "freeze": "localization_freeze_detected",
        }
        event = "localization_motion_estimated" if result.quality_valid else events.get(result.failure_reason, "localization_low_quality")
        logger.info(
            event,
            extra={
                "event": event,
                "session_id": frame.session_id,
                "frame_id": frame.frame_id,
                "tracked_points": result.tracked_points,
                "inlier_count": result.inlier_count,
                "inlier_ratio": result.inlier_ratio,
                "rms_residual": result.rms_residual,
                "reason": result.failure_reason,
            },
        )
