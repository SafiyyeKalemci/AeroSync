from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser().resolve() if value else None


def _project_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _first_path(*names: str) -> Path | None:
    for name in names:
        value = _path(name)
        if value is not None:
            return value
    return None


def _normalized_base_url(name: str) -> str:
    value = os.getenv(name, "").strip()
    return value.rstrip("/") + "/" if value else ""


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    environment: str
    log_level: str
    api_key: str
    team_user_url: str
    backend_url: str
    competition_url: str
    http_timeout_seconds: float
    session_duration_seconds: float

    team_name: str
    password: str = field(repr=False)
    evaluation_server_url: str
    official_session_name: str
    official_interface_path: Path | None
    official_media_dir: Path
    competition_max_retries: int
    competition_retry_initial_seconds: float
    competition_frame_interval_seconds: float
    competition_task_timeout_seconds: float

    detection_enabled: bool
    detection_model_path: Path | None
    detection_confidence: float
    detection_iou: float
    detection_motion_enabled: bool
    detection_motion_method: str
    detection_motion_threshold_px: float
    detection_motion_min_valid_pixels: int
    detection_motion_inner_crop_ratio: float
    detection_motion_max_frame_gap: int
    detection_motion_warmup_frames: int
    detection_motion_flow_downscale: float
    detection_motion_freeze_threshold: float
    detection_motion_session_ttl_seconds: float
    detection_motion_max_sessions: int
    detection_motion_homography_min_features: int
    detection_motion_homography_min_inliers: int
    detection_motion_homography_min_inlier_ratio: float
    detection_motion_homography_ransac_threshold: float
    detection_motion_homography_max_condition_number: float
    detection_motion_homography_residual_threshold_px: float
    detection_motion_homography_quality_gate: str
    detection_motion_homography_adaptive_high_inlier_ratio: float
    detection_motion_homography_adaptive_low_inlier_ratio: float
    detection_motion_homography_adaptive_max_reprojection_error_px: float
    detection_motion_homography_adaptive_min_spatial_coverage: float
    detection_motion_homography_adaptive_min_projected_overlap_ratio: float
    detection_motion_bbox_match_min_iou: float
    detection_motion_bbox_match_max_center_distance_ratio: float
    detection_motion_bbox_match_min_score: float
    detection_motion_bbox_stationary_threshold_px: float
    detection_motion_bbox_moving_threshold_px: float
    detection_motion_bbox_min_size_ratio: float
    detection_motion_bbox_max_size_ratio: float
    detection_motion_bbox_min_visible_ratio: float
    detection_motion_adaptive_background_median_max: float
    detection_motion_adaptive_background_p90_max: float
    detection_motion_adaptive_grid_spread_max: float
    detection_motion_adaptive_min_valid_background_ratio: float
    detection_motion_hybrid_strong_moving_residual_px: float
    detection_motion_hybrid_min_association_score: float
    detection_motion_hybrid_min_iou: float
    detection_motion_local_ring_expansion_ratio: float
    detection_motion_local_min_background_pixels: int
    detection_motion_local_stationary_threshold_px: float
    detection_motion_local_moving_threshold_px: float
    detection_motion_local_min_valid_ratio: float
    detection_landing_enabled: bool
    detection_landing_edge_margin_px: float
    detection_landing_edge_margin_ratio: float
    detection_landing_min_intersection_pixels: float
    detection_landing_occupancy_ratio: float
    detection_landing_use_center_check: bool
    detection_landing_use_bottom_center_check: bool
    detection_landing_min_area_pixels: float

    localization_enabled: bool
    localization_model_path: Path | None
    localization_vo_enabled: bool
    localization_min_features: int
    localization_max_features: int
    localization_feature_quality_level: float
    localization_feature_min_distance: float
    localization_lk_win_size: int
    localization_lk_max_level: int
    localization_lk_fb_error_threshold: float
    localization_ransac_iterations: int
    localization_ransac_residual_threshold: float
    localization_min_inliers: int
    localization_min_inlier_ratio: float
    localization_max_frame_gap: int
    localization_warmup_frames: int
    localization_freeze_threshold: float
    localization_session_ttl_seconds: float
    localization_max_sessions: int
    localization_camera_width: int
    localization_camera_height: int
    localization_camera_fx: float
    localization_camera_fy: float
    localization_camera_cx: float
    localization_camera_cy: float
    localization_camera_distortion: str
    localization_camera_calibration_path: Path | None
    localization_calibration_enabled: bool
    localization_calibration_min_samples: int
    localization_calibration_max_samples: int
    localization_calibration_min_camera_step_px: float
    localization_calibration_min_gps_step: float
    localization_calibration_max_rms_residual: float
    localization_calibration_min_inlier_ratio: float
    localization_calibration_min_directional_diversity: float
    localization_calibration_scale_min: float
    localization_calibration_scale_max: float
    localization_calibration_outlier_mad_factor: float
    localization_calibration_expected_max_frame: int
    localization_allow_reflection: bool
    localization_max_delta_per_frame: float
    localization_z_policy: str
    localization_recovery_min_healthy_frames: int

    matching_enabled: bool
    matching_timeout_seconds: float
    matching_device: str
    matching_allow_cpu_fallback: bool
    matching_dinov2_repo_path: Path | None
    matching_dinov2_weights_path: Path | None
    matching_dinov2_model_name: str
    matching_aliked_weights_path: Path | None
    matching_lightglue_weights_path: Path | None
    matching_xoftr_model_path: Path | None
    matching_xoftr_enabled: bool
    matching_xoftr_repo_path: Path | None
    matching_xoftr_ckpt_path: Path | None
    matching_xoftr_device: str
    matching_xoftr_max_edge: int
    matching_xoftr_timeout_seconds: float
    matching_xoftr_min_inliers: int
    matching_xoftr_rotation_sweep: bool
    matching_local_refinement_enabled: bool
    matching_geometry_method: str
    matching_aliked_model_path: Path | None
    matching_aliked_device: str
    matching_lightglue_model_path: Path | None
    matching_lightglue_device: str
    matching_local_min_keypoints: int
    matching_local_min_matches: int
    matching_local_min_inliers: int
    matching_local_min_inlier_ratio: float
    matching_local_max_reprojection_error: float
    matching_local_fallback_to_dinov2: bool
    matching_local_refinement_timeout_sec: float
    matching_preload_models: bool
    matching_warmup_enabled: bool
    matching_min_confidence: float
    matching_min_inliers: int
    matching_min_bbox_area: float
    matching_max_bbox_area_ratio: float
    matching_similarity_threshold: float
    matching_max_image_edge: int
    matching_reference_ttl_seconds: float
    matching_max_reference_sessions: int
    matching_reference_hash_enabled: bool
    matching_reference_cache_enabled: bool
    matching_dinov2_enabled: bool
    matching_dinov2_device: str
    matching_dinov2_max_long_edge: int
    matching_dinov2_patch_size: int
    matching_dinov2_descriptor_dtype: str
    matching_dinov2_allow_cpu_fallback: bool
    matching_dinov2_normalize_descriptors: bool
    matching_dinov2_max_cached_references: int
    matching_dinov2_timeout_seconds: float
    matching_dinov2_cache_device: str
    matching_dinov2_max_cache_mb: float
    matching_coarse_min_similarity: float
    matching_coarse_min_correspondences: int
    matching_coarse_max_correspondences: int
    matching_coarse_topk_per_reference: int
    matching_coarse_chunk_size: int
    matching_coarse_spatial_dedup_radius_px: float
    matching_coarse_timeout_seconds: float
    matching_reference_timeout_seconds: float
    matching_homography_method: str
    matching_homography_reprojection_threshold: float
    matching_homography_confidence: float
    matching_homography_max_iterations: int
    matching_homography_min_inliers: int
    matching_homography_min_inlier_ratio: float
    matching_homography_max_rms_reprojection_error: float
    matching_geometry_min_projected_area_px: float
    matching_geometry_max_frame_area_ratio: float
    matching_geometry_min_visible_ratio: float
    matching_geometry_min_edge_length_px: float
    matching_geometry_max_aspect_ratio: float
    matching_geometry_max_perspective_distortion: float
    matching_bbox_min_width_px: float
    matching_bbox_min_height_px: float
    matching_bbox_min_area_px: float
    matching_bbox_max_frame_area_ratio: float
    matching_confidence_weight_inlier: float
    matching_confidence_weight_similarity: float
    matching_confidence_weight_reprojection: float
    matching_confidence_weight_visibility: float
    matching_confidence_weight_coverage: float
    # "pipeline": mevcut DINOv2 tabanli hat; "gorev3": teknofest_gorev3'ten
    # tasinan kanitlanmis ObjectMatcher (bkz. app/services/matching/gorev3/)
    matching_engine: str = "pipeline"
    # gorev3: politika kutu birakmadiysa pencere-ici dusuk esikli (0.30) en iyi
    # heatmap adayini gonder (asiri bakis acisi farkli referanslar icin)
    matching_gorev3_window_fallback: bool = True

    def validate_detection_motion(self) -> None:
        errors: list[str] = []
        if self.detection_motion_method not in {
            "global_median",
            "homography",
            "homography_bbox",
            "homography_hybrid",
            "homography_local",
            "homography_adaptive",
        }:
            errors.append(
                "DETECTION_MOTION_METHOD global_median, homography, homography_bbox, homography_hybrid, homography_local veya homography_adaptive olmalıdır"
            )
        finite_nonnegative = (
            ("DETECTION_MOTION_THRESHOLD_PX", self.detection_motion_threshold_px),
            ("DETECTION_MOTION_FREEZE_THRESHOLD", self.detection_motion_freeze_threshold),
        )
        for name, value in finite_nonnegative:
            if not math.isfinite(value) or value < 0:
                errors.append(f"{name} sonlu ve negatif olmayan bir sayı olmalıdır")
        if self.detection_motion_min_valid_pixels < 1:
            errors.append("DETECTION_MOTION_MIN_VALID_PIXELS en az 1 olmalıdır")
        if not 0 <= self.detection_motion_inner_crop_ratio < 0.5:
            errors.append("DETECTION_MOTION_INNER_CROP_RATIO 0 dahil, 0.5 hariç aralıkta olmalıdır")
        if self.detection_motion_max_frame_gap < 1:
            errors.append("DETECTION_MOTION_MAX_FRAME_GAP en az 1 olmalıdır")
        if self.detection_motion_warmup_frames < 0:
            errors.append("DETECTION_MOTION_WARMUP_FRAMES negatif olamaz")
        if not math.isfinite(self.detection_motion_flow_downscale) or not 0 < self.detection_motion_flow_downscale <= 1:
            errors.append("DETECTION_MOTION_FLOW_DOWNSCALE 0'dan büyük, 1'den küçük/eşit olmalıdır")
        if not math.isfinite(self.detection_motion_session_ttl_seconds) or self.detection_motion_session_ttl_seconds <= 0:
            errors.append("DETECTION_MOTION_SESSION_TTL_SECONDS pozitif ve sonlu olmalıdır")
        if self.detection_motion_max_sessions < 1:
            errors.append("DETECTION_MOTION_MAX_SESSIONS en az 1 olmalıdır")
        if self.detection_motion_homography_min_features < 4:
            errors.append("DETECTION_MOTION_HOMOGRAPHY_MIN_FEATURES en az 4 olmalıdır")
        if self.detection_motion_homography_min_inliers < 4:
            errors.append("DETECTION_MOTION_HOMOGRAPHY_MIN_INLIERS en az 4 olmalıdır")
        if (
            self.detection_motion_homography_min_inliers
            > self.detection_motion_homography_min_features
        ):
            errors.append(
                "DETECTION_MOTION_HOMOGRAPHY_MIN_INLIERS minimum feature sayısını aşamaz"
            )
        if not math.isfinite(self.detection_motion_homography_min_inlier_ratio) or not (
            0 < self.detection_motion_homography_min_inlier_ratio <= 1
        ):
            errors.append(
                "DETECTION_MOTION_HOMOGRAPHY_MIN_INLIER_RATIO (0, 1] aralığında olmalıdır"
            )
        for name, value in (
            (
                "DETECTION_MOTION_HOMOGRAPHY_RANSAC_THRESHOLD",
                self.detection_motion_homography_ransac_threshold,
            ),
            (
                "DETECTION_MOTION_HOMOGRAPHY_MAX_CONDITION_NUMBER",
                self.detection_motion_homography_max_condition_number,
            ),
            (
                "DETECTION_MOTION_HOMOGRAPHY_RESIDUAL_THRESHOLD_PX",
                self.detection_motion_homography_residual_threshold_px,
            ),
        ):
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{name} pozitif ve sonlu olmalıdır")
        if self.detection_motion_homography_quality_gate not in {"fixed", "adaptive"}:
            errors.append(
                "DETECTION_MOTION_HOMOGRAPHY_QUALITY_GATE fixed veya adaptive olmalıdır"
            )
        adaptive_low = self.detection_motion_homography_adaptive_low_inlier_ratio
        adaptive_high = self.detection_motion_homography_adaptive_high_inlier_ratio
        if not (
            math.isfinite(adaptive_low)
            and math.isfinite(adaptive_high)
            and 0 < adaptive_low < adaptive_high <= 1
        ):
            errors.append(
                "Adaptive homography inlier ratio eşikleri 0 < low < high <= 1 koşulunu sağlamalıdır"
            )
        if (
            not math.isfinite(
                self.detection_motion_homography_adaptive_max_reprojection_error_px
            )
            or self.detection_motion_homography_adaptive_max_reprojection_error_px <= 0
        ):
            errors.append(
                "DETECTION_MOTION_HOMOGRAPHY_ADAPTIVE_MAX_REPROJECTION_ERROR_PX pozitif ve sonlu olmalıdır"
            )
        for name, value in (
            (
                "DETECTION_MOTION_HOMOGRAPHY_ADAPTIVE_MIN_SPATIAL_COVERAGE",
                self.detection_motion_homography_adaptive_min_spatial_coverage,
            ),
            (
                "DETECTION_MOTION_HOMOGRAPHY_ADAPTIVE_MIN_PROJECTED_OVERLAP_RATIO",
                self.detection_motion_homography_adaptive_min_projected_overlap_ratio,
            ),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                errors.append(f"{name} [0, 1] aralığında olmalıdır")
        if (
            not math.isfinite(self.detection_motion_local_ring_expansion_ratio)
            or self.detection_motion_local_ring_expansion_ratio <= 0
        ):
            errors.append(
                "DETECTION_MOTION_LOCAL_RING_EXPANSION_RATIO pozitif ve sonlu olmalıdır"
            )
        if self.detection_motion_local_min_background_pixels < 1:
            errors.append("DETECTION_MOTION_LOCAL_MIN_BACKGROUND_PIXELS en az 1 olmalıdır")
        for name, value in (
            (
                "DETECTION_MOTION_LOCAL_STATIONARY_THRESHOLD_PX",
                self.detection_motion_local_stationary_threshold_px,
            ),
            (
                "DETECTION_MOTION_LOCAL_MOVING_THRESHOLD_PX",
                self.detection_motion_local_moving_threshold_px,
            ),
        ):
            if not math.isfinite(value) or value < 0:
                errors.append(f"{name} sonlu ve negatif olmayan olmalıdır")
        if (
            self.detection_motion_local_moving_threshold_px
            <= self.detection_motion_local_stationary_threshold_px
        ):
            errors.append(
                "DETECTION_MOTION_LOCAL_MOVING_THRESHOLD_PX stationary eşiğinden büyük olmalıdır"
            )
        if (
            not math.isfinite(self.detection_motion_local_min_valid_ratio)
            or not 0 <= self.detection_motion_local_min_valid_ratio <= 1
        ):
            errors.append("DETECTION_MOTION_LOCAL_MIN_VALID_RATIO [0, 1] aralığında olmalıdır")
        for name, value in (
            ("DETECTION_MOTION_BBOX_MATCH_MIN_IOU", self.detection_motion_bbox_match_min_iou),
            ("DETECTION_MOTION_BBOX_MATCH_MIN_SCORE", self.detection_motion_bbox_match_min_score),
            ("DETECTION_MOTION_BBOX_MIN_VISIBLE_RATIO", self.detection_motion_bbox_min_visible_ratio),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                errors.append(f"{name} [0, 1] aralığında olmalıdır")
        for name, value in (
            (
                "DETECTION_MOTION_BBOX_MATCH_MAX_CENTER_DISTANCE_RATIO",
                self.detection_motion_bbox_match_max_center_distance_ratio,
            ),
            (
                "DETECTION_MOTION_BBOX_STATIONARY_THRESHOLD_PX",
                self.detection_motion_bbox_stationary_threshold_px,
            ),
            (
                "DETECTION_MOTION_BBOX_MOVING_THRESHOLD_PX",
                self.detection_motion_bbox_moving_threshold_px,
            ),
            ("DETECTION_MOTION_BBOX_MIN_SIZE_RATIO", self.detection_motion_bbox_min_size_ratio),
            ("DETECTION_MOTION_BBOX_MAX_SIZE_RATIO", self.detection_motion_bbox_max_size_ratio),
        ):
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{name} pozitif ve sonlu olmalıdır")
        if (
            self.detection_motion_bbox_moving_threshold_px
            <= self.detection_motion_bbox_stationary_threshold_px
        ):
            errors.append(
                "DETECTION_MOTION_BBOX_MOVING_THRESHOLD_PX stationary eşiğinden büyük olmalıdır"
            )
        if self.detection_motion_bbox_max_size_ratio < self.detection_motion_bbox_min_size_ratio:
            errors.append(
                "DETECTION_MOTION_BBOX_MAX_SIZE_RATIO minimum size ratio değerinden küçük olamaz"
            )
        for name, value in (
            ("DETECTION_MOTION_ADAPTIVE_BACKGROUND_MEDIAN_MAX", self.detection_motion_adaptive_background_median_max),
            ("DETECTION_MOTION_ADAPTIVE_BACKGROUND_P90_MAX", self.detection_motion_adaptive_background_p90_max),
            ("DETECTION_MOTION_ADAPTIVE_GRID_SPREAD_MAX", self.detection_motion_adaptive_grid_spread_max),
        ):
            if not math.isfinite(value) or value < 0:
                errors.append(f"{name} must be finite and nonnegative")
        if (
            not math.isfinite(self.detection_motion_adaptive_min_valid_background_ratio)
            or not 0 < self.detection_motion_adaptive_min_valid_background_ratio <= 1
        ):
            errors.append("DETECTION_MOTION_ADAPTIVE_MIN_VALID_BACKGROUND_RATIO must be in (0, 1]")
        if (
            not math.isfinite(self.detection_motion_hybrid_strong_moving_residual_px)
            or self.detection_motion_hybrid_strong_moving_residual_px <= 0
        ):
            errors.append(
                "DETECTION_MOTION_HYBRID_STRONG_MOVING_RESIDUAL_PX pozitif ve sonlu olmalıdır"
            )
        for name, value in (
            (
                "DETECTION_MOTION_HYBRID_MIN_ASSOCIATION_SCORE",
                self.detection_motion_hybrid_min_association_score,
            ),
            ("DETECTION_MOTION_HYBRID_MIN_IOU", self.detection_motion_hybrid_min_iou),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                errors.append(f"{name} [0, 1] aralığında olmalıdır")
        if errors:
            raise ValueError("; ".join(errors))

    def validate_detection_landing(self) -> None:
        errors: list[str] = []
        if not math.isfinite(self.detection_landing_edge_margin_px) or self.detection_landing_edge_margin_px < 0:
            errors.append("DETECTION_LANDING_EDGE_MARGIN_PX must be finite and nonnegative")
        if not math.isfinite(self.detection_landing_edge_margin_ratio) or not 0 <= self.detection_landing_edge_margin_ratio < 0.5:
            errors.append("DETECTION_LANDING_EDGE_MARGIN_RATIO must be in [0, 0.5)")
        if not math.isfinite(self.detection_landing_min_intersection_pixels) or self.detection_landing_min_intersection_pixels <= 0:
            errors.append("DETECTION_LANDING_MIN_INTERSECTION_PIXELS must be positive and finite")
        if not math.isfinite(self.detection_landing_occupancy_ratio) or not 0 <= self.detection_landing_occupancy_ratio <= 1:
            errors.append("DETECTION_LANDING_OCCUPANCY_RATIO must be in [0, 1]")
        if not math.isfinite(self.detection_landing_min_area_pixels) or self.detection_landing_min_area_pixels <= 0:
            errors.append("DETECTION_LANDING_MIN_AREA_PIXELS must be positive and finite")
        if errors:
            raise ValueError("; ".join(errors))

    def validate_localization_vo(self) -> None:
        errors: list[str] = []
        if self.localization_min_features < 3:
            errors.append("LOCALIZATION_MIN_FEATURES must be at least 3")
        if self.localization_max_features < self.localization_min_features:
            errors.append("LOCALIZATION_MAX_FEATURES must be >= LOCALIZATION_MIN_FEATURES")
        if not math.isfinite(self.localization_feature_quality_level) or not 0 < self.localization_feature_quality_level <= 1:
            errors.append("LOCALIZATION_FEATURE_QUALITY_LEVEL must be in (0, 1]")
        if not math.isfinite(self.localization_feature_min_distance) or self.localization_feature_min_distance <= 0:
            errors.append("LOCALIZATION_FEATURE_MIN_DISTANCE must be positive and finite")
        if self.localization_lk_win_size < 3 or self.localization_lk_win_size % 2 == 0:
            errors.append("LOCALIZATION_LK_WIN_SIZE must be an odd integer >= 3")
        if self.localization_lk_max_level < 0:
            errors.append("LOCALIZATION_LK_MAX_LEVEL cannot be negative")
        for name, value in (
            ("LOCALIZATION_LK_FB_ERROR_THRESHOLD", self.localization_lk_fb_error_threshold),
            ("LOCALIZATION_RANSAC_RESIDUAL_THRESHOLD", self.localization_ransac_residual_threshold),
            ("LOCALIZATION_FREEZE_THRESHOLD", self.localization_freeze_threshold),
        ):
            if not math.isfinite(value) or value < 0:
                errors.append(f"{name} must be finite and nonnegative")
        if self.localization_ransac_iterations < 1:
            errors.append("LOCALIZATION_RANSAC_ITERATIONS must be at least 1")
        if self.localization_min_inliers < 3:
            errors.append("LOCALIZATION_MIN_INLIERS must be at least 3")
        if self.localization_min_inliers > self.localization_max_features:
            errors.append("LOCALIZATION_MIN_INLIERS cannot exceed LOCALIZATION_MAX_FEATURES")
        if not math.isfinite(self.localization_min_inlier_ratio) or not 0 < self.localization_min_inlier_ratio <= 1:
            errors.append("LOCALIZATION_MIN_INLIER_RATIO must be in (0, 1]")
        if self.localization_max_frame_gap < 1:
            errors.append("LOCALIZATION_MAX_FRAME_GAP must be at least 1")
        if self.localization_warmup_frames < 0:
            errors.append("LOCALIZATION_WARMUP_FRAMES cannot be negative")
        if not math.isfinite(self.localization_session_ttl_seconds) or self.localization_session_ttl_seconds <= 0:
            errors.append("LOCALIZATION_SESSION_TTL_SECONDS must be positive and finite")
        if self.localization_max_sessions < 1:
            errors.append("LOCALIZATION_MAX_SESSIONS must be at least 1")
        if self.localization_camera_width < 1 or self.localization_camera_height < 1:
            errors.append("LOCALIZATION_CAMERA_WIDTH/HEIGHT must be positive")
        for name, value in (
            ("LOCALIZATION_CAMERA_FX", self.localization_camera_fx),
            ("LOCALIZATION_CAMERA_FY", self.localization_camera_fy),
        ):
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{name} must be positive and finite")
        if not math.isfinite(self.localization_camera_cx) or not 0 <= self.localization_camera_cx <= self.localization_camera_width:
            errors.append("LOCALIZATION_CAMERA_CX must be inside image width")
        if not math.isfinite(self.localization_camera_cy) or not 0 <= self.localization_camera_cy <= self.localization_camera_height:
            errors.append("LOCALIZATION_CAMERA_CY must be inside image height")
        try:
            distortion = [float(item.strip()) for item in self.localization_camera_distortion.split(",") if item.strip()]
            if not all(math.isfinite(item) for item in distortion):
                raise ValueError
        except ValueError:
            errors.append("LOCALIZATION_CAMERA_DISTORTION must contain finite comma-separated numbers")
        if self.localization_camera_calibration_path is not None and not self.localization_camera_calibration_path.is_file():
            errors.append("LOCALIZATION_CAMERA_CALIBRATION_PATH must point to an existing file")
        if self.localization_calibration_min_samples < 3:
            errors.append("LOCALIZATION_CALIBRATION_MIN_SAMPLES must be at least 3")
        if self.localization_calibration_max_samples < self.localization_calibration_min_samples:
            errors.append("LOCALIZATION_CALIBRATION_MAX_SAMPLES must be >= minimum samples")
        for name, value in (
            ("LOCALIZATION_CALIBRATION_MIN_CAMERA_STEP_PX", self.localization_calibration_min_camera_step_px),
            ("LOCALIZATION_CALIBRATION_MIN_GPS_STEP", self.localization_calibration_min_gps_step),
            ("LOCALIZATION_CALIBRATION_MAX_RMS_RESIDUAL", self.localization_calibration_max_rms_residual),
            ("LOCALIZATION_CALIBRATION_MIN_DIRECTIONAL_DIVERSITY", self.localization_calibration_min_directional_diversity),
            ("LOCALIZATION_CALIBRATION_SCALE_MIN", self.localization_calibration_scale_min),
            ("LOCALIZATION_CALIBRATION_SCALE_MAX", self.localization_calibration_scale_max),
            ("LOCALIZATION_CALIBRATION_OUTLIER_MAD_FACTOR", self.localization_calibration_outlier_mad_factor),
            ("LOCALIZATION_MAX_DELTA_PER_FRAME", self.localization_max_delta_per_frame),
        ):
            if not math.isfinite(value) or value < 0:
                errors.append(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.localization_calibration_min_inlier_ratio) or not 0 < self.localization_calibration_min_inlier_ratio <= 1:
            errors.append("LOCALIZATION_CALIBRATION_MIN_INLIER_RATIO must be in (0, 1]")
        if not 0 <= self.localization_calibration_min_directional_diversity <= 1:
            errors.append("LOCALIZATION_CALIBRATION_MIN_DIRECTIONAL_DIVERSITY must be in [0, 1]")
        if self.localization_calibration_scale_min <= 0 or self.localization_calibration_scale_max <= self.localization_calibration_scale_min:
            errors.append("LOCALIZATION_CALIBRATION_SCALE_MIN/MAX range is invalid")
        if self.localization_calibration_outlier_mad_factor <= 0:
            errors.append("LOCALIZATION_CALIBRATION_OUTLIER_MAD_FACTOR must be positive")
        if self.localization_calibration_expected_max_frame < 1:
            errors.append("LOCALIZATION_CALIBRATION_EXPECTED_MAX_FRAME must be at least 1")
        if self.localization_z_policy not in {"hold_last_valid_z", "zero_delta_from_anchor", "return_none_if_schema_allows"}:
            errors.append("LOCALIZATION_Z_POLICY is unsupported")
        if self.localization_recovery_min_healthy_frames < 1:
            errors.append("LOCALIZATION_RECOVERY_MIN_HEALTHY_FRAMES must be at least 1")
        if errors:
            raise ValueError("; ".join(errors))

    def validate_matching_local(self) -> None:
        errors: list[str] = []
        if self.matching_geometry_method not in {"dinov2", "aliked_lightglue", "hybrid"}:
            errors.append(
                "MATCHING_GEOMETRY_METHOD dinov2, aliked_lightglue veya hybrid olmalidir"
            )
        if self.matching_aliked_device not in {"auto", "cpu", "cuda"}:
            errors.append("MATCHING_ALIKED_DEVICE auto, cpu veya cuda olmalidir")
        if self.matching_lightglue_device not in {"auto", "cpu", "cuda"}:
            errors.append("MATCHING_LIGHTGLUE_DEVICE auto, cpu veya cuda olmalidir")
        if self.matching_local_min_keypoints < 4:
            errors.append("MATCHING_LOCAL_MIN_KEYPOINTS en az 4 olmalidir")
        if self.matching_local_min_matches < 4:
            errors.append("MATCHING_LOCAL_MIN_MATCHES en az 4 olmalidir")
        if self.matching_local_min_inliers < 4:
            errors.append("MATCHING_LOCAL_MIN_INLIERS en az 4 olmalidir")
        if self.matching_local_min_inliers > self.matching_local_min_matches:
            errors.append("MATCHING_LOCAL_MIN_INLIERS match esigini asamaz")
        if not math.isfinite(self.matching_local_min_inlier_ratio) or not (
            0 <= self.matching_local_min_inlier_ratio <= 1
        ):
            errors.append("MATCHING_LOCAL_MIN_INLIER_RATIO [0, 1] araliginda olmalidir")
        if (
            not math.isfinite(self.matching_local_max_reprojection_error)
            or self.matching_local_max_reprojection_error <= 0
        ):
            errors.append("MATCHING_LOCAL_MAX_REPROJECTION_ERROR pozitif ve sonlu olmalidir")
        if (
            not math.isfinite(self.matching_local_refinement_timeout_sec)
            or self.matching_local_refinement_timeout_sec <= 0
        ):
            errors.append("MATCHING_LOCAL_REFINEMENT_TIMEOUT_SEC pozitif ve sonlu olmalidir")
        if errors:
            raise ValueError("; ".join(errors))

    def validate_matching_xoftr(self) -> None:
        if not self.matching_xoftr_enabled:
            return
        errors: list[str] = []
        if self.matching_xoftr_repo_path is None or not self.matching_xoftr_repo_path.is_dir():
            errors.append("MATCHING_XOFTR_REPO_PATH mevcut bir klasor olmalidir")
        if self.matching_xoftr_ckpt_path is None or not self.matching_xoftr_ckpt_path.is_file():
            errors.append("MATCHING_XOFTR_CKPT_PATH mevcut bir dosya olmalidir")
        if self.matching_xoftr_device not in {"auto", "cpu", "cuda"}:
            errors.append("MATCHING_XOFTR_DEVICE auto, cpu veya cuda olmalidir")
        if self.matching_xoftr_max_edge < 32:
            errors.append("MATCHING_XOFTR_MAX_EDGE en az 32 olmalidir")
        if (
            not math.isfinite(self.matching_xoftr_timeout_seconds)
            or self.matching_xoftr_timeout_seconds <= 0
        ):
            errors.append("MATCHING_XOFTR_TIMEOUT_SECONDS pozitif olmalidir")
        if self.matching_xoftr_min_inliers < 4:
            errors.append("MATCHING_XOFTR_MIN_INLIERS en az 4 olmalidir")
        if errors:
            raise ValueError("; ".join(errors))

    def validate_official_integration(self) -> None:
        missing = [
            name
            for name, value in (
                ("TEAM_NAME", self.team_name),
                ("PASSWORD", self.password),
                ("EVALUATION_SERVER_URL", self.evaluation_server_url),
                ("SESSION_NAME", self.official_session_name),
                ("OFFICIAL_INTERFACE_PATH", self.official_interface_path),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Resmî yarışma entegrasyonu için eksik ortam değişkenleri: "
                + ", ".join(missing)
            )
        parsed = urlparse(self.evaluation_server_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("EVALUATION_SERVER_URL geçerli bir http/https URL olmalıdır.")
        if not self.official_interface_path.is_dir():
            raise RuntimeError("OFFICIAL_INTERFACE_PATH mevcut bir klasör olmalıdır.")
        if self.competition_max_retries < 1:
            raise RuntimeError("COMPETITION_MAX_RETRIES en az 1 olmalıdır.")
        if self.competition_retry_initial_seconds < 0:
            raise RuntimeError("COMPETITION_RETRY_INITIAL_SECONDS negatif olamaz.")
        if self.competition_frame_interval_seconds < 0:
            raise RuntimeError("COMPETITION_FRAME_INTERVAL_SECONDS negatif olamaz.")
        if self.competition_task_timeout_seconds <= 0:
            raise RuntimeError("COMPETITION_TASK_TIMEOUT_SECONDS pozitif olmalıdır.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        environment=os.getenv("ENVIRONMENT", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        api_key=os.getenv("AEROSYNC_SECRET_KEY", ""),
        team_user_url=os.getenv("TEAM_USER_URL", ""),
        backend_url=os.getenv("LOCAL_BACKEND_URL", "http://127.0.0.1:8000"),
        competition_url=os.getenv("COMPETITION_URL", ""),
        http_timeout_seconds=_float("HTTP_TIMEOUT_SECONDS", 5.0),
        session_duration_seconds=_float("SESSION_DURATION_SECONDS", 3600.0),
        team_name=os.getenv("TEAM_NAME", "").strip(),
        password=os.getenv("PASSWORD", ""),
        evaluation_server_url=_normalized_base_url("EVALUATION_SERVER_URL"),
        official_session_name=os.getenv("SESSION_NAME", "").strip(),
        official_interface_path=_project_path("OFFICIAL_INTERFACE_PATH"),
        official_media_dir=_path("OFFICIAL_MEDIA_DIR") or Path("work/official_media").resolve(),
        competition_max_retries=_int("COMPETITION_MAX_RETRIES", 5),
        competition_retry_initial_seconds=_float(
            "COMPETITION_RETRY_INITIAL_SECONDS", 0.25
        ),
        competition_frame_interval_seconds=_float(
            "COMPETITION_FRAME_INTERVAL_SECONDS", 0.25
        ),
        competition_task_timeout_seconds=_float(
            "COMPETITION_TASK_TIMEOUT_SECONDS", 30.0
        ),
        detection_enabled=_bool("DETECTION_ENABLED", False),
        detection_model_path=_path("DETECTION_MODEL_PATH"),
        detection_confidence=_float("DETECTION_CONFIDENCE", 0.25),
        detection_iou=_float("DETECTION_IOU", 0.45),
        detection_motion_enabled=_bool("DETECTION_MOTION_ENABLED", True),
        detection_motion_method=os.getenv(
            "DETECTION_MOTION_METHOD", "global_median"
        ).strip().lower(),
        detection_motion_threshold_px=_float("DETECTION_MOTION_THRESHOLD_PX", 2.0),
        detection_motion_min_valid_pixels=_int("DETECTION_MOTION_MIN_VALID_PIXELS", 25),
        detection_motion_inner_crop_ratio=_float("DETECTION_MOTION_INNER_CROP_RATIO", 0.15),
        detection_motion_max_frame_gap=_int("DETECTION_MOTION_MAX_FRAME_GAP", 1),
        detection_motion_warmup_frames=_int("DETECTION_MOTION_WARMUP_FRAMES", 1),
        detection_motion_flow_downscale=_float("DETECTION_MOTION_FLOW_DOWNSCALE", 0.5),
        detection_motion_freeze_threshold=_float("DETECTION_MOTION_FREEZE_THRESHOLD", 0.0),
        detection_motion_session_ttl_seconds=_float("DETECTION_MOTION_SESSION_TTL_SECONDS", 1800.0),
        detection_motion_max_sessions=_int("DETECTION_MOTION_MAX_SESSIONS", 32),
        detection_motion_homography_min_features=_int(
            "DETECTION_MOTION_HOMOGRAPHY_MIN_FEATURES", 40
        ),
        detection_motion_homography_min_inliers=_int(
            "DETECTION_MOTION_HOMOGRAPHY_MIN_INLIERS", 20
        ),
        detection_motion_homography_min_inlier_ratio=_float(
            "DETECTION_MOTION_HOMOGRAPHY_MIN_INLIER_RATIO", 0.5
        ),
        detection_motion_homography_ransac_threshold=_float(
            "DETECTION_MOTION_HOMOGRAPHY_RANSAC_THRESHOLD", 3.0
        ),
        detection_motion_homography_max_condition_number=_float(
            "DETECTION_MOTION_HOMOGRAPHY_MAX_CONDITION_NUMBER", 100000.0
        ),
        detection_motion_homography_residual_threshold_px=_float(
            "DETECTION_MOTION_HOMOGRAPHY_RESIDUAL_THRESHOLD_PX", 2.0
        ),
        detection_motion_homography_quality_gate=os.getenv(
            "DETECTION_MOTION_HOMOGRAPHY_QUALITY_GATE", "fixed"
        ).strip().lower(),
        detection_motion_homography_adaptive_high_inlier_ratio=_float(
            "DETECTION_MOTION_HOMOGRAPHY_ADAPTIVE_HIGH_INLIER_RATIO", 0.50
        ),
        detection_motion_homography_adaptive_low_inlier_ratio=_float(
            "DETECTION_MOTION_HOMOGRAPHY_ADAPTIVE_LOW_INLIER_RATIO", 0.35
        ),
        detection_motion_homography_adaptive_max_reprojection_error_px=_float(
            "DETECTION_MOTION_HOMOGRAPHY_ADAPTIVE_MAX_REPROJECTION_ERROR_PX", 3.0
        ),
        detection_motion_homography_adaptive_min_spatial_coverage=_float(
            "DETECTION_MOTION_HOMOGRAPHY_ADAPTIVE_MIN_SPATIAL_COVERAGE", 0.08
        ),
        detection_motion_homography_adaptive_min_projected_overlap_ratio=_float(
            "DETECTION_MOTION_HOMOGRAPHY_ADAPTIVE_MIN_PROJECTED_OVERLAP_RATIO", 0.50
        ),
        detection_motion_bbox_match_min_iou=_float(
            "DETECTION_MOTION_BBOX_MATCH_MIN_IOU", 0.10
        ),
        detection_motion_bbox_match_max_center_distance_ratio=_float(
            "DETECTION_MOTION_BBOX_MATCH_MAX_CENTER_DISTANCE_RATIO", 1.5
        ),
        detection_motion_bbox_match_min_score=_float(
            "DETECTION_MOTION_BBOX_MATCH_MIN_SCORE", 0.25
        ),
        detection_motion_bbox_stationary_threshold_px=_float(
            "DETECTION_MOTION_BBOX_STATIONARY_THRESHOLD_PX", 3.0
        ),
        detection_motion_bbox_moving_threshold_px=_float(
            "DETECTION_MOTION_BBOX_MOVING_THRESHOLD_PX", 8.0
        ),
        detection_motion_bbox_min_size_ratio=_float(
            "DETECTION_MOTION_BBOX_MIN_SIZE_RATIO", 0.5
        ),
        detection_motion_bbox_max_size_ratio=_float(
            "DETECTION_MOTION_BBOX_MAX_SIZE_RATIO", 2.0
        ),
        detection_motion_bbox_min_visible_ratio=_float(
            "DETECTION_MOTION_BBOX_MIN_VISIBLE_RATIO", 0.75
        ),
        detection_motion_adaptive_background_median_max=_float(
            "DETECTION_MOTION_ADAPTIVE_BACKGROUND_MEDIAN_MAX", 2.0
        ),
        detection_motion_adaptive_background_p90_max=_float(
            "DETECTION_MOTION_ADAPTIVE_BACKGROUND_P90_MAX", 5.0
        ),
        detection_motion_adaptive_grid_spread_max=_float(
            "DETECTION_MOTION_ADAPTIVE_GRID_SPREAD_MAX", 3.0
        ),
        detection_motion_adaptive_min_valid_background_ratio=_float(
            "DETECTION_MOTION_ADAPTIVE_MIN_VALID_BACKGROUND_RATIO", 0.20
        ),
        detection_motion_hybrid_strong_moving_residual_px=_float(
            "DETECTION_MOTION_HYBRID_STRONG_MOVING_RESIDUAL_PX", 8.0
        ),
        detection_motion_hybrid_min_association_score=_float(
            "DETECTION_MOTION_HYBRID_MIN_ASSOCIATION_SCORE", 0.35
        ),
        detection_motion_hybrid_min_iou=_float(
            "DETECTION_MOTION_HYBRID_MIN_IOU", 0.20
        ),
        detection_motion_local_ring_expansion_ratio=_float(
            "DETECTION_MOTION_LOCAL_RING_EXPANSION_RATIO", 0.50
        ),
        detection_motion_local_min_background_pixels=_int(
            "DETECTION_MOTION_LOCAL_MIN_BACKGROUND_PIXELS", 100
        ),
        detection_motion_local_stationary_threshold_px=_float(
            "DETECTION_MOTION_LOCAL_STATIONARY_THRESHOLD_PX", 2.0
        ),
        detection_motion_local_moving_threshold_px=_float(
            "DETECTION_MOTION_LOCAL_MOVING_THRESHOLD_PX", 6.0
        ),
        detection_motion_local_min_valid_ratio=_float(
            "DETECTION_MOTION_LOCAL_MIN_VALID_RATIO", 0.50
        ),
        detection_landing_enabled=_bool("DETECTION_LANDING_ENABLED", True),
        detection_landing_edge_margin_px=_float("DETECTION_LANDING_EDGE_MARGIN_PX", 2.0),
        detection_landing_edge_margin_ratio=_float("DETECTION_LANDING_EDGE_MARGIN_RATIO", 0.001),
        detection_landing_min_intersection_pixels=_float("DETECTION_LANDING_MIN_INTERSECTION_PIXELS", 16.0),
        detection_landing_occupancy_ratio=_float("DETECTION_LANDING_OCCUPANCY_RATIO", 0.01),
        detection_landing_use_center_check=_bool("DETECTION_LANDING_USE_CENTER_CHECK", True),
        detection_landing_use_bottom_center_check=_bool("DETECTION_LANDING_USE_BOTTOM_CENTER_CHECK", True),
        detection_landing_min_area_pixels=_float("DETECTION_LANDING_MIN_AREA_PIXELS", 64.0),
        localization_enabled=_bool("LOCALIZATION_ENABLED", False),
        localization_model_path=_path("LOCALIZATION_MODEL_PATH"),
        localization_vo_enabled=_bool("LOCALIZATION_VO_ENABLED", True),
        localization_min_features=_int("LOCALIZATION_MIN_FEATURES", 12),
        localization_max_features=_int("LOCALIZATION_MAX_FEATURES", 300),
        localization_feature_quality_level=_float("LOCALIZATION_FEATURE_QUALITY_LEVEL", 0.01),
        localization_feature_min_distance=_float("LOCALIZATION_FEATURE_MIN_DISTANCE", 7.0),
        localization_lk_win_size=_int("LOCALIZATION_LK_WIN_SIZE", 21),
        localization_lk_max_level=_int("LOCALIZATION_LK_MAX_LEVEL", 3),
        localization_lk_fb_error_threshold=_float("LOCALIZATION_LK_FB_ERROR_THRESHOLD", 1.0),
        localization_ransac_iterations=_int("LOCALIZATION_RANSAC_ITERATIONS", 200),
        localization_ransac_residual_threshold=_float("LOCALIZATION_RANSAC_RESIDUAL_THRESHOLD", 2.0),
        localization_min_inliers=_int("LOCALIZATION_MIN_INLIERS", 8),
        localization_min_inlier_ratio=_float("LOCALIZATION_MIN_INLIER_RATIO", 0.5),
        localization_max_frame_gap=_int("LOCALIZATION_MAX_FRAME_GAP", 1),
        localization_warmup_frames=_int("LOCALIZATION_WARMUP_FRAMES", 1),
        localization_freeze_threshold=_float("LOCALIZATION_FREEZE_THRESHOLD", 0.0),
        localization_session_ttl_seconds=_float("LOCALIZATION_SESSION_TTL_SECONDS", 1800.0),
        localization_max_sessions=_int("LOCALIZATION_MAX_SESSIONS", 32),
        localization_camera_width=_int("LOCALIZATION_CAMERA_WIDTH", 1920),
        localization_camera_height=_int("LOCALIZATION_CAMERA_HEIGHT", 1080),
        localization_camera_fx=_float("LOCALIZATION_CAMERA_FX", 1389.7),
        localization_camera_fy=_float("LOCALIZATION_CAMERA_FY", 1387.1),
        localization_camera_cx=_float("LOCALIZATION_CAMERA_CX", 954.007),
        localization_camera_cy=_float("LOCALIZATION_CAMERA_CY", 558.896),
        localization_camera_distortion=os.getenv("LOCALIZATION_CAMERA_DISTORTION", "0,0,0,0,0"),
        localization_camera_calibration_path=_path("LOCALIZATION_CAMERA_CALIBRATION_PATH"),
        localization_calibration_enabled=_bool("LOCALIZATION_CALIBRATION_ENABLED", True),
        localization_calibration_min_samples=_int("LOCALIZATION_CALIBRATION_MIN_SAMPLES", 8),
        localization_calibration_max_samples=_int("LOCALIZATION_CALIBRATION_MAX_SAMPLES", 450),
        localization_calibration_min_camera_step_px=_float("LOCALIZATION_CALIBRATION_MIN_CAMERA_STEP_PX", 1.0),
        localization_calibration_min_gps_step=_float("LOCALIZATION_CALIBRATION_MIN_GPS_STEP", 0.02),
        localization_calibration_max_rms_residual=_float("LOCALIZATION_CALIBRATION_MAX_RMS_RESIDUAL", 0.5),
        localization_calibration_min_inlier_ratio=_float("LOCALIZATION_CALIBRATION_MIN_INLIER_RATIO", 0.7),
        localization_calibration_min_directional_diversity=_float("LOCALIZATION_CALIBRATION_MIN_DIRECTIONAL_DIVERSITY", 0.1),
        localization_calibration_scale_min=_float("LOCALIZATION_CALIBRATION_SCALE_MIN", 0.000001),
        localization_calibration_scale_max=_float("LOCALIZATION_CALIBRATION_SCALE_MAX", 10.0),
        localization_calibration_outlier_mad_factor=_float("LOCALIZATION_CALIBRATION_OUTLIER_MAD_FACTOR", 3.5),
        localization_calibration_expected_max_frame=_int("LOCALIZATION_CALIBRATION_EXPECTED_MAX_FRAME", 450),
        localization_allow_reflection=_bool("LOCALIZATION_ALLOW_REFLECTION", False),
        localization_max_delta_per_frame=_float("LOCALIZATION_MAX_DELTA_PER_FRAME", 5.0),
        localization_z_policy=os.getenv("LOCALIZATION_Z_POLICY", "hold_last_valid_z").strip(),
        localization_recovery_min_healthy_frames=_int("LOCALIZATION_RECOVERY_MIN_HEALTHY_FRAMES", 1),
        matching_enabled=_bool("MATCHING_ENABLED", True),
        matching_engine=os.getenv("MATCHING_ENGINE", "pipeline").strip().lower(),
        matching_gorev3_window_fallback=_bool("MATCHING_GOREV3_WINDOW_FALLBACK", True),
        matching_timeout_seconds=_float("MATCHING_TIMEOUT_SECONDS", 5.0),
        matching_device=os.getenv("MATCHING_DEVICE", "auto"),
        matching_allow_cpu_fallback=_bool("MATCHING_ALLOW_CPU_FALLBACK", True),
        matching_dinov2_repo_path=_path("MATCHING_DINOV2_REPO_PATH"),
        matching_dinov2_weights_path=_first_path(
            "MATCHING_DINOV2_WEIGHTS_PATH", "DINOV2_MODEL_PATH"
        ),
        matching_dinov2_model_name=os.getenv("MATCHING_DINOV2_MODEL_NAME", "dinov2_vitb14"),
        matching_aliked_weights_path=_path("MATCHING_ALIKED_WEIGHTS_PATH"),
        matching_lightglue_weights_path=_first_path(
            "LIGHTGLUE_MODEL_PATH", "MATCHING_LIGHTGLUE_WEIGHTS_PATH"
        ),
        matching_xoftr_model_path=_first_path(
            "XOFTR_MODEL_PATH", "MATCHING_XOFTR_MODEL_PATH"
        ),
        matching_xoftr_enabled=_bool("MATCHING_XOFTR_ENABLED", False),
        matching_xoftr_repo_path=_project_path("MATCHING_XOFTR_REPO_PATH"),
        matching_xoftr_ckpt_path=_project_path("MATCHING_XOFTR_CKPT_PATH"),
        matching_xoftr_device=os.getenv("MATCHING_XOFTR_DEVICE", "auto").strip().lower(),
        matching_xoftr_max_edge=_int("MATCHING_XOFTR_MAX_EDGE", 840),
        matching_xoftr_timeout_seconds=_float("MATCHING_XOFTR_TIMEOUT_SECONDS", 10.0),
        matching_xoftr_min_inliers=_int("MATCHING_XOFTR_MIN_INLIERS", 12),
        matching_xoftr_rotation_sweep=_bool("MATCHING_XOFTR_ROTATION_SWEEP", True),
        matching_local_refinement_enabled=_bool("MATCHING_LOCAL_REFINEMENT_ENABLED", True),
        matching_geometry_method=os.getenv("MATCHING_GEOMETRY_METHOD", "dinov2").strip().lower(),
        matching_aliked_model_path=_first_path(
            "MATCHING_ALIKED_MODEL_PATH", "MATCHING_ALIKED_WEIGHTS_PATH"
        ),
        matching_aliked_device=os.getenv("MATCHING_ALIKED_DEVICE", "auto").strip().lower(),
        matching_lightglue_model_path=_first_path(
            "MATCHING_LIGHTGLUE_MODEL_PATH", "MATCHING_LIGHTGLUE_WEIGHTS_PATH", "LIGHTGLUE_MODEL_PATH"
        ),
        matching_lightglue_device=os.getenv("MATCHING_LIGHTGLUE_DEVICE", "auto").strip().lower(),
        matching_local_min_keypoints=_int("MATCHING_LOCAL_MIN_KEYPOINTS", 64),
        matching_local_min_matches=_int("MATCHING_LOCAL_MIN_MATCHES", 20),
        matching_local_min_inliers=_int("MATCHING_LOCAL_MIN_INLIERS", 12),
        matching_local_min_inlier_ratio=_float("MATCHING_LOCAL_MIN_INLIER_RATIO", 0.35),
        matching_local_max_reprojection_error=_float(
            "MATCHING_LOCAL_MAX_REPROJECTION_ERROR", 4.0
        ),
        matching_local_fallback_to_dinov2=_bool(
            "MATCHING_LOCAL_FALLBACK_TO_DINOV2", True
        ),
        matching_local_refinement_timeout_sec=_float(
            "MATCHING_LOCAL_REFINEMENT_TIMEOUT_SEC", 5.0
        ),
        matching_preload_models=_bool("MATCHING_PRELOAD_MODELS", True),
        matching_warmup_enabled=_bool("MATCHING_WARMUP_ENABLED", True),
        matching_min_confidence=_float("MATCHING_MIN_CONFIDENCE", 0.35),
        matching_min_inliers=int(os.getenv("MATCHING_MIN_INLIERS", "6")),
        matching_min_bbox_area=_float("MATCHING_MIN_BBOX_AREA", 64.0),
        matching_max_bbox_area_ratio=_float("MATCHING_MAX_BBOX_AREA_RATIO", 0.50),
        matching_similarity_threshold=_float("MATCHING_SIMILARITY_THRESHOLD", 0.20),
        matching_max_image_edge=int(os.getenv("MATCHING_MAX_IMAGE_EDGE", "1120")),
        matching_reference_ttl_seconds=_float("MATCHING_REFERENCE_TTL_SECONDS", 3600.0),
        matching_max_reference_sessions=_int("MATCHING_MAX_REFERENCE_SESSIONS", 16),
        matching_reference_hash_enabled=_bool("MATCHING_REFERENCE_HASH_ENABLED", True),
        matching_reference_cache_enabled=_bool("MATCHING_REFERENCE_CACHE_ENABLED", True),
        matching_dinov2_enabled=_bool("MATCHING_DINOV2_ENABLED", False),
        matching_dinov2_device=os.getenv("MATCHING_DINOV2_DEVICE", "auto"),
        matching_dinov2_max_long_edge=_int("MATCHING_DINOV2_MAX_LONG_EDGE", 1120),
        matching_dinov2_patch_size=_int("MATCHING_DINOV2_PATCH_SIZE", 14),
        matching_dinov2_descriptor_dtype=os.getenv(
            "MATCHING_DINOV2_DESCRIPTOR_DTYPE", "float32"
        ).strip().lower(),
        matching_dinov2_allow_cpu_fallback=_bool(
            "MATCHING_DINOV2_ALLOW_CPU_FALLBACK", True
        ),
        matching_dinov2_normalize_descriptors=_bool(
            "MATCHING_DINOV2_NORMALIZE_DESCRIPTORS", True
        ),
        matching_dinov2_max_cached_references=_int(
            "MATCHING_DINOV2_MAX_CACHED_REFERENCES", 32
        ),
        matching_dinov2_timeout_seconds=_float("MATCHING_DINOV2_TIMEOUT_SECONDS", 5.0),
        matching_dinov2_cache_device=os.getenv(
            "MATCHING_DINOV2_CACHE_DEVICE", "cpu"
        ).strip().lower(),
        matching_dinov2_max_cache_mb=_float("MATCHING_DINOV2_MAX_CACHE_MB", 512.0),
        matching_coarse_min_similarity=_float("MATCHING_COARSE_MIN_SIMILARITY", 0.45),
        matching_coarse_min_correspondences=_int("MATCHING_COARSE_MIN_CORRESPONDENCES", 6),
        matching_coarse_max_correspondences=_int("MATCHING_COARSE_MAX_CORRESPONDENCES", 512),
        matching_coarse_topk_per_reference=_int("MATCHING_COARSE_TOPK_PER_REFERENCE", 1),
        matching_coarse_chunk_size=_int("MATCHING_COARSE_CHUNK_SIZE", 1024),
        matching_coarse_spatial_dedup_radius_px=_float(
            "MATCHING_COARSE_SPATIAL_DEDUP_RADIUS_PX", 4.0
        ),
        matching_coarse_timeout_seconds=_float("MATCHING_COARSE_TIMEOUT_SECONDS", 2.0),
        matching_reference_timeout_seconds=_float("MATCHING_REFERENCE_TIMEOUT_SECONDS", 3.0),
        matching_homography_method=os.getenv(
            "MATCHING_HOMOGRAPHY_METHOD", "USAC_MAGSAC"
        ).strip().upper(),
        matching_homography_reprojection_threshold=_float(
            "MATCHING_HOMOGRAPHY_REPROJECTION_THRESHOLD", 6.0
        ),
        matching_homography_confidence=_float("MATCHING_HOMOGRAPHY_CONFIDENCE", 0.999),
        matching_homography_max_iterations=_int(
            "MATCHING_HOMOGRAPHY_MAX_ITERATIONS", 10000
        ),
        matching_homography_min_inliers=_int("MATCHING_HOMOGRAPHY_MIN_INLIERS", 6),
        matching_homography_min_inlier_ratio=_float(
            "MATCHING_HOMOGRAPHY_MIN_INLIER_RATIO", 0.35
        ),
        matching_homography_max_rms_reprojection_error=_float(
            "MATCHING_HOMOGRAPHY_MAX_RMS_REPROJECTION_ERROR", 8.0
        ),
        matching_geometry_min_projected_area_px=_float(
            "MATCHING_GEOMETRY_MIN_PROJECTED_AREA_PX", 64.0
        ),
        matching_geometry_max_frame_area_ratio=_float(
            "MATCHING_GEOMETRY_MAX_FRAME_AREA_RATIO", 0.75
        ),
        matching_geometry_min_visible_ratio=_float(
            "MATCHING_GEOMETRY_MIN_VISIBLE_RATIO", 0.25
        ),
        matching_geometry_min_edge_length_px=_float(
            "MATCHING_GEOMETRY_MIN_EDGE_LENGTH_PX", 4.0
        ),
        matching_geometry_max_aspect_ratio=_float(
            "MATCHING_GEOMETRY_MAX_ASPECT_RATIO", 10.0
        ),
        matching_geometry_max_perspective_distortion=_float(
            "MATCHING_GEOMETRY_MAX_PERSPECTIVE_DISTORTION", 8.0
        ),
        matching_bbox_min_width_px=_float("MATCHING_BBOX_MIN_WIDTH_PX", 4.0),
        matching_bbox_min_height_px=_float("MATCHING_BBOX_MIN_HEIGHT_PX", 4.0),
        matching_bbox_min_area_px=_float("MATCHING_BBOX_MIN_AREA_PX", 64.0),
        matching_bbox_max_frame_area_ratio=_float(
            "MATCHING_BBOX_MAX_FRAME_AREA_RATIO", 0.75
        ),
        matching_confidence_weight_inlier=_float(
            "MATCHING_CONFIDENCE_WEIGHT_INLIER", 0.30
        ),
        matching_confidence_weight_similarity=_float(
            "MATCHING_CONFIDENCE_WEIGHT_SIMILARITY", 0.25
        ),
        matching_confidence_weight_reprojection=_float(
            "MATCHING_CONFIDENCE_WEIGHT_REPROJECTION", 0.15
        ),
        matching_confidence_weight_visibility=_float(
            "MATCHING_CONFIDENCE_WEIGHT_VISIBILITY", 0.15
        ),
        matching_confidence_weight_coverage=_float(
            "MATCHING_CONFIDENCE_WEIGHT_COVERAGE", 0.15
        ),
    )
