from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, replace

from app.core.config import Settings
from app.schemas import MatchedReferenceObject
from app.services.matching.aliked_runtime import AlikedRuntime, AlikedRuntimeRegistry
from app.services.matching.bbox_validator import ProjectedBoundingBoxValidator
from app.services.matching.coarse_matcher import CoarseMatcher
from app.services.matching.descriptor_types import CoarseMatchSet, DenseDescriptorSet
from app.services.matching.geometry import ConfidenceScorer, HomographyEstimator, ProjectedPolygonValidator
from app.services.matching.lightglue_runtime import LightGlueRuntime, LightGlueRuntimeRegistry
from app.services.matching.local_features import (
    LocalArtifactUnavailable,
    LocalFeatureSet,
    LocalRefinementDiagnostics,
)

logger = logging.getLogger(__name__)


class ReferenceLocalFeatureCache:
    """Session-owned cache keyed by reference image and ALIKED artifact hashes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[tuple[str, str], tuple[LocalFeatureSet, object]] = {}

    def get(self, image_hash: str, model_hash: str):
        with self._lock:
            return self._entries.get((image_hash, model_hash))

    def put(self, image_hash: str, model_hash: str, features, metrics) -> None:
        with self._lock:
            self._entries[(image_hash, model_hash)] = (features, metrics)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


@dataclass(frozen=True, slots=True)
class LocalPipelineResult:
    matched: MatchedReferenceObject | None
    diagnostics: LocalRefinementDiagnostics


class LocalRefinementPipeline:
    """DINO candidate gate followed by ALIKED + LightGlue local geometry."""

    def __init__(
        self,
        settings: Settings,
        *,
        aliked_factory=AlikedRuntimeRegistry.get,
        lightglue_factory=LightGlueRuntimeRegistry.get,
    ) -> None:
        settings.validate_matching_local()
        self._settings = settings
        self._coarse = CoarseMatcher(settings)
        self._homography = HomographyEstimator(
            settings,
            min_inliers=settings.matching_local_min_inliers,
            min_inlier_ratio=settings.matching_local_min_inlier_ratio,
            max_rms_reprojection_error=settings.matching_local_max_reprojection_error,
        )
        self._polygon = ProjectedPolygonValidator(settings)
        self._bbox = ProjectedBoundingBoxValidator(settings)
        # Keep the wire-format scorer; use the local RMS limit for its reprojection normalization.
        self._confidence = ConfidenceScorer(
            replace(
                settings,
                matching_homography_max_rms_reprojection_error=
                settings.matching_local_max_reprojection_error,
            )
        )
        self._aliked_factory = aliked_factory
        self._lightglue_factory = lightglue_factory

    def prepare_reference(self, image, reference_hash: str):
        """Compute a reference's local features without running any matching."""
        aliked: AlikedRuntime = self._aliked_factory(self._settings)
        features, metrics = aliked.extract(image, reference_hash)
        return aliked.model_hash, features, metrics

    def match_reference(
        self,
        *,
        method: str,
        object_id: int,
        reference_descriptor: DenseDescriptorSet,
        frame_descriptor: DenseDescriptorSet,
        reference_image,
        frame_image,
        reference_hash: str,
        frame_hash: str,
        cache: ReferenceLocalFeatureCache,
    ) -> LocalPipelineResult:
        started = time.perf_counter()
        coarse = self._coarse.match(reference_descriptor, frame_descriptor)
        base = LocalRefinementDiagnostics(
            method=method,
            accepted=False,
            reason=coarse.failure_reason or "local_refinement_pending",
            coarse_correspondence_count=coarse.correspondence_count,
            coarse_similarity=coarse.mean_similarity,
            coarse_coverage=coarse.spatial_coverage,
        )
        if method == "hybrid" and coarse.failure_reason:
            return LocalPipelineResult(None, replace(base, total_seconds=time.perf_counter() - started))

        aliked: AlikedRuntime = self._aliked_factory(self._settings)
        lightglue: LightGlueRuntime = self._lightglue_factory(self._settings)
        model_hash = aliked.model_hash
        cached = cache.get(reference_hash, model_hash)
        if cached is None:
            reference_features, reference_metrics = aliked.extract(reference_image, reference_hash)
            cache.put(reference_hash, model_hash, reference_features, reference_metrics)
            cache_status = "MISS"
        else:
            reference_features, reference_metrics = cached
            cache_status = "HIT"
        frame_features, frame_metrics = aliked.extract(frame_image, frame_hash)
        diagnostic = replace(
            base,
            reason="local_features_ready",
            aliked_reference_keypoints=reference_features.keypoint_count,
            aliked_frame_keypoints=frame_features.keypoint_count,
            aliked_reference_seconds=(
                reference_metrics.preprocessing_seconds + reference_metrics.forward_seconds
                if cache_status == "MISS" else 0.0
            ),
            aliked_frame_seconds=frame_metrics.preprocessing_seconds + frame_metrics.forward_seconds,
            cache_status=cache_status,
            device=f"ALIKED:{aliked.device};LightGlue:not_loaded",
        )
        if (
            reference_features.keypoint_count < self._settings.matching_local_min_keypoints
            or frame_features.keypoint_count < self._settings.matching_local_min_keypoints
        ):
            return LocalPipelineResult(
                None, replace(diagnostic, reason="low_local_keypoints", total_seconds=time.perf_counter() - started)
            )
        local, lightglue_seconds = lightglue.match(reference_features, frame_features)
        diagnostic = replace(
            diagnostic,
            lightglue_match_count=local.match_count,
            lightglue_mean_score=local.mean_score,
            lightglue_seconds=lightglue_seconds,
            device=f"ALIKED:{aliked.device};LightGlue:{lightglue.device}",
        )
        if local.failure_reason or local.match_count < self._settings.matching_local_min_matches:
            return LocalPipelineResult(
                None, replace(diagnostic, reason="low_local_matches", total_seconds=time.perf_counter() - started)
            )
        matches = self._as_coarse(local, reference_features)
        homography_started = time.perf_counter()
        homography = self._homography.estimate(matches)
        homography_seconds = time.perf_counter() - homography_started
        diagnostic = replace(
            diagnostic,
            local_homography_inliers=homography.inlier_count,
            local_homography_inlier_ratio=homography.inlier_ratio,
            local_homography_rms=(
                homography.rms_reprojection_error
                if math.isfinite(homography.rms_reprojection_error) else None
            ),
            homography_seconds=homography_seconds,
        )
        if not homography.valid:
            return LocalPipelineResult(
                None,
                replace(
                    diagnostic,
                    reason=homography.failure_reason or "local_homography_invalid",
                    total_seconds=time.perf_counter() - started,
                ),
            )
        polygon = self._polygon.project_and_validate(
            homography.matrix,
            reference_width=reference_features.image_width,
            reference_height=reference_features.image_height,
            frame_width=frame_features.image_width,
            frame_height=frame_features.image_height,
        )
        if not polygon.valid:
            return LocalPipelineResult(None, replace(diagnostic, reason=polygon.failure_reason or "polygon_invalid", total_seconds=time.perf_counter() - started))
        bbox = self._bbox.validate(
            polygon, frame_width=frame_features.image_width, frame_height=frame_features.image_height
        )
        if bbox is None:
            return LocalPipelineResult(None, replace(diagnostic, reason="bbox_invalid", total_seconds=time.perf_counter() - started))
        quality = self._confidence.score(matches, homography, polygon)
        if quality is None:
            return LocalPipelineResult(None, replace(diagnostic, reason="confidence_below_threshold", total_seconds=time.perf_counter() - started))
        x1, y1, x2, y2 = bbox.clipped_box
        matched = MatchedReferenceObject(
            object_id=object_id,
            top_left_x=x1,
            top_left_y=y1,
            bottom_right_x=x2,
            bottom_right_y=y2,
            confidence=quality.confidence,
        )
        return LocalPipelineResult(
            matched, replace(diagnostic, accepted=True, reason="accepted", total_seconds=time.perf_counter() - started)
        )

    @staticmethod
    def _as_coarse(local, reference: LocalFeatureSet) -> CoarseMatchSet:
        import cv2
        import numpy as np
        points = np.asarray(local.reference_points_px, dtype=np.float32)
        coverage = 0.0
        if len(points) >= 3:
            hull = cv2.convexHull(points)
            coverage = min(
                1.0,
                max(0.0, float(cv2.contourArea(hull)) / (reference.image_width * reference.image_height)),
            )
        scores = np.asarray(local.scores, dtype=np.float32)
        return CoarseMatchSet(
            reference_points_px=points,
            frame_points_px=np.asarray(local.frame_points_px, dtype=np.float32),
            similarities=scores,
            correspondence_count=local.match_count,
            mean_similarity=float(scores.mean()),
            median_similarity=float(np.median(scores)),
            min_similarity=float(scores.min()),
            max_similarity=float(scores.max()),
            spatial_coverage=coverage,
        )
