from __future__ import annotations

import logging

from app.core.config import Settings
from app.services.matching.descriptor_types import CoarseMatchSet, DenseDescriptorSet
from app.schemas import MatchedReferenceObject
from app.services.matching.bbox_validator import ProjectedBoundingBoxValidator
from app.services.matching.geometry import ConfidenceScorer, HomographyEstimator, ProjectedPolygonValidator

logger = logging.getLogger(__name__)


class CoarseMatcher:
    """Chunked cosine mutual-nearest-neighbour matcher for dense patch tokens."""

    def __init__(self, settings: Settings) -> None:
        self._min_similarity = settings.matching_coarse_min_similarity
        self._min_count = settings.matching_coarse_min_correspondences
        self._max_count = settings.matching_coarse_max_correspondences
        self._topk = settings.matching_coarse_topk_per_reference
        self._chunk_size = settings.matching_coarse_chunk_size
        self._dedup_radius = settings.matching_coarse_spatial_dedup_radius_px
        if not 0 <= self._min_similarity <= 1:
            raise ValueError("MATCHING_COARSE_MIN_SIMILARITY 0..1 araliginda olmali.")
        if self._min_count < 4 or self._max_count < self._min_count:
            raise ValueError("Coarse correspondence limitleri gecersiz.")
        if self._topk < 1 or self._chunk_size < 1 or self._dedup_radius < 0:
            raise ValueError("Coarse matcher config gecersiz.")

    def match(
        self,
        reference: DenseDescriptorSet,
        frame: DenseDescriptorSet,
    ) -> CoarseMatchSet:
        import numpy as np
        import torch

        logger.info("matching_coarse_started", extra={"event": "matching_coarse_started"})
        if reference.descriptor_dim != frame.descriptor_dim:
            logger.warning(
                "matching_coarse_descriptor_mismatch",
                extra={"event": "matching_coarse_descriptor_mismatch"},
            )
            return self._failure("descriptor_dimension_mismatch")
        if reference.shape[0] != reference.grid_width * reference.grid_height:
            return self._failure("reference_grid_mismatch")
        if frame.shape[0] != frame.grid_width * frame.grid_height:
            return self._failure("frame_grid_mismatch")

        ref = reference.descriptors.detach().float()
        frm = frame.descriptors.detach().float().to(ref.device)
        if ref.ndim != 2 or frm.ndim != 2 or ref.shape[1] != frm.shape[1]:
            return self._failure("descriptor_shape_invalid")
        if not bool(torch.isfinite(ref).all()) or not bool(torch.isfinite(frm).all()):
            return self._failure("descriptor_non_finite")
        if bool((torch.linalg.vector_norm(ref, dim=1) <= 1e-12).any()) or bool(
            (torch.linalg.vector_norm(frm, dim=1) <= 1e-12).any()
        ):
            return self._failure("descriptor_zero_norm")
        ref = torch.nn.functional.normalize(ref, dim=1)
        frm = torch.nn.functional.normalize(frm, dim=1)
        if not bool(torch.isfinite(ref).all()) or not bool(torch.isfinite(frm).all()):
            return self._failure("descriptor_zero_or_non_finite")

        ref_best_values = torch.full((len(ref),), -float("inf"), device=ref.device)
        ref_best_frame = torch.full((len(ref),), -1, dtype=torch.long, device=ref.device)
        frame_best_values = torch.full((len(frm),), -float("inf"), device=ref.device)
        frame_best_ref = torch.full((len(frm),), -1, dtype=torch.long, device=ref.device)
        for start in range(0, len(ref), self._chunk_size):
            stop = min(start + self._chunk_size, len(ref))
            similarity = ref[start:stop] @ frm.T
            if not bool(torch.isfinite(similarity).all()):
                return self._failure("similarity_non_finite")
            values, indices = similarity.max(dim=1)
            ref_best_values[start:stop] = values
            ref_best_frame[start:stop] = indices
            reverse_values, reverse_local = similarity.max(dim=0)
            better = reverse_values > frame_best_values
            frame_best_values[better] = reverse_values[better]
            frame_best_ref[better] = reverse_local[better] + start

        ref_indices = torch.arange(len(ref), device=ref.device)
        valid_frame = ref_best_frame >= 0
        mutual = torch.zeros_like(valid_frame)
        mutual[valid_frame] = frame_best_ref[ref_best_frame[valid_frame]] == ref_indices[valid_frame]
        keep = mutual & (ref_best_values >= self._min_similarity)
        selected_ref = ref_indices[keep].cpu().numpy()
        selected_frame = ref_best_frame[keep].cpu().numpy()
        similarities = ref_best_values[keep].cpu().numpy().astype(np.float32)
        if len(selected_ref) == 0:
            return self._insufficient("no_mutual_match")

        reference_points = self._indices_to_points(selected_ref, reference)
        frame_points = self._indices_to_points(selected_frame, frame)
        order = np.argsort(-similarities, kind="stable")
        selected = self._spatial_deduplicate(
            reference_points[order], frame_points[order], similarities[order]
        )
        reference_points, frame_points, similarities = selected
        if len(similarities) > self._max_count:
            reference_points = reference_points[: self._max_count]
            frame_points = frame_points[: self._max_count]
            similarities = similarities[: self._max_count]
        if len(similarities) < self._min_count:
            return self._insufficient("below_minimum_correspondences")

        coverage = min(
            self._coverage(reference_points, reference.image_width, reference.image_height),
            self._coverage(frame_points, frame.image_width, frame.image_height),
        )
        minimum_coverage = max(
            1e-5,
            (self._dedup_radius**2)
            / max(1.0, min(reference.image_width * reference.image_height,
                           frame.image_width * frame.image_height)),
        )
        if coverage < minimum_coverage:
            return self._insufficient("insufficient_spatial_coverage")

        return CoarseMatchSet(
            reference_points_px=reference_points,
            frame_points_px=frame_points,
            similarities=similarities,
            correspondence_count=len(similarities),
            mean_similarity=float(np.mean(similarities)),
            median_similarity=float(np.median(similarities)),
            min_similarity=float(np.min(similarities)),
            max_similarity=float(np.max(similarities)),
            spatial_coverage=float(coverage),
        )

    @staticmethod
    def _indices_to_points(indices, descriptor: DenseDescriptorSet):
        import numpy as np

        if descriptor.grid_width < 1 or descriptor.grid_height < 1:
            raise ValueError("Descriptor grid gecersiz.")
        if (
            descriptor.grid_width * descriptor.patch_size != descriptor.resized_width
            or descriptor.grid_height * descriptor.patch_size != descriptor.resized_height
            or descriptor.scale_x <= 0
            or descriptor.scale_y <= 0
        ):
            raise ValueError("Descriptor grid/scale metadata gecersiz.")
        rows = indices // descriptor.grid_width
        columns = indices % descriptor.grid_width
        resized_x = (columns.astype(np.float64) + 0.5) * descriptor.patch_size
        resized_y = (rows.astype(np.float64) + 0.5) * descriptor.patch_size
        points = np.column_stack((resized_x / descriptor.scale_x, resized_y / descriptor.scale_y))
        if not np.isfinite(points).all():
            raise ValueError("Patch koordinatlari sonlu degil.")
        if (
            (points[:, 0] < 0).any() or (points[:, 0] > descriptor.image_width).any()
            or (points[:, 1] < 0).any() or (points[:, 1] > descriptor.image_height).any()
        ):
            raise ValueError("Patch koordinatlari goruntu siniri disinda.")
        return points.astype(np.float32)

    def _spatial_deduplicate(self, reference_points, frame_points, similarities):
        import numpy as np

        if self._dedup_radius <= 0:
            return reference_points, frame_points, similarities
        kept: list[int] = []
        radius_squared = self._dedup_radius**2
        for index, (ref_point, frame_point) in enumerate(zip(reference_points, frame_points)):
            duplicate = any(
                float(np.sum((ref_point - reference_points[prior]) ** 2)) < radius_squared
                or float(np.sum((frame_point - frame_points[prior]) ** 2)) < radius_squared
                for prior in kept
            )
            if not duplicate:
                kept.append(index)
        return reference_points[kept], frame_points[kept], similarities[kept]

    @staticmethod
    def _coverage(points, width: int, height: int) -> float:
        import cv2
        import numpy as np

        if len(points) < 3 or width <= 0 or height <= 0:
            return 0.0
        hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
        return max(0.0, min(1.0, float(cv2.contourArea(hull)) / float(width * height)))

    @staticmethod
    def _failure(reason: str) -> CoarseMatchSet:
        import numpy as np

        return CoarseMatchSet(
            reference_points_px=np.empty((0, 2), dtype=np.float32),
            frame_points_px=np.empty((0, 2), dtype=np.float32),
            similarities=np.empty((0,), dtype=np.float32),
            correspondence_count=0,
            mean_similarity=0.0,
            median_similarity=0.0,
            min_similarity=0.0,
            max_similarity=0.0,
            spatial_coverage=0.0,
            failure_reason=reason,
        )

    def _insufficient(self, reason: str) -> CoarseMatchSet:
        logger.info(
            "matching_coarse_insufficient_correspondences",
            extra={"event": "matching_coarse_insufficient_correspondences", "reason": reason},
        )
        return self._failure(reason)


class CoarseMatchingPipeline:
    """Stateless RGB-RGB geometric matching pipeline for one reference."""

    def __init__(self, settings: Settings) -> None:
        self._coarse = CoarseMatcher(settings)
        self._homography = HomographyEstimator(settings)
        self._polygon = ProjectedPolygonValidator(settings)
        self._bbox = ProjectedBoundingBoxValidator(settings)
        self._confidence = ConfidenceScorer(settings)

    def match_reference(
        self,
        object_id: int,
        reference: DenseDescriptorSet,
        frame: DenseDescriptorSet,
    ) -> MatchedReferenceObject | None:
        matches = self._coarse.match(reference, frame)
        if matches.failure_reason:
            return None
        homography = self._homography.estimate(matches)
        if not homography.valid:
            return None
        polygon = self._polygon.project_and_validate(
            homography.matrix,
            reference_width=reference.image_width,
            reference_height=reference.image_height,
            frame_width=frame.image_width,
            frame_height=frame.image_height,
        )
        if not polygon.valid:
            return None
        bbox = self._bbox.validate(
            polygon, frame_width=frame.image_width, frame_height=frame.image_height
        )
        if bbox is None:
            return None
        quality = self._confidence.score(matches, homography, polygon)
        if quality is None:
            return None
        x_min, y_min, x_max, y_max = bbox.clipped_box
        result = MatchedReferenceObject(
            object_id=object_id,
            top_left_x=x_min,
            top_left_y=y_min,
            bottom_right_x=x_max,
            bottom_right_y=y_max,
            confidence=quality.confidence,
        )
        logger.info(
            "matching_reference_matched",
            extra={"event": "matching_reference_matched", "object_id": object_id,
                   "confidence": quality.confidence,
                   "inlier_ratio": quality.inlier_ratio},
        )
        return result
