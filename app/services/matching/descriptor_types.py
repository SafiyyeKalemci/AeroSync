from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    tensor: object
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    grid_width: int
    grid_height: int
    scale_x: float
    scale_y: float
    patch_size: int


@dataclass(frozen=True, slots=True)
class DenseDescriptorSet:
    descriptors: object
    grid_width: int
    grid_height: int
    descriptor_dim: int
    image_width: int
    image_height: int
    resized_width: int
    resized_height: int
    patch_size: int
    scale_x: float
    scale_y: float
    device: str
    dtype: str
    source_hash: str

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.descriptors.shape)

    @property
    def nbytes(self) -> int:
        return int(self.descriptors.numel() * self.descriptors.element_size())


@dataclass(frozen=True, slots=True)
class DescriptorMetrics:
    preprocessing_seconds: float
    forward_seconds: float
    descriptor_count: int
    descriptor_dimension: int
    descriptor_bytes: int


@dataclass(frozen=True, slots=True)
class CoarseMatchSet:
    reference_points_px: object
    frame_points_px: object
    similarities: object
    correspondence_count: int
    mean_similarity: float
    median_similarity: float
    min_similarity: float
    max_similarity: float
    spatial_coverage: float
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class HomographyResult:
    matrix: object | None
    inlier_mask: object | None
    inlier_count: int
    inlier_ratio: float
    rms_reprojection_error: float
    valid: bool
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectedPolygon:
    points: object | None
    raw_area: float
    visible_area: float
    visible_ratio: float
    valid: bool
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ValidatedBoundingBox:
    raw_box: tuple[float, float, float, float]
    clipped_box: tuple[float, float, float, float]
    area: float


@dataclass(frozen=True, slots=True)
class MatchingQuality:
    confidence: float
    inlier_ratio: float
    similarity_score: float
    reprojection_score: float
    visibility_score: float
    coverage_score: float
