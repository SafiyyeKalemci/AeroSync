from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LocalFeatureSet:
    keypoints: object
    descriptors: object
    scores: object
    image_width: int
    image_height: int
    device: str
    source_hash: str

    @property
    def keypoint_count(self) -> int:
        return int(self.keypoints.shape[0])

    @property
    def descriptor_dimension(self) -> int:
        return int(self.descriptors.shape[1])


@dataclass(frozen=True, slots=True)
class LocalFeatureMetrics:
    preprocessing_seconds: float
    forward_seconds: float
    keypoint_count: int
    descriptor_dimension: int
    image_shape: tuple[int, int]
    device: str
    runtime: str = "ALIKED TorchScript"


@dataclass(frozen=True, slots=True)
class LocalMatchSet:
    reference_points_px: object
    frame_points_px: object
    scores: object
    match_count: int
    mean_score: float
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class LocalRefinementDiagnostics:
    method: str
    accepted: bool
    reason: str
    coarse_correspondence_count: int = 0
    coarse_similarity: float = 0.0
    coarse_coverage: float = 0.0
    aliked_reference_keypoints: int = 0
    aliked_frame_keypoints: int = 0
    lightglue_match_count: int = 0
    lightglue_mean_score: float = 0.0
    local_homography_inliers: int = 0
    local_homography_inlier_ratio: float = 0.0
    local_homography_rms: float | None = None
    aliked_reference_seconds: float = 0.0
    aliked_frame_seconds: float = 0.0
    lightglue_seconds: float = 0.0
    homography_seconds: float = 0.0
    total_seconds: float = 0.0
    cache_status: str = "SKIPPED"
    device: str = "unavailable"


class LocalArtifactUnavailable(RuntimeError):
    pass


class LocalFeatureError(RuntimeError):
    pass
