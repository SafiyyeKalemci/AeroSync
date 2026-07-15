from __future__ import annotations

import logging

from app.core.config import Settings
from app.services.matching.model_runtime import (
    MatchingModelConfigurationError,
    MatchingModelRuntimeRegistry,
)

logger = logging.getLogger(__name__)


class DinoV2DenseMatcher:
    """Local-only DINOv2 dense matcher with geometric verification.

    This keeps the strongest production-relevant portion of the Safiyye
    implementation: dense patch similarity, mutual-nearest-neighbor filtering,
    RANSAC homography and optional cross-modal XoFTR verification. Model loading
    is lazy and never calls a remote hub.
    """

    patch_size = 14

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._runtime = MatchingModelRuntimeRegistry.get(settings)
        self._torch = self._runtime.torch
        self._device = self._runtime.device
        self._model = self._runtime.dinov2
        self._lightglue = self._runtime.lightglue
        self._xoftr = self._runtime.xoftr
        self._references: dict[int, tuple[object, object]] = {}

    @staticmethod
    def decode(content: bytes):
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Görüntü decode edilemedi.")
        return image

    def _snap_size(self, height: int, width: int) -> tuple[int, int]:
        edge = min(self._settings.matching_max_image_edge, max(height, width))
        scale = edge / max(height, width)
        new_height = max(self.patch_size, round(height * scale / self.patch_size) * self.patch_size)
        new_width = max(self.patch_size, round(width * scale / self.patch_size) * self.patch_size)
        return new_height, new_width

    def _features(self, image):
        import cv2
        import numpy as np

        height, width = image.shape[:2]
        new_height, new_width = self._snap_size(height, width)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        tensor = self._torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)
        mean = self._torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = self._torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self._device)
        with self._runtime.inference_lock, self._torch.inference_mode():
            output = self._model.forward_features(tensor)
        patches = output["x_norm_patchtokens"].squeeze(0)
        patch_height = new_height // self.patch_size
        patch_width = new_width // self.patch_size
        patches = self._torch.nn.functional.normalize(patches, dim=-1)
        return patches, patch_height, patch_width, height, width

    def set_reference(self, object_id: int, image) -> None:
        self._references[object_id] = (image, self._features(image))

    def remove_reference(self, object_id: int) -> None:
        self._references.pop(object_id, None)

    def prepare_frame(self, frame):
        return self._features(frame)

    def clear(self) -> None:
        self._references.clear()

    def _dense_correspondences(self, reference_features, frame_features):
        import numpy as np

        ref, ref_h, ref_w, original_ref_h, original_ref_w = reference_features
        frm, frm_h, frm_w, original_frame_h, original_frame_w = frame_features
        similarity = ref @ frm.T
        ref_to_frame = similarity.argmax(dim=1)
        frame_to_ref = similarity.argmax(dim=0)
        ref_indices = self._torch.arange(len(ref), device=self._device)
        mutual = frame_to_ref[ref_to_frame] == ref_indices
        mutual &= similarity[ref_indices, ref_to_frame] >= self._settings.matching_similarity_threshold
        selected_ref = ref_indices[mutual]
        selected_frame = ref_to_frame[mutual]
        if len(selected_ref) < 4:
            return None

        def coordinates(indices, grid_width, grid_height, image_width, image_height):
            rows = (indices // grid_width).float()
            cols = (indices % grid_width).float()
            return self._torch.stack(
                [
                    (cols + 0.5) / grid_width * image_width,
                    (rows + 0.5) / grid_height * image_height,
                ],
                dim=1,
            ).cpu().numpy().astype(np.float32)

        return (
            coordinates(selected_ref, ref_w, ref_h, original_ref_w, original_ref_h),
            coordinates(selected_frame, frm_w, frm_h, original_frame_w, original_frame_h),
        )

    def _homography_bbox(self, reference, frame, reference_features, frame_features):
        import cv2
        import numpy as np

        pairs = self._dense_correspondences(reference_features, frame_features)
        if pairs is None:
            return None
        points0, points1 = pairs
        homography, mask = cv2.findHomography(
            points0,
            points1,
            cv2.USAC_MAGSAC,
            ransacReprojThreshold=12.0,
            maxIters=10000,
            confidence=0.999,
        )
        inliers = int(mask.sum()) if mask is not None else 0
        if homography is None or inliers < self._settings.matching_min_inliers:
            return None
        ref_height, ref_width = reference.shape[:2]
        corners = np.float32(
            [[0, 0], [ref_width, 0], [ref_width, ref_height], [0, ref_height]]
        ).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        x1, y1 = projected.min(axis=0)
        x2, y2 = projected.max(axis=0)
        frame_height, frame_width = frame.shape[:2]
        if x2 <= x1 or y2 <= y1:
            return None
        if x2 < 0 or y2 < 0 or x1 >= frame_width or y1 >= frame_height:
            return None
        return {
            "top_left_x": float(max(0.0, x1)),
            "top_left_y": float(max(0.0, y1)),
            "bottom_right_x": float(min(frame_width, x2)),
            "bottom_right_y": float(min(frame_height, y2)),
            "confidence": float(inliers / max(len(points0), 1)),
        }

    def match_reference(
        self,
        object_id: int,
        frame,
        *,
        frame_features=None,
        cross_modal: bool = False,
    ) -> dict[str, float | int] | None:
        stored = self._references.get(object_id)
        if stored is None:
            return None
        reference, reference_features = stored
        with self._runtime.inference_lock:
            if cross_modal:
                if self._xoftr is None:
                    return None
                box = self._xoftr.bbox(reference, frame, self._settings.matching_min_inliers)
            else:
                if frame_features is None:
                    frame_features = self._features(frame)
                box = self._homography_bbox(reference, frame, reference_features, frame_features)
                if box is not None and self._lightglue is not None:
                    refined = self._lightglue.refine(
                        reference,
                        frame,
                        box,
                        self._settings.matching_min_inliers,
                    )
                    if refined is not None:
                        box = refined
        return {"object_id": object_id, **box} if box is not None else None

    def match(self, frame) -> list[dict[str, float | int]]:
        frame_features = self.prepare_frame(frame)
        results: list[dict[str, float | int]] = []
        for object_id in self._references:
            result = self.match_reference(
                object_id,
                frame,
                frame_features=frame_features,
                cross_modal=False,
            )
            if result is not None:
                results.append(result)
        return results
