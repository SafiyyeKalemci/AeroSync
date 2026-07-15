from __future__ import annotations

from pathlib import Path


class LocalAlikedLightGlueAdapter:
    """ALIKED + LightGlue refinement using local TorchScript artifacts only.

    The ALIKED artifact accepts an RGB tensor ``[1, 3, H, W]`` and returns a
    feature dictionary. The LightGlue artifact accepts two feature dictionaries
    and returns either matched ``keypoints0``/``keypoints1`` or
    ``keypoints0``/``keypoints1`` plus a ``matches`` index tensor. Wrapping the
    upstream models once into this stable contract avoids runtime downloads and
    shields the service from upstream API changes.
    """

    def __init__(self, aliked_path: Path, lightglue_path: Path, device: str) -> None:
        if not aliked_path.is_file():
            raise FileNotFoundError(f"ALIKED modeli bulunamadı: {aliked_path}")
        if not lightglue_path.is_file():
            raise FileNotFoundError(f"LightGlue modeli bulunamadı: {lightglue_path}")
        import torch

        self._torch = torch
        self._device = device
        self._extractor = torch.jit.load(str(aliked_path), map_location=device).eval()
        self._matcher = torch.jit.load(str(lightglue_path), map_location=device).eval()

    def _tensor(self, image):
        import cv2
        import numpy as np

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return (
            self._torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)[None]
            .float()
            .to(self._device)
            .div(255.0)
        )

    def refine(self, reference, frame, rough_box: dict[str, float], min_inliers: int):
        import cv2
        import numpy as np

        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, int(rough_box["top_left_x"]) - 32)
        y1 = max(0, int(rough_box["top_left_y"]) - 32)
        x2 = min(frame_width, int(rough_box["bottom_right_x"]) + 32)
        y2 = min(frame_height, int(rough_box["bottom_right_y"]) + 32)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        with self._torch.inference_mode():
            features0 = self._extractor(self._tensor(reference))
            features1 = self._extractor(self._tensor(crop))
            output = self._matcher(features0, features1)
        points0 = output["keypoints0"].detach().cpu().numpy().astype(np.float32)
        points1 = output["keypoints1"].detach().cpu().numpy().astype(np.float32)
        if "matches" in output:
            matches = output["matches"].detach().cpu().numpy()
            points0 = points0[matches[:, 0]]
            points1 = points1[matches[:, 1]]
        if len(points0) < min_inliers:
            return None
        homography, mask = cv2.findHomography(
            points0,
            points1,
            cv2.USAC_MAGSAC,
            ransacReprojThreshold=5.0,
            maxIters=5000,
            confidence=0.999,
        )
        inliers = int(mask.sum()) if mask is not None else 0
        if homography is None or inliers < min_inliers:
            return None
        ref_height, ref_width = reference.shape[:2]
        corners = np.float32(
            [[0, 0], [ref_width, 0], [ref_width, ref_height], [0, ref_height]]
        ).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        bx1, by1 = projected.min(axis=0)
        bx2, by2 = projected.max(axis=0)
        if bx2 <= bx1 or by2 <= by1:
            return None
        return {
            "top_left_x": float(max(0.0, x1 + bx1)),
            "top_left_y": float(max(0.0, y1 + by1)),
            "bottom_right_x": float(min(frame_width, x1 + bx2)),
            "bottom_right_y": float(min(frame_height, y1 + by2)),
            "confidence": float(inliers / max(len(points0), 1)),
        }
