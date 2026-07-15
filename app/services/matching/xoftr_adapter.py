from __future__ import annotations

from pathlib import Path


class LocalXoFTRAdapter:
    """Optional local-only XoFTR adapter.

    The configured artifact must be a TorchScript module accepting two
    grayscale tensors shaped ``[1, 1, H, W]`` and returning a dictionary with
    ``keypoints0``, ``keypoints1`` and optional ``confidence`` tensors. No
    network fallback is performed.
    """

    def __init__(self, model_path: Path, device: str) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"XoFTR modeli bulunamadı: {model_path}")
        import torch

        self._torch = torch
        self._device = device
        self._model = torch.jit.load(str(model_path), map_location=device).eval()

    def matches(self, reference, frame):
        import cv2
        import numpy as np

        def tensor(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return self._torch.from_numpy(gray)[None, None].float().to(self._device) / 255.0

        with self._torch.inference_mode():
            output = self._model(tensor(reference), tensor(frame))
        points0 = output["keypoints0"].detach().cpu().numpy().astype(np.float32)
        points1 = output["keypoints1"].detach().cpu().numpy().astype(np.float32)
        return points0, points1

    def bbox(self, reference, frame, min_inliers: int) -> dict[str, float] | None:
        import cv2
        import numpy as np

        points0, points1 = self.matches(reference, frame)
        if len(points0) < min_inliers:
            return None
        homography, mask = cv2.findHomography(
            points0,
            points1,
            cv2.USAC_MAGSAC,
            ransacReprojThreshold=3.0,
            maxIters=10000,
            confidence=0.999,
        )
        if homography is None or mask is None or int(mask.sum()) < min_inliers:
            return None
        height, width = reference.shape[:2]
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        frame_height, frame_width = frame.shape[:2]
        x1, y1 = projected.min(axis=0)
        x2, y2 = projected.max(axis=0)
        if x2 <= x1 or y2 <= y1:
            return None
        return {
            "top_left_x": float(max(0.0, x1)),
            "top_left_y": float(max(0.0, y1)),
            "bottom_right_x": float(min(frame_width, x2)),
            "bottom_right_y": float(min(frame_height, y2)),
            "confidence": float(mask.mean()),
        }
