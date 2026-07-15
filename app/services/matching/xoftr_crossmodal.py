"""Checkpoint tabanlı XoFTR çapraz-modal (termal↔RGB) eşleyici.

Resmî LoFTR-türevi XoFTR modeli TorchScript'e çevrilemediği için model,
yerel repo kaynak kodu (``MATCHING_XOFTR_REPO_PATH``) ve Lightning
checkpoint dosyası (``MATCHING_XOFTR_CKPT_PATH``) üzerinden yüklenir.
Hiçbir koşulda ağdan model indirilmez.

Repo paketi ``xoftr_src`` adıyla kopyalanmıştır; resmî arayüzün ``src``
paketiyle isim çakışması bilinçli olarak bu şekilde önlenir.
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass

from app.core.config import Settings

logger = logging.getLogger(__name__)

_ROTATION_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)
_STRONG_INLIER_EARLY_STOP = 25
_BBOX_MARGIN_PX = 20
_PERCENTILE_LOW = 5
_PERCENTILE_HIGH = 95
# Eski pipeline'daki kalite kapısı inlier SAYISIDIR; ham inlier/eşleşme oranı
# yoğun eşleyicilerde tipik olarak 0.0x mertebesinde kalır ve genel
# MATCHING_MIN_CONFIDENCE eşiğiyle (0.35) uyumsuzdur. Bu yüzden bildirilen
# güven, inlier sayısının normalize edilmiş halidir: min_inliers (12) ~0.4 verir.
_INLIER_CONFIDENCE_SCALE = 30.0


class XoFTRUnavailable(RuntimeError):
    """Model artefaktları eksik veya yükleme başarısız."""


@dataclass(frozen=True)
class _RuntimeKey:
    repo_path: str
    ckpt_path: str
    device: str
    max_edge: int


class XoFTRCrossModalMatcher:
    """Lazy yüklenen, kilitle korunan XoFTR çıkarım sarmalayıcısı."""

    def __init__(self, settings: Settings) -> None:
        settings.validate_matching_xoftr()
        if not settings.matching_xoftr_enabled:
            raise XoFTRUnavailable("MATCHING_XOFTR_ENABLED kapalı.")
        assert settings.matching_xoftr_repo_path is not None
        assert settings.matching_xoftr_ckpt_path is not None
        self._repo_path = settings.matching_xoftr_repo_path
        self._ckpt_path = settings.matching_xoftr_ckpt_path
        self._device_preference = settings.matching_xoftr_device
        self._max_edge = settings.matching_xoftr_max_edge
        self._min_inliers = settings.matching_xoftr_min_inliers
        self._rotation_sweep = settings.matching_xoftr_rotation_sweep
        self._model = None
        self._torch = None
        self._device: str | None = None
        self._load_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self._last_good_angle = 0

    def _resolve_device(self) -> str:
        import torch

        if self._device_preference == "cuda":
            if not torch.cuda.is_available():
                raise XoFTRUnavailable("MATCHING_XOFTR_DEVICE=cuda ancak CUDA yok.")
            return "cuda"
        if self._device_preference == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import torch

            repo_text = str(self._repo_path)
            inserted = repo_text not in sys.path
            if inserted:
                sys.path.insert(0, repo_text)
            try:
                from xoftr_src.config.default import get_cfg_defaults
                from xoftr_src.xoftr import XoFTR
            except Exception as exc:
                raise XoFTRUnavailable(
                    "XoFTR yerel repo modülleri import edilemedi."
                ) from exc
            finally:
                if inserted:
                    sys.path.remove(repo_text)

            def lower(node):
                try:
                    from yacs.config import CfgNode
                except Exception as exc:  # pragma: no cover - yacs kurulu değilse
                    raise XoFTRUnavailable("yacs bağımlılığı eksik.") from exc
                if not isinstance(node, CfgNode):
                    return node
                return {key.lower(): lower(value) for key, value in node.items()}

            config = lower(get_cfg_defaults(inference=True))
            config["xoftr"]["match_coarse"]["thr"] = 0.25
            config["xoftr"]["fine"]["thr"] = 0.1
            config["xoftr"]["fine"]["denser"] = False
            try:
                model = XoFTR(config=config["xoftr"])
                checkpoint = torch.load(
                    str(self._ckpt_path), map_location="cpu", weights_only=False
                )
                model.load_state_dict(checkpoint["state_dict"], strict=True)
            except Exception as exc:
                raise XoFTRUnavailable("XoFTR checkpoint yüklenemedi.") from exc
            device = self._resolve_device()
            self._model = model.eval().to(device)
            self._torch = torch
            self._device = device
            logger.info(
                "matching_xoftr_loaded",
                extra={
                    "event": "matching_xoftr_loaded",
                    "device": device,
                    "max_edge": self._max_edge,
                },
            )

    def _prepare(self, image_bgr):
        import cv2

        height, width = image_bgr.shape[:2]
        scale = min(1.0, self._max_edge / max(height, width))
        new_width = max(int(width * scale) // 8 * 8, 32)
        new_height = max(int(height * scale) // 8 * 8, 32)
        resized = cv2.resize(image_bgr, (new_width, new_height))
        gray = (
            cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            if resized.ndim == 3
            else resized
        )
        tensor = (
            self._torch.from_numpy(gray)[None][None].float().to(self._device) / 255.0
        )
        return tensor, (width / new_width, height / new_height)

    def match(self, reference_bgr, frame_bgr):
        """Referans↔frame eşleşmeleri; koordinatlar orijinal ölçekte döner."""
        import numpy as np

        self._ensure_loaded()
        image0, (sx0, sy0) = self._prepare(reference_bgr)
        image1, (sx1, sy1) = self._prepare(frame_bgr)
        batch = {"image0": image0, "image1": image1}
        with self._torch.inference_mode():
            self._model(batch)
        points_reference = batch["mkpts0_f"].cpu().numpy()
        points_frame = batch["mkpts1_f"].cpu().numpy()
        confidence = batch["mconf_f"].cpu().numpy()
        if len(points_reference):
            points_reference = points_reference * np.array([sx0, sy0])
            points_frame = points_frame * np.array([sx1, sy1])
        return (
            points_frame.astype(np.float32),
            confidence.astype(np.float32),
            points_reference.astype(np.float32),
        )

    @staticmethod
    def _rotate(image, angle: int):
        import cv2

        if angle == 0:
            return image
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_width = int(height * sin + width * cos)
        new_height = int(height * cos + width * sin)
        matrix[0, 2] += new_width / 2 - center[0]
        matrix[1, 2] += new_height / 2 - center[1]
        return cv2.warpAffine(image, matrix, (new_width, new_height))

    def bbox(self, reference_bgr, frame_bgr) -> dict[str, float] | None:
        """XoFTR + MAGSAC homografi → inlier persentil bbox.

        Referans rotasyon taraması drone yönelim farkını tolere eder; bbox
        frame koordinatında üretildiği için referans rotasyonu sonucu bozmaz.
        Yetersiz eşleşmede None döner.
        """
        import cv2
        import numpy as np

        self._ensure_loaded()
        if self._rotation_sweep:
            first = self._last_good_angle
            angles = [first] + [a for a in _ROTATION_ANGLES if a != first]
        else:
            angles = [0]

        best: tuple[int, object, object, int] | None = None
        with self.inference_lock:
            for angle in angles:
                rotated = self._rotate(reference_bgr, angle)
                points_frame, _, points_reference = self.match(rotated, frame_bgr)
                if len(points_frame) < self._min_inliers:
                    continue
                cv2.setRNGSeed(0)
                homography, mask = cv2.findHomography(
                    points_reference.astype(np.float32),
                    points_frame.astype(np.float32),
                    cv2.USAC_MAGSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=10000,
                    confidence=0.9999,
                )
                if homography is None or mask is None:
                    continue
                inlier_count = int(mask.sum())
                if inlier_count < self._min_inliers:
                    continue
                if best is None or inlier_count > best[0]:
                    best = (inlier_count, mask, points_frame, angle)
                if inlier_count >= _STRONG_INLIER_EARLY_STOP:
                    break

        if best is None:
            return None
        inlier_count, mask, points_frame, angle = best
        self._last_good_angle = angle
        inliers = points_frame[mask.ravel() > 0]
        x1, x2 = np.percentile(inliers[:, 0], [_PERCENTILE_LOW, _PERCENTILE_HIGH])
        y1, y2 = np.percentile(inliers[:, 1], [_PERCENTILE_LOW, _PERCENTILE_HIGH])
        frame_height, frame_width = frame_bgr.shape[:2]
        return {
            "top_left_x": float(max(0.0, x1 - _BBOX_MARGIN_PX)),
            "top_left_y": float(max(0.0, y1 - _BBOX_MARGIN_PX)),
            "bottom_right_x": float(min(frame_width, x2 + _BBOX_MARGIN_PX)),
            "bottom_right_y": float(min(frame_height, y2 + _BBOX_MARGIN_PX)),
            "confidence": float(min(1.0, inlier_count / _INLIER_CONFIDENCE_SCALE)),
            "inlier_ratio": float(inlier_count / max(len(points_frame), 1)),
            "inliers": inlier_count,
            "angle": angle,
        }


class XoFTRRuntimeRegistry:
    """Ayar kombinasyonu başına tek matcher; DINOv2 kayıt deseniyle uyumlu."""

    _instances: dict[_RuntimeKey, XoFTRCrossModalMatcher] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, settings: Settings) -> XoFTRCrossModalMatcher:
        settings.validate_matching_xoftr()
        if not settings.matching_xoftr_enabled:
            raise XoFTRUnavailable("MATCHING_XOFTR_ENABLED kapalı.")
        assert settings.matching_xoftr_repo_path is not None
        assert settings.matching_xoftr_ckpt_path is not None
        key = _RuntimeKey(
            repo_path=str(settings.matching_xoftr_repo_path),
            ckpt_path=str(settings.matching_xoftr_ckpt_path),
            device=settings.matching_xoftr_device,
            max_edge=settings.matching_xoftr_max_edge,
        )
        with cls._lock:
            instance = cls._instances.get(key)
            if instance is None:
                instance = XoFTRCrossModalMatcher(settings)
                cls._instances[key] = instance
            return instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instances.clear()
