"""
XoFTR sarmalayici: tembel yukleme + basit match()/bbox() arayuzu.
Termal<->RGB cross-modal eslesme icin dinov2_matcher.py tarafindan kullanilir.

teknofest_gorev3/xoftr_wrapper.py'den tasindi; yalnizca repo/agirlik yollari
bu projenin yerel artefaktlarina uyarlandi (external/xoftr + models/matching).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_REPO = str(_ROOT / "external" / "xoftr")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import cv2
import numpy as np
import torch
from yacs.config import CfgNode as CN

_DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
_MAX_EDGE = 840
_CKPT     = str(_ROOT / "models" / "matching" / "weights_xoftr_840.ckpt")

_matcher = None   # tembel singleton


def _lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): _lower_config(v) for k, v in yacs_cfg.items()}


def _get_matcher():
    global _matcher
    if _matcher is None:
        from xoftr_src.xoftr import XoFTR
        from xoftr_src.config.default import get_cfg_defaults
        config = _lower_config(get_cfg_defaults(inference=True))
        config["xoftr"]["match_coarse"]["thr"] = 0.25
        config["xoftr"]["fine"]["thr"] = 0.1
        config["xoftr"]["fine"]["denser"] = False
        m = XoFTR(config=config["xoftr"])
        ckpt = torch.load(_CKPT, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["state_dict"], strict=True)
        _matcher = m.eval().to(_DEVICE)
        print(f"[XoFTR] Yuklendi ({_DEVICE}, res={_MAX_EDGE})")
    return _matcher


def _prep(img_bgr, max_edge=_MAX_EDGE):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_edge / max(h, w))
    nw, nh = max(int(w * scale) // 8 * 8, 32), max(int(h * scale) // 8 * 8, 32)
    resized = cv2.resize(img_bgr, (nw, nh))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    t = torch.from_numpy(gray)[None][None].float().to(_DEVICE) / 255.0
    return t, (w / nw, h / nh)


@torch.no_grad()
def xoftr_match(ref_bgr: np.ndarray, frame_bgr: np.ndarray):
    """
    Ref <-> frame XoFTR eslesmeleri (koordinatlar orijinal olceklerde).
    Donen: (pts_frame Nx2, conf N, pts_ref Nx2)
    """
    m = _get_matcher()
    img0, (sx0, sy0) = _prep(ref_bgr)
    img1, (sx1, sy1) = _prep(frame_bgr)
    batch = {"image0": img0, "image1": img1}
    m(batch)
    mkpts0 = batch["mkpts0_f"].cpu().numpy()
    mkpts1 = batch["mkpts1_f"].cpu().numpy()
    mconf  = batch["mconf_f"].cpu().numpy()
    if len(mkpts0):
        mkpts0 = mkpts0 * np.array([sx0, sy0])
        mkpts1 = mkpts1 * np.array([sx1, sy1])
    return mkpts1, mconf, mkpts0


def _rotate_image(img: np.ndarray, angle_deg: int) -> np.ndarray:
    """Goruntuyu kayipsiz dondur."""
    if angle_deg == 0:
        return img
    if angle_deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle_deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle_deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = img.shape[:2]
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    return cv2.warpAffine(img, M, (nw, nh))


# Son basarili aci (drone yonelimi ani degismez -> sonraki karede once bu denenir)
_last_good_angle = [0]

_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]


def _best_homography(pts_r, pts_f, thr=3.0, n_seeds=5):
    """RANSAC'i 5 sabit cekilisle calistir, en cok inlier bulani don.
    Tek sansli/sanssiz cekilise bagimliligi kaldirir, sonuc tekrarlanabilir kalir."""
    best_mask, best_n = None, 0
    for seed in range(n_seeds):
        cv2.setRNGSeed(seed)
        H, mask = cv2.findHomography(pts_r.astype(np.float32),
                                     pts_f.astype(np.float32),
                                     cv2.USAC_MAGSAC,
                                     ransacReprojThreshold=thr,
                                     maxIters=10000, confidence=0.9999)
        if H is None or mask is None:
            continue
        n = int(mask.sum())
        if n > best_n:
            best_n, best_mask = n, mask
    return best_mask, best_n


def xoftr_bbox(ref_bgr: np.ndarray, frame_bgr: np.ndarray,
               min_inliers: int = 12, margin: int = 20,
               rotation_sweep: bool = True):
    """
    XoFTR + MAGSAC homografi -> inlier noktalarin 5-95 persentil bbox'i.
    rotation_sweep: referansi 45'er derece dondurerek dene (drone yonelimi
    farkli olabilir). Bbox frame koordinatinda uretildigi icin ref rotasyonu
    sonucu bozmaz, sadece eslesme kurulmasini saglar. Yetersiz eslesme -> None.
    """
    if rotation_sweep:
        first  = _last_good_angle[0]
        angles = [first] + [a for a in _ANGLES if a != first]
    else:
        angles = [0]

    best = None
    for ang in angles:
        ref_rot = _rotate_image(ref_bgr, ang)
        pts_f, conf, pts_r = xoftr_match(ref_rot, frame_bgr)
        if len(pts_f) < min_inliers:
            continue
        cv2.setRNGSeed(0)   # RANSAC tekrarlanabilir olsun
        H, mask = cv2.findHomography(pts_r.astype(np.float32),
                                     pts_f.astype(np.float32),
                                     cv2.USAC_MAGSAC,
                                     ransacReprojThreshold=3.0,
                                     maxIters=10000, confidence=0.9999)
        if H is None or mask is None or int(mask.sum()) < min_inliers:
            continue
        n_inl = int(mask.sum())
        if best is None or n_inl > best[0]:
            best = (n_inl, mask, pts_f, ang)
        if n_inl >= 25:      # yeterince guclu -> kalan acilari deneme
            break

    if best is None:
        return None
    n_inl, mask, pts_f, ang = best
    _last_good_angle[0] = ang

    inl = pts_f[mask.ravel() > 0]
    x1, x2 = np.percentile(inl[:, 0], [5, 95])
    y1, y2 = np.percentile(inl[:, 1], [5, 95])
    fh, fw = frame_bgr.shape[:2]
    return {
        "top_left_x":     float(max(0,  x1 - margin)),
        "top_left_y":     float(max(0,  y1 - margin)),
        "bottom_right_x": float(min(fw, x2 + margin)),
        "bottom_right_y": float(min(fh, y2 + margin)),
        "conf":           float(n_inl / max(len(pts_f), 1)),
        "inliers":        n_inl,
        "angle":          ang,
    }


def xoftr_bbox_2stage(ref_bgr: np.ndarray, frame_bgr: np.ndarray,
                      coarse_inliers: int = 6,
                      fine_inliers: int = 15,
                      margin: int = 20):
    """
    Kaba -> ince XoFTR tespiti.

    Asama 1 (KABA): Tum kare 840px'e kuculur, nesne kucuk kalir, eslesme azdir.
    Dusuk esikle (8) sadece "nesne muhtemelen surada" isareti alinir.

    Asama 2 (INCE): Isaret edilen bolge kirpilir ve KUCULTULMEDEN yeniden
    eslestirilir. Asil karar burada: gercek nesne tam cozunurlukte yuzlerce
    eslesme verir, sahte adaylar yuksek esigi (15) gecemez.

    Bulunamazsa None.
    """
    fh, fw = frame_bgr.shape[:2]

    # --- Asama 1: kaba isaretleme (tum kare) ---
    pts_f, _, pts_r = xoftr_match(ref_bgr, frame_bgr)
    if len(pts_f) < coarse_inliers:
        return None
    mask, n_coarse = _best_homography(pts_r, pts_f)
    if mask is None or n_coarse < coarse_inliers:
        return None
    inl = pts_f[mask.ravel() > 0]
    cx, cy = float(np.median(inl[:, 0])), float(np.median(inl[:, 1]))

    # --- Asama 2: kirpma dogrulamasi (tam cozunurluk) ---
    half = int(max(ref_bgr.shape[:2]) * 1.5)   # ref uzun kenarinin 1.5 kati yaricap
    x1c, y1c = max(0, int(cx - half)), max(0, int(cy - half))
    x2c, y2c = min(fw, int(cx + half)), min(fh, int(cy + half))
    crop = frame_bgr[y1c:y2c, x1c:x2c]
    if crop.shape[0] < 60 or crop.shape[1] < 60:
        return None

    pts_fc, _, pts_rc = xoftr_match(ref_bgr, crop)
    if len(pts_fc) < fine_inliers:
        return None
    mask_c, n_fine = _best_homography(pts_rc, pts_fc)
    if mask_c is None or n_fine < fine_inliers:
        return None

    inl_c = pts_fc[mask_c.ravel() > 0]
    x1, x2 = np.percentile(inl_c[:, 0], [5, 95])
    y1, y2 = np.percentile(inl_c[:, 1], [5, 95])
    return {
        "top_left_x":     float(max(0,  x1c + x1 - margin)),
        "top_left_y":     float(max(0,  y1c + y1 - margin)),
        "bottom_right_x": float(min(fw, x1c + x2 + margin)),
        "bottom_right_y": float(min(fh, y1c + y2 + margin)),
        "conf":           float(n_fine / max(len(pts_fc), 1)),
        "inliers":        n_fine,
        "coarse_inliers": n_coarse,
    }
