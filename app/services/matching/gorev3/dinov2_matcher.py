"""
DINOv2 Dense Patch Matching + ALIKED/LightGlue Refinement

Global heatmap yerine dense patch eşleştirme kullanır:
1. Referans ve frame'den DINOv2 patch feature'ları çıkar
2. Mutual nearest-neighbor ile yoğun eşleşmeler bul
3. RANSAC homografi → kaba bbox
4. ALIKED+LightGlue ile hassas refinement

RGB↔termal / farklı irtifa / bakış açısı değişimine karşı dayanıklıdır.

teknofest_gorev3/dinov2_matcher.py'den tasindi. Algoritma birebir aynidir;
yalnizca model yukleme yerel artefaktlara uyarlandi:
  - DINOv2: external/dinov2-main (repo) + models/matching/dinov2_vitb14_pretrain.pth
  - ALIKED/LightGlue: external/LightGlue repo'su sys.path'e eklenir
    (agirliklar torch hub cache'inden okunur; bu makinede mevcut)
  - XoFTR: ayni paketteki xoftr_wrapper uzerinden (external/xoftr)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

_ROOT = Path(__file__).resolve().parents[4]
_LIGHTGLUE_REPO = str(_ROOT / "external" / "LightGlue")
if _LIGHTGLUE_REPO not in sys.path:
    sys.path.insert(0, _LIGHTGLUE_REPO)

from lightglue import LightGlue, ALIKED
from lightglue.utils import rbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DINOv2Matcher] Cihaz: {device}")

PATCH_SIZE    = 14
_MIN_INLIERS  = 6
_RANSAC_THR   = 12.0    # piksel — patch merkez aralığı göz önünde bulundurularak
_SIM_THR      = 0.20    # minimum kosinüs benzerliği filtresi

_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# ── Modeller ─────────────────────────────────────────────────────────────────
print("[DINOv2Matcher] DINOv2 ViT-B/14 yukleniyor (yerel)...")
_dino = torch.hub.load(
    str(_ROOT / "external" / "dinov2-main"),
    "dinov2_vitb14",
    source="local",
    pretrained=False,
    verbose=False,
)
_ckpt = torch.load(
    str(_ROOT / "models" / "matching" / "dinov2_vitb14_pretrain.pth"),
    map_location="cpu",
)
if isinstance(_ckpt, dict) and "state_dict" in _ckpt:
    _ckpt = _ckpt["state_dict"]
_dino.load_state_dict(
    {str(k).removeprefix("module."): v for k, v in _ckpt.items()}, strict=False
)
del _ckpt
_dino.eval().to(device)

print("[DINOv2Matcher] ALIKED + LightGlue yükleniyor...")
_extractor = ALIKED(max_num_keypoints=1024).eval().to(device)
_lg = LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1).eval().to(device)
print("[DINOv2Matcher] Hazır.")


# ── DINOv2 helper ─────────────────────────────────────────────────────────────

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _is_thermal(img_bgr: np.ndarray) -> bool:
    """Kanallar arası ortalama fark < 8 ise termal/gri görüntü."""
    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)
    return float(np.mean(np.abs(b - g)) + np.mean(np.abs(g - r))) < 8.0


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """BGR → CLAHE-enhanced grayscale (3 kanal)."""
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _to_lcn(img_bgr: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Local Contrast Normalization: her piksel komşularına göre normalize edilir.
    Termal ve RGB aynı fiziksel yapıyı kodladığından modalite farkını azaltır."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    local_mean = cv2.GaussianBlur(gray, (k, k), 0)
    diff       = gray - local_mean
    local_std  = cv2.GaussianBlur(diff ** 2, (k, k), 0) ** 0.5
    local_std  = np.maximum(local_std, 1e-3)
    lcn = diff / local_std
    lcn = cv2.normalize(lcn, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(lcn, cv2.COLOR_GRAY2BGR)


def _snap(h, w, long_edge):
    s  = long_edge / max(h, w)
    nh = max(PATCH_SIZE, round(h * s / PATCH_SIZE) * PATCH_SIZE)
    nw = max(PATCH_SIZE, round(w * s / PATCH_SIZE) * PATCH_SIZE)
    return nh, nw


@torch.no_grad()
def _dino_patches(img_bgr: np.ndarray, long_edge: int):
    """Döner: feat_map [D, pH, pW]"""
    h, w = img_bgr.shape[:2]
    nh, nw = _snap(h, w, long_edge)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    t   = _norm(torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0))
    out = _dino.forward_features(t.unsqueeze(0).to(device))
    pH, pW  = nh // PATCH_SIZE, nw // PATCH_SIZE
    patches = out["x_norm_patchtokens"].squeeze(0)           # [pH*pW, D]
    return patches.reshape(pH, pW, -1).permute(2, 0, 1)     # [D, pH, pW]


# ── Dense matching core ───────────────────────────────────────────────────────

@torch.no_grad()
def _dense_match(ref_bgr, frame_bgr, ref_scale, frame_scale):
    """
    DINOv2 dense matching: referans patch'leri ↔ frame patch'leri.
    Gradient maskesi → mutual nearest-neighbor → benzerlik filtresi.
    Döner: (kp_ref [N,2], kp_frame [N,2]) piksel koordinatlarında, ya da None, None.
    """
    rh, rw = ref_bgr.shape[:2]
    fh, fw = frame_bgr.shape[:2]

    fm_ref   = _dino_patches(ref_bgr,   ref_scale)    # [D, H1, W1]
    fm_frame = _dino_patches(frame_bgr, frame_scale)  # [D, H2, W2]

    D, H1, W1 = fm_ref.shape
    D, H2, W2 = fm_frame.shape
    N1, N2 = H1 * W1, H2 * W2

    r_flat = F.normalize(fm_ref.reshape(D, N1).T,   dim=-1)  # [N1, D]
    f_flat = F.normalize(fm_frame.reshape(D, N2).T, dim=-1)  # [N2, D]

    sim = r_flat @ f_flat.T   # [N1, N2]

    nn_r2f = sim.argmax(dim=1)   # [N1]  — ref→frame en yakın
    nn_f2r = sim.argmax(dim=0)   # [N2]  — frame→ref en yakın

    # Mutual nearest neighbor
    ref_idx = torch.arange(N1, device=device)
    mutual  = (nn_f2r[nn_r2f] == ref_idx)

    # Benzerlik eşiği
    best_sim = sim[ref_idx[mutual], nn_r2f[mutual]]
    mutual[mutual.clone()] = best_sim > _SIM_THR

    if mutual.sum() < 4:
        return None, None

    matched_r = ref_idx[mutual]
    matched_f = nn_r2f[mutual]

    # Patch merkezini orijinal piksel koordinatlarına çevir
    r_rows = (matched_r // W1).float()
    r_cols = (matched_r %  W1).float()
    f_rows = (matched_f // W2).float()
    f_cols = (matched_f %  W2).float()

    kp_ref = torch.stack([
        (r_cols + 0.5) / W1 * rw,
        (r_rows + 0.5) / H1 * rh,
    ], dim=1).cpu().numpy().astype(np.float32)

    kp_frame = torch.stack([
        (f_cols + 0.5) / W2 * fw,
        (f_rows + 0.5) / H2 * fh,
    ], dim=1).cpu().numpy().astype(np.float32)

    return kp_ref, kp_frame


# ── Local DINOv2 heatmap refinement ──────────────────────────────────────────

def _ref_global_feature(ref_bgr: np.ndarray) -> torch.Tensor:
    """Referans için normalize edilmiş global DINOv2 feature vektörü."""
    vecs = []
    for s in [224, 336]:
        fmap, cls = _dino_patches(ref_bgr, s), None
        # _dino_patches sadece fmap döndürüyor, cls için forward_features kullanalım
        h, w = ref_bgr.shape[:2]
        nh, nw = _snap(h, w, s)
        rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        t   = _norm(torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0))
        with torch.no_grad():
            out = _dino.forward_features(t.unsqueeze(0).to(device))
        cls  = out["x_norm_clstoken"].squeeze(0)
        fmap = out["x_norm_patchtokens"].squeeze(0).mean(0)   # [D]
        vec  = F.normalize((fmap + cls) * 0.5, dim=0)
        vecs.append(vec)
    return F.normalize(torch.stack(vecs).mean(0), dim=0)


@torch.no_grad()
def _local_dino_refine(frame_bgr: np.ndarray,
                        rough: dict, ref_feat: torch.Tensor,
                        sigma: float = 2.0,
                        max_scale: int = 448,
                        dilation: int = 0) -> dict | None:
    """
    Kaba bbox icinde DINOv2 heatmap ile bbox daralt.
    sigma: esik sikıligi (yuksek -> daha secici)
    max_scale: DINOv2 islem olcegi
    dilation: binary heatmap patch-seviyesi morfolojik dilasyon (>0 -> genis bbox)
    """
    fh, fw = frame_bgr.shape[:2]

    pad = 40
    cx1 = max(0,  int(rough["top_left_x"])     - pad)
    cy1 = max(0,  int(rough["top_left_y"])     - pad)
    cx2 = min(fw, int(rough["bottom_right_x"]) + pad)
    cy2 = min(fh, int(rough["bottom_right_y"]) + pad)

    crop = frame_bgr[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    if ch < PATCH_SIZE * 2 or cw < PATCH_SIZE * 2:
        return None

    crop_scale = max(PATCH_SIZE * 2, min(max_scale, max(ch, cw)))
    fmap = _dino_patches(crop, crop_scale)
    D, pH, pW = fmap.shape

    patches_n = F.normalize(fmap.reshape(D, -1).T, dim=-1)
    ref_n     = F.normalize(ref_feat, dim=0).unsqueeze(1)
    sim = (patches_n @ ref_n).reshape(pH, pW).cpu().numpy()

    mean, std = float(sim.mean()), float(sim.std())
    bin_thr = mean + sigma * std
    binary  = (sim >= bin_thr).astype(np.uint8)

    if dilation > 0:
        k = 2 * dilation + 1
        binary = cv2.dilate(binary, np.ones((k, k), np.uint8))

    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    py, px = np.unravel_index(sim.argmax(), sim.shape)
    lbl = int(labels[py, px])
    if lbl == 0:
        return None

    bx, by, bw, bh, _ = stats[lbl]
    px1 = int(bx        * cw / pW)
    py1 = int(by        * ch / pH)
    px2 = int((bx + bw) * cw / pW)
    py2 = int((by + bh) * ch / pH)

    if px2 - px1 < 8 or py2 - py1 < 8:
        return None

    return {
        "top_left_x":     float(cx1 + px1),
        "top_left_y":     float(cy1 + py1),
        "bottom_right_x": float(cx1 + px2),
        "bottom_right_y": float(cy1 + py2),
        "conf":           float(sim.max()),
    }


@torch.no_grad()
def _local_dino_multi(frame_bgr: np.ndarray,
                       ref_feat: torch.Tensor,
                       sigma: float = 2.0,
                       max_scale: int = 896,
                       pad: int = 10) -> list:
    """Full-frame DINOv2 heatmap -> tum esleşen bbox'ler (cok instance). Bos = bulunamadi."""
    fh, fw = frame_bgr.shape[:2]
    frame_area = float(fh * fw)
    crop_scale = max(PATCH_SIZE * 2, min(max_scale, max(fh, fw)))
    fmap = _dino_patches(frame_bgr, crop_scale)
    D, pH, pW = fmap.shape
    patches_n = F.normalize(fmap.reshape(D, -1).T, dim=-1)
    ref_n     = F.normalize(ref_feat, dim=0).unsqueeze(1)
    sim = (patches_n @ ref_n).reshape(pH, pW).cpu().numpy()
    mean, std = float(sim.mean()), float(sim.std())
    binary = (sim >= mean + sigma * std).astype(np.uint8)
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    results = []
    for lbl in range(1, num_lbl):
        bx, by, bw, bh, _ = stats[lbl]
        px1 = int(bx        * fw / pW)
        py1 = int(by        * fh / pH)
        px2 = int((bx + bw) * fw / pW)
        py2 = int((by + bh) * fh / pH)
        w, h = px2 - px1, py2 - py1
        if w < 8 or h < 8:
            continue
        box_area = float(w * h)
        # Ust sinir 0.35: buyuk nesneler (futbol sahasi gibi) karenin ucte birini
        # kaplayabilir; eski 0.12 siniri sahayi butun olarak yakalayan bileseni
        # silip kenar parcalarini birakiyordu (parcalanmis kutu gorunumu)
        if box_area < frame_area * 0.0005 or box_area > frame_area * 0.35:
            continue
        box_conf = float(sim[by:by+bh, bx:bx+bw].max())
        if box_conf < 0.35:   # dusuk benzerlik -> false positive
            continue
        results.append({
            "top_left_x":     float(max(0,  px1 - pad)),
            "top_left_y":     float(max(0,  py1 - pad)),
            "bottom_right_x": float(min(fw, px2 + pad)),
            "bottom_right_y": float(min(fh, py2 + pad)),
            "conf":           box_conf,
        })
    results.sort(key=lambda r: r["conf"], reverse=True)
    # NMS: cakisan kutulari birleştir
    kept = []
    for b in results:
        overlap = False
        for k in kept:
            ix1 = max(b["top_left_x"], k["top_left_x"])
            iy1 = max(b["top_left_y"], k["top_left_y"])
            ix2 = min(b["bottom_right_x"], k["bottom_right_x"])
            iy2 = min(b["bottom_right_y"], k["bottom_right_y"])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                b_area = (b["bottom_right_x"]-b["top_left_x"]) * (b["bottom_right_y"]-b["top_left_y"])
                k_area = (k["bottom_right_x"]-k["top_left_x"]) * (k["bottom_right_y"]-k["top_left_y"])
                if inter / (b_area + k_area - inter) > 0.30:
                    overlap = True
                    break
        if not overlap:
            kept.append(b)

    # Zincirleme birlestirme: degen/yakin kutular ayni nesnenin parcalaridir.
    # Buyuyen kutu baska kutulara da degerse onlar da katilir (zincir),
    # boylece ikiye-uce bolunmus nesneler tek kutuda toplanir.
    GAP = 90.0   # ~3 patch genisligi: heatmap ayni nesnede 2-3 patchlik adacik
                 # boslugu birakabiliyor (futbol sahasi 6 parcaya bolunuyordu)
    merged = []
    for b in kept:
        b = dict(b)
        changed = True
        while changed:
            changed = False
            for m in merged[:]:
                ayrik = (b["bottom_right_x"] + GAP < m["top_left_x"] or
                         m["bottom_right_x"] + GAP < b["top_left_x"] or
                         b["bottom_right_y"] + GAP < m["top_left_y"] or
                         m["bottom_right_y"] + GAP < b["top_left_y"])
                if not ayrik:
                    b["top_left_x"]     = min(b["top_left_x"],     m["top_left_x"])
                    b["top_left_y"]     = min(b["top_left_y"],     m["top_left_y"])
                    b["bottom_right_x"] = max(b["bottom_right_x"], m["bottom_right_x"])
                    b["bottom_right_y"] = max(b["bottom_right_y"], m["bottom_right_y"])
                    b["conf"]           = max(b["conf"], m["conf"])
                    merged.remove(m)
                    changed = True
        merged.append(b)
    merged.sort(key=lambda r: r["conf"], reverse=True)
    return merged


def _to_gradmag(img_bgr: np.ndarray) -> np.ndarray:
    """Sobel gradyan buyuklugu + CLAHE -> 3 kanal. Modaliteden bagimsiz kenar/yapi temsili."""
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag / (mag.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    mag = clahe.apply(mag)
    return cv2.merge([mag, mag, mag])


@torch.no_grad()
def _xmodal_sim(ref_repr: np.ndarray, frame_repr: np.ndarray,
                max_scale: int = 896) -> np.ndarray:
    """Tek temsil icin full-frame DINOv2 benzerlik haritasi."""
    ref_feat = _ref_global_feature(ref_repr)
    fh, fw = frame_repr.shape[:2]
    crop_scale = max(PATCH_SIZE * 2, min(max_scale, max(fh, fw)))
    fmap = _dino_patches(frame_repr, crop_scale)
    D, pH, pW = fmap.shape
    patches_n = F.normalize(fmap.reshape(D, -1).T, dim=-1)
    ref_n     = F.normalize(ref_feat, dim=0).unsqueeze(1)
    return (patches_n @ ref_n).reshape(pH, pW).cpu().numpy()


def _xmodal_multi(ref_bgr: np.ndarray, frame_bgr: np.ndarray,
                  sigma: float = 2.5, pad: int = 25) -> list:
    """
    Cross-modal (RGB<->termal) coklu tespit.
    Gray + gradyan temsillerinin z-normalize DINOv2 heatmap toplami ->
    hysteresis bolge buyutme -> baglantili bilesenler -> bbox listesi.
    Bos liste = bulunamadi.
    """
    fh, fw = frame_bgr.shape[:2]
    frame_area = float(fh * fw)

    sim_gray = _xmodal_sim(_to_gray(ref_bgr),    _to_gray(frame_bgr))
    sim_grad = _xmodal_sim(_to_gradmag(ref_bgr), _to_gradmag(frame_bgr))
    if sim_grad.shape != sim_gray.shape:
        sim_grad = cv2.resize(sim_grad, (sim_gray.shape[1], sim_gray.shape[0]))

    z = lambda s: (s - s.mean()) / (s.std() + 1e-6)
    ens = z(z(sim_gray) + z(sim_grad))
    pH, pW = ens.shape

    peak_z = float(ens.max())
    if peak_z < 1.4:          # zayif tepe -> nesne muhtemelen sahnede yok
        return []

    # Hysteresis: tohum = tepeye yakin patchler, bolge = dusuk esikle buyutulmus alan.
    # Esikler frame'in kendi tepe degerine orantili -> sahneden bagimsiz genel kural.
    thr_high  = peak_z - 0.30
    thr_low   = max(1.2, peak_z * 0.70)
    seed_mask = (ens >= thr_high).astype(np.uint8)
    low_mask  = (ens >= thr_low).astype(np.uint8)
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(low_mask, connectivity=8)

    results = []
    for lbl in range(1, num_lbl):
        comp = (labels == lbl)
        # Yalnizca icinde tohum bulunan bilesenler nesne sayilir
        if not (seed_mask[comp]).any():
            continue
        bx, by, bw, bh, _ = stats[lbl]
        px1 = int(bx        * fw / pW)
        py1 = int(by        * fh / pH)
        px2 = int((bx + bw) * fw / pW)
        py2 = int((by + bh) * fh / pH)
        w, h = px2 - px1, py2 - py1
        if w < 8 or h < 8:
            continue
        box_area = float(w * h)
        if box_area < frame_area * 0.0001 or box_area > frame_area * 0.50:
            continue
        results.append({
            "top_left_x":     float(max(0,  px1 - pad)),
            "top_left_y":     float(max(0,  py1 - pad)),
            "bottom_right_x": float(min(fw, px2 + pad)),
            "bottom_right_y": float(min(fh, py2 + pad)),
            "conf":           float(ens[comp].sum()),   # bilesen enerjisi (buyuk tutarli bolge kazanir)
            "peak":           float(ens[comp].max()),
        })
    results.sort(key=lambda r: r["conf"], reverse=True)

    # NMS (IoU + kapsama kontrolu)
    kept = []
    for b in results:
        overlap = False
        for k in kept:
            ix1 = max(b["top_left_x"], k["top_left_x"])
            iy1 = max(b["top_left_y"], k["top_left_y"])
            ix2 = min(b["bottom_right_x"], k["bottom_right_x"])
            iy2 = min(b["bottom_right_y"], k["bottom_right_y"])
            if ix2 > ix1 and iy2 > iy1:
                inter  = (ix2 - ix1) * (iy2 - iy1)
                b_area = (b["bottom_right_x"]-b["top_left_x"]) * (b["bottom_right_y"]-b["top_left_y"])
                k_area = (k["bottom_right_x"]-k["top_left_x"]) * (k["bottom_right_y"]-k["top_left_y"])
                iou     = inter / (b_area + k_area - inter)
                contain = inter / min(b_area, k_area)
                if iou > 0.30 or contain > 0.70:
                    overlap = True
                    break
        if not overlap:
            kept.append(b)
    # Enerjisi en iyinin yarisindan dusuk adaylari ele
    if kept:
        best_e = kept[0]["conf"]
        kept = [b for b in kept if b["conf"] >= best_e * 0.50]
    return kept[:5]


def _akaze_bbox(ref_gray: np.ndarray, frame_gray: np.ndarray,
                pad: int = 15) -> "dict | None":
    """AKAZE keypoint eslestirme -> homografi -> bbox. Termal<->termal icin."""
    akaze = cv2.AKAZE_create()
    kp_ref,   des_ref   = akaze.detectAndCompute(ref_gray,   None)
    kp_frame, des_frame = akaze.detectAndCompute(frame_gray, None)
    if des_ref is None or des_frame is None:
        return None
    if len(kp_ref) < 4 or len(kp_frame) < 4:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = bf.knnMatch(des_ref, des_frame, k=2)
    except cv2.error:
        return None
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 6:
        return None
    src_pts = np.float32([kp_ref[m.queryIdx].pt   for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or mask is None or int(mask.sum()) < 4:
        return None
    rh, rw = ref_gray.shape[:2]
    fh, fw = frame_gray.shape[:2]
    frame_area = float(fh * fw)
    corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    x1, y1 = float(proj[:, 0].min()), float(proj[:, 1].min())
    x2, y2 = float(proj[:, 0].max()), float(proj[:, 1].max())
    w, h = x2 - x1, y2 - y1
    if w * h > frame_area * 0.5 or w < 10 or h < 10:
        return None
    inlier_ratio = float(mask.sum()) / max(len(good), 1)
    return {
        "top_left_x":     max(0.0,       x1 - pad),
        "top_left_y":     max(0.0,       y1 - pad),
        "bottom_right_x": min(float(fw), x2 + pad),
        "bottom_right_y": min(float(fh), y2 + pad),
        "conf":           inlier_ratio,
    }


def _homography_bbox(H, ref_shape, frame_shape):
    """
    RANSAC'tan gelen homografi → frame'de axis-aligned bbox.
    Frame sınırlarını 1.5x aşan veya küçücük sonuçları reddeder.
    """
    rh, rw = ref_shape[:2]
    fh, fw = frame_shape[:2]
    corners = np.float32([[0,0],[rw,0],[rw,rh],[0,rh]]).reshape(-1,1,2)
    try:
        t = cv2.perspectiveTransform(corners, H)
    except cv2.error:
        return None
    rx, ry, rbw, rbh = cv2.boundingRect(t.astype(np.int32))

    # Dejenere homografi: frame sınırını 1.5x aşıyorsa veya çok küçükse reddet
    if (rbw < 5 or rbh < 5
            or rx < -fw * 1.5 or ry < -fh * 1.5
            or rx + rbw > fw * 2.5 or ry + rbh > fh * 2.5
            or rbw > fw * 3 or rbh > fh * 3):
        return None

    return {"top_left_x": float(rx), "top_left_y": float(ry),
            "bottom_right_x": float(rx + rbw), "bottom_right_y": float(ry + rbh)}


# ── LightGlue refinement ──────────────────────────────────────────────────────

def _img_tensor(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)


def _lightglue_refine(ref_bgr, frame_bgr, rough):
    """
    Kaba bbox → crop → ALIKED+LightGlue → hassas bbox.
    Başarısız olursa rough döner.
    """
    fh, fw = frame_bgr.shape[:2]
    pad = 60
    x1 = max(0,  int(rough["top_left_x"])     - pad)
    y1 = max(0,  int(rough["top_left_y"])     - pad)
    x2 = min(fw, int(rough["bottom_right_x"]) + pad)
    y2 = min(fh, int(rough["bottom_right_y"]) + pad)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.shape[0] < 20 or crop.shape[1] < 20:
        return rough

    with torch.no_grad():
        f0  = _extractor.extract(_img_tensor(ref_bgr))
        f1  = _extractor.extract(_img_tensor(crop))
        res = rbd(_lg({"image0": f0, "image1": f1}))

    m = res["matches"]
    m = m[res["scores"] > 0.15]
    if len(m) < _MIN_INLIERS:
        return rough

    kp0 = f0["keypoints"][0][m[:, 0]].cpu().numpy()
    kp1 = f1["keypoints"][0][m[:, 1]].cpu().numpy()

    H, mask = cv2.findHomography(kp0, kp1, cv2.RANSAC, 5.0)
    if H is None or mask is None or int(mask.sum()) < _MIN_INLIERS:
        return rough

    rh, rw = ref_bgr.shape[:2]
    corners = np.float32([[0,0],[rw,0],[rw,rh],[0,rh]]).reshape(-1,1,2)
    try:
        t = cv2.perspectiveTransform(corners, H)
    except cv2.error:
        return rough

    rx, ry, rbw, rbh = cv2.boundingRect(t.astype(np.int32))

    # Dejenere sonuç kontrolü
    crop_w, crop_h = x2 - x1, y2 - y1
    if rbw > crop_w * 2.5 or rbh > crop_h * 2.5 or rbw < 5 or rbh < 5:
        return rough

    return {
        "top_left_x":     float(x1 + rx),
        "top_left_y":     float(y1 + ry),
        "bottom_right_x": float(x1 + rx + rbw),
        "bottom_right_y": float(y1 + ry + rbh),
    }


def _lightglue_verify(ref_bgr, frame_bgr, box, margin: float = 0.2) -> int:
    """
    Aday kutuyu kirp, referansla LightGlue eslesmesi kur, homografi inlier sayisini don.
    Olcum (490 kutu): sahte kutular maks 23 inlier, gercekler 37-77 -> esik 25.
    """
    fh, fw = frame_bgr.shape[:2]
    x1, y1 = int(box["top_left_x"]), int(box["top_left_y"])
    x2, y2 = int(box["bottom_right_x"]), int(box["bottom_right_y"])
    mw, mh = int((x2 - x1) * margin), int((y2 - y1) * margin)
    crop = frame_bgr[max(0, y1 - mh):min(fh, y2 + mh),
                     max(0, x1 - mw):min(fw, x2 + mw)]
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        return 0
    with torch.no_grad():
        f0  = _extractor.extract(_img_tensor(ref_bgr))
        f1  = _extractor.extract(_img_tensor(crop))
        res = rbd(_lg({"image0": f0, "image1": f1}))
    m = res["matches"]
    if len(m) < 4:
        return 0
    kp0 = f0["keypoints"][0][m[:, 0]].cpu().numpy()
    kp1 = f1["keypoints"][0][m[:, 1]].cpu().numpy()
    cv2.setRNGSeed(0)
    H, mask = cv2.findHomography(kp0, kp1, cv2.USAC_MAGSAC, 5.0,
                                 maxIters=5000, confidence=0.999)
    return int(mask.sum()) if mask is not None else 0


@torch.no_grad()
def fallback_best_candidate(ref_feat: torch.Tensor, frame_bgr: np.ndarray,
                            min_conf: float = 0.30,
                            sigma: float = 2.0,
                            max_scale: int = 896) -> dict | None:
    """Dusuk esikli son-care aday: full-frame heatmap'in en iyi bileseni.

    Normal politika hicbir kutu birakmadiginda, YALNIZCA referansin aktif
    penceresi icinde cagrilmak uzere tasarlandi (pencerede nesnenin sahnede
    oldugu garanti -> kacirmanin maliyeti sahte kutudan yuksek).
    Olcum (ref1 penceresi, 45 kare): 0.30+ adaylarin tamami nesnenin ustune
    dustu; 0.30 alti adaylar yer yer sahneden kopuktu -> esik 0.30.
    """
    fh, fw = frame_bgr.shape[:2]
    frame_area = float(fh * fw)
    crop_scale = max(PATCH_SIZE * 2, min(max_scale, max(fh, fw)))
    fmap = _dino_patches(frame_bgr, crop_scale)
    D, pH, pW = fmap.shape
    patches_n = F.normalize(fmap.reshape(D, -1).T, dim=-1)
    ref_n = F.normalize(ref_feat, dim=0).unsqueeze(1)
    sim = (patches_n @ ref_n).reshape(pH, pW).cpu().numpy()
    mean, std = float(sim.mean()), float(sim.std())
    binary = (sim >= mean + sigma * std).astype(np.uint8)
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best = None
    for lbl in range(1, num_lbl):
        bx, by, bw, bh, _ = stats[lbl]
        px1 = int(bx * fw / pW)
        py1 = int(by * fh / pH)
        px2 = int((bx + bw) * fw / pW)
        py2 = int((by + bh) * fh / pH)
        w, h = px2 - px1, py2 - py1
        if w < 8 or h < 8:
            continue
        box_area = float(w * h)
        if box_area < frame_area * 0.0005 or box_area > frame_area * 0.35:
            continue
        conf = float(sim[by:by + bh, bx:bx + bw].max())
        if conf < min_conf:
            continue
        if best is None or conf > best["conf"]:
            best = {
                "top_left_x":     float(max(0, px1 - 10)),
                "top_left_y":     float(max(0, py1 - 10)),
                "bottom_right_x": float(min(fw, px2 + 10)),
                "bottom_right_y": float(min(fh, py2 + 10)),
                "conf":           conf,
            }
    return best


# ── Public API ─────────────────────────────────────────────────────────────────

class ObjectMatcher:
    """
    DINOv2 dense matching + ALIKED/LightGlue refinement.
    matcher.py ile aynı API — drop-in replacement.

    Pipeline:
      1. Referans + frame DINOv2 patch feature'ları (çoklu ölçek)
      2. Mutual nearest-neighbor → yoğun eşleşmeler
      3. RANSAC homografi → kaba bbox
      4. ALIKED+LightGlue crop refinement → hassas bbox
    """

    # (referans uzun kenar, frame uzun kenar) kombinasyonları
    _SCALE_PAIRS = [
        (224, 560),
        (224, 1120),
        (336, 560),
        (336, 1120),
    ]

    def __init__(self):
        self._ref_bgr        = None
        self._ref_feat       = None   # same-modal DINOv2 feature
        self._ref_bgr_xm     = None   # cross-modal versiyonu (gray/LCN)
        self._ref_feat_xm    = None   # cross-modal DINOv2 feature (Gray+LCN ensemble)
        self._is_thermal_ref = False

    _REF_MAX_EDGE = 448   # büyük referansları bu boyuta indir

    def set_reference(self, ref_bgr: np.ndarray):
        h, w = ref_bgr.shape[:2]
        # Takip durumu: yeni referansla sifirlanir
        self._track_center = None
        self._track_ttl    = 0
        self._track_area   = 0.0
        self._xm_track_center = None   # cross-modal izi (ayri tutulur)
        self._xm_track_ttl    = 0
        # LightGlue dogrulamasi icin orijinali (<=1024px) sakla — 448'lik kucultme
        # keypoint detayini oldurur, olcum 1024 ile yapildi
        if max(h, w) > 1024:
            s = 1024 / max(h, w)
            self._ref_verify_img = cv2.resize(ref_bgr, (int(w * s), int(h * s)),
                                              interpolation=cv2.INTER_AREA)
        else:
            self._ref_verify_img = ref_bgr.copy()
        if max(h, w) > self._REF_MAX_EDGE:
            scale  = self._REF_MAX_EDGE / max(h, w)
            new_w  = max(1, int(w * scale))
            new_h  = max(1, int(h * scale))
            ref_bgr = cv2.resize(ref_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"[ObjectMatcher] Referans kucultuldu: ({h},{w}) -> {ref_bgr.shape}")
        self._ref_bgr_orig  = ref_bgr.copy()  # cross-modal ensemble + XoFTR icin islenmemis hali

        # Cross-modal özellikler her iki yön için önceden hesapla (Gray+LCN ensemble)
        ref_gray = _to_gray(ref_bgr)
        ref_lcn  = _to_lcn(ref_bgr)
        feat_gray = _ref_global_feature(ref_gray)
        feat_lcn  = _ref_global_feature(ref_lcn)
        xm_feat = F.normalize(feat_gray + feat_lcn, dim=0)

        if _is_thermal(ref_bgr):
            self._is_thermal_ref = True
            self._ref_bgr     = ref_lcn    # termal ref → LCN
            self._ref_feat    = xm_feat    # termal için ensemble zaten cross-modal
            self._ref_bgr_xm  = ref_gray
            self._ref_feat_xm = xm_feat
            print("[ObjectMatcher] Termal referans tespit edildi -> Gray+LCN ensemble uygulandı.")
        else:
            self._is_thermal_ref = False
            self._ref_bgr     = ref_bgr    # RGB ref olduğu gibi
            self._ref_feat    = _ref_global_feature(ref_bgr)
            self._ref_bgr_xm  = ref_lcn   # cross-modal durumda LCN versiyonu
            self._ref_feat_xm = xm_feat
        print(f"[ObjectMatcher] Referans yuklendi: {self._ref_bgr.shape}")
    def _find_cross_modal(self, frame_bgr):
        """Ensemble (gray+gradyan DINOv2) + XoFTR uzlasma -> cross-modal coklu bbox.

        Termal video olcumleriyle dogrulanmis yol:
        - Enerji tabani 60: nesne ekrandayken medyan 67-103, yokken 23-47
        - XoFTR kutusu ensemble ile ortusuyorsa parcalari birlestirir (yapistirici)
        """
        ens_results = _xmodal_multi(self._ref_bgr_orig, frame_bgr, sigma=2.5)
        if not ens_results:
            ens_results = _xmodal_multi(self._ref_bgr_orig, frame_bgr, sigma=2.0)

        MIN_ENERGY = 60.0
        ens_results = [b for b in ens_results if b.get("conf", 0.0) >= MIN_ENERGY]
        if not ens_results:
            return self._xm_gate([])   # iz yaslansin

        # Uzlasma: XoFTR kutusu ensemble kutulariyla merkez-ortusuyorsa birlestir
        try:
            from app.services.matching.gorev3.xoftr_wrapper import xoftr_bbox
            xf = xoftr_bbox(self._ref_bgr_orig, frame_bgr, min_inliers=8)
        except Exception as e:
            print(f"[XMODAL] XoFTR kullanilamadi ({e})")
            xf = None

        if xf is not None:
            def _center_in(a, b):
                cx = (a["top_left_x"] + a["bottom_right_x"]) / 2
                cy = (a["top_left_y"] + a["bottom_right_y"]) / 2
                return (b["top_left_x"] <= cx <= b["bottom_right_x"] and
                        b["top_left_y"] <= cy <= b["bottom_right_y"])

            agree = [b for b in ens_results
                     if _center_in(xf, b) or _center_in(b, xf)]
            rest  = [b for b in ens_results if b not in agree]
            if agree:
                def _union(box, b):
                    box["top_left_x"]     = min(box["top_left_x"],     b["top_left_x"])
                    box["top_left_y"]     = min(box["top_left_y"],     b["top_left_y"])
                    box["bottom_right_x"] = max(box["bottom_right_x"], b["bottom_right_x"])
                    box["bottom_right_y"] = max(box["bottom_right_y"], b["bottom_right_y"])

                # Cikti kutusu: SADECE ensemble parcalarinin birlesimi
                # (XoFTR uzantilari guvenilmez -> kutuya girmez, yalnizca yapistirici)
                merged = {
                    "top_left_x":     min(b["top_left_x"]     for b in agree),
                    "top_left_y":     min(b["top_left_y"]     for b in agree),
                    "bottom_right_x": max(b["bottom_right_x"] for b in agree),
                    "bottom_right_y": max(b["bottom_right_y"] for b in agree),
                    "conf":           max(b["conf"] for b in agree),
                    "peak":           max(b.get("peak", 0.0) for b in agree),
                }
                region = dict(merged)
                _union(region, xf)
                changed = True
                while changed:
                    changed = False
                    for b in rest[:]:
                        ix1 = max(region["top_left_x"], b["top_left_x"])
                        iy1 = max(region["top_left_y"], b["top_left_y"])
                        ix2 = min(region["bottom_right_x"], b["bottom_right_x"])
                        iy2 = min(region["bottom_right_y"], b["bottom_right_y"])
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue
                        inter  = (ix2 - ix1) * (iy2 - iy1)
                        b_area = ((b["bottom_right_x"] - b["top_left_x"]) *
                                  (b["bottom_right_y"] - b["top_left_y"]))
                        if inter / b_area >= 0.30:
                            _union(merged, b)
                            _union(region, b)
                            merged["conf"] = max(merged["conf"], b["conf"])
                            rest.remove(b)
                            changed = True
                print(f"[XMODAL] Uzlasma: XoFTR ({xf['inliers']} inlier) dogruladi, "
                      f"{len(ens_results) - len(rest)} ensemble kutusu birlestirildi")
                return self._xm_gate([merged] + rest)
            print("[XMODAL] XoFTR kutusu ensemble ile ortusmuyor -> ensemble kullaniliyor")

        print(f"[XMODAL] Ensemble esleme: {len(ens_results)} kutu, "
              f"en iyi conf={ens_results[0]['conf']:.2f}")
        return self._xm_gate(ens_results)

    def _xm_gate(self, boxes):
        """Cross-modal siki-baslangic / takip kapisi.

        Olcum (105 kare x 3 ref): nesne ekrandayken peak medyani 1.88-2.37,
        yokken 1.57-1.82. Enerji genis-vasat bolgelerde de yukselebildigi icin
        tek basina yetmiyor; sivrilik (peak) sarti sahte baslangici keser.
        - Iz baslatma: peak >= 1.85 olan kutu(lar) -> raporla, izi tazele
        - Devam: iz aktifken son konumun yakinindaki en iyi kutu (enerji 60'i
          zaten gecmis) tek basina raporlanir
        - Ikisi de yoksa bos
        """
        _TTL, _R = 20, 250.0
        strong = [b for b in boxes if b.get("peak", 0.0) >= 1.85]
        if strong:
            best = max(strong, key=lambda r: r.get("conf", 0.0))
            self._xm_track_center = ((best["top_left_x"] + best["bottom_right_x"]) / 2,
                                     (best["top_left_y"] + best["bottom_right_y"]) / 2)
            self._xm_track_ttl = _TTL
            return strong
        if self._xm_track_ttl > 0 and self._xm_track_center is not None:
            self._xm_track_ttl -= 1
            tx, ty = self._xm_track_center
            adaylar = []
            for b in boxes:
                cx = (b["top_left_x"] + b["bottom_right_x"]) / 2
                cy = (b["top_left_y"] + b["bottom_right_y"]) / 2
                if abs(cx - tx) <= _R and abs(cy - ty) <= _R:
                    adaylar.append((b, cx, cy))
            if adaylar:
                b, cx, cy = max(adaylar, key=lambda t: t[0].get("conf", 0.0))
                self._xm_track_center = (cx, cy)
                return [b]
        return []

    def find(self, frame_bgr: np.ndarray) -> list:
        """
        Frame icinde referans nesneyi ara (cok instance destekler).
        Bulunursa  -> [{"top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y", "conf"}, ...]
        Bulunmazsa -> []
        """
        if self._ref_bgr is None:
            raise RuntimeError("Önce set_reference() çağırın.")

        # Modalite tespiti → cross-modal ise her ikisini de ortak temsile çevir
        frame_is_thermal = _is_thermal(frame_bgr)
        cross_modal = (self._is_thermal_ref != frame_is_thermal)

        if cross_modal:
            # Termal↔RGB: gray+gradyan DINOv2 ensemble + XoFTR uzlasma
            return self._find_cross_modal(frame_bgr)

        if self._is_thermal_ref:
            # Termal+termal → grayscale normalize + geometrik dogrulamali yol
            frame_bgr   = _to_gray(frame_bgr)
            _ref_bgr    = self._ref_bgr_xm
            _ref_feat   = self._ref_feat_xm
            # 1) AKAZE (hizli; futbol sahasi gibi cizgili yapilarda guclu)
            akaze_res = _akaze_bbox(_ref_bgr, frame_bgr)
            if akaze_res is not None:
                print(f"[AKAZE] Termal esleme basarili: conf={akaze_res['conf']:.2f}")
                return [akaze_res]
            # 2) XoFTR iki asamali (kaba isaret -> kirpma dogrulamasi)
            try:
                from app.services.matching.gorev3.xoftr_wrapper import xoftr_bbox_2stage
                xf_res = xoftr_bbox_2stage(self._ref_bgr_orig, frame_bgr)
            except Exception as e:
                print(f"[XoFTR] Kullanilamadi ({e})")
                xf_res = None
            if xf_res is not None:
                print(f"[XoFTR] Termal esleme basarili: kaba={xf_res.get('coarse_inliers','?')} "
                      f"ince={xf_res['inliers']}")
                return [xf_res]
            # 3) Geometrik dogrulama yoksa nesne yok say (DINOv2 yedegi termal icin
            # kaldirildi: her karede sahte kutu uretiyordu)
            print("[XoFTR] Esleme basarisiz -> bulunamadi.")
            return []
        else:
            # RGB↔RGB → orijinal temsil
            _ref_bgr    = self._ref_bgr
            _ref_feat   = self._ref_feat

        best_H       = None
        best_inliers = 0

        for ref_scale, frame_scale in self._SCALE_PAIRS:
            kp_ref, kp_frame = _dense_match(
                _ref_bgr, frame_bgr, ref_scale, frame_scale
            )
            if kp_ref is None or len(kp_ref) < 4:
                continue

            H, mask = cv2.findHomography(kp_ref, kp_frame, cv2.RANSAC, _RANSAC_THR)
            if H is None or mask is None:
                continue

            inliers = int(mask.sum())
            if inliers > best_inliers:
                best_inliers = inliers
                best_H       = H

        fh2, fw2 = frame_bgr.shape[:2]
        frame_area = float(fh2 * fw2)
        PAD = 10

        # DINOv2 homografi başarısız veya çok düşük inlier → renk yoluna geç
        dino_failed = (best_H is None or best_inliers < _MIN_INLIERS)
        rough = None if dino_failed else _homography_bbox(
            best_H, _ref_bgr.shape, frame_bgr.shape
        )

        # Kaba bbox frame'in %40'ından büyükse veya inlier az → DINOv2 güvenilmez
        if rough is not None:
            rough_area = ((rough["bottom_right_x"] - rough["top_left_x"]) *
                          (rough["bottom_right_y"] - rough["top_left_y"]))
            dino_low_conf = rough_area > frame_area * 0.40 or best_inliers < 8
        else:
            dino_low_conf = True

        # Rough bbox yoksa tüm frame'i kullan
        search = rough or {
            "top_left_x": 0.0, "top_left_y": 0.0,
            "bottom_right_x": float(fw2), "bottom_right_y": float(fh2),
        }

        if dino_low_conf:
            # Aşama 1: Tam frame'de heatmap ile kaba konum bul
            coarse = _local_dino_refine(
                frame_bgr, search, _ref_feat, sigma=2.0, max_scale=448
            )
            if coarse is None:
                coarse = _local_dino_refine(
                    frame_bgr, search, _ref_feat, sigma=2.0, max_scale=896
                )
            if coarse is None:
                coarse = _local_dino_refine(
                    frame_bgr, search, _ref_feat, sigma=1.5, max_scale=896
                )
            if coarse is None:
                return []

            coarse_w = coarse["bottom_right_x"] - coarse["top_left_x"]
            coarse_h = coarse["bottom_right_y"] - coarse["top_left_y"]

            # Çok küçük → sigma=1.0 ile genişlet (thermal cross-modal durumunda olabilir)
            if coarse_w < 60 or coarse_h < 60:
                loose = _local_dino_refine(
                    frame_bgr, search, _ref_feat, sigma=1.0, max_scale=896
                )
                if loose is not None:
                    lw = loose["bottom_right_x"] - loose["top_left_x"]
                    lh = loose["bottom_right_y"] - loose["top_left_y"]
                    if lw * lh < frame_area * 0.10:
                        coarse, coarse_w, coarse_h = loose, lw, lh

            coarse_area = coarse_w * coarse_h

            # Çok büyük → belirsiz eşleşme
            if coarse_area > frame_area * 0.10:
                return []

            # Çok küçük → gürültü (tek bir patch)
            if coarse_w < 15 or coarse_h < 15:
                return []

            # Köşe false positive filtresi (tek kenar geçerli olabilir)
            border = 3
            at_corner = (
                (coarse["top_left_x"] <= border and coarse["top_left_y"] <= border) or
                (coarse["bottom_right_x"] >= fw2-border and coarse["bottom_right_y"] >= fh2-border) or
                (coarse["top_left_x"] <= border and coarse["bottom_right_y"] >= fh2-border) or
                (coarse["bottom_right_x"] >= fw2-border and coarse["top_left_y"] <= border)
            )
            if at_corner:
                return []

            # Aşama 2: Coarse bölgede yüksek çözünürlüklü (896px) fine refine
            fine = _local_dino_refine(
                frame_bgr, coarse, _ref_feat, sigma=1.5, max_scale=896
            )
            if fine is None:
                base = coarse
            else:
                fine_area = ((fine["bottom_right_x"] - fine["top_left_x"]) *
                             (fine["bottom_right_y"]  - fine["top_left_y"]))
                # Fine çok küçüldüyse (<%20 coarse) veya çok büyüdüyse → coarse daha sağlam
                if fine_area < coarse_area * 0.20 or fine_area > coarse_area * 2.0:
                    base = coarse
                else:
                    base = fine
        else:
            # Normal pipeline: local heatmap → LightGlue (küçüldüyse)
            refined = _local_dino_refine(
                frame_bgr, search, _ref_feat, sigma=2.0, max_scale=448
            )
            if refined is None:
                return []
            refined_area = ((refined["bottom_right_x"] - refined["top_left_x"]) *
                            (refined["bottom_right_y"]  - refined["top_left_y"]))
            lg      = _lightglue_refine(_ref_bgr, frame_bgr, refined)
            lg_area = ((lg["bottom_right_x"] - lg["top_left_x"]) *
                       (lg["bottom_right_y"]  - lg["top_left_y"]))
            base = lg if lg_area < refined_area else refined

        # Birincil tespit bulundu -> tum instance'lari ara (conf < 0.35 zaten filtrelenir)
        multi = _local_dino_multi(frame_bgr, _ref_feat, sigma=2.5, max_scale=896, pad=PAD)
        if not multi:
            multi = _local_dino_multi(frame_bgr, _ref_feat, sigma=2.0, max_scale=896, pad=PAD)

        # RGB<->RGB dogrulama politikasi (video olcumlerinden turetildi):
        #   conf >= 0.45          -> dogrudan kabul (sahtelerin tavani 0.484'tu, cogu <0.435)
        #   0.35 <= conf < 0.45   -> LightGlue kirpma dogrulamasi: inlier >= 25 sart
        #                            (sahteler maks 23, gercekler 37-77 vermisti)
        # Termal referanslarda dokunma (o yol icin olculmedi).
        if multi and not self._is_thermal_ref:
            kept = []
            dogrulanan = 0
            for b in sorted(multi, key=lambda r: r.get("conf", 0.0), reverse=True):
                c = b.get("conf", 0.0)
                if c >= 0.45:
                    kept.append(b)
                elif dogrulanan < 3:   # kare basi en fazla 3 dogrulama (hiz sinirlamasi)
                    dogrulanan += 1
                    inl = _lightglue_verify(self._ref_verify_img, frame_bgr, b)
                    if getattr(self, "_debug_policy", False):
                        print(f"    [POLITIKA] aday ({int(b['top_left_x'])},{int(b['top_left_y'])})-"
                              f"({int(b['bottom_right_x'])},{int(b['bottom_right_y'])}) "
                              f"conf={c:.3f} lg_inlier={inl}")
                    if inl >= 25:
                        kept.append(b)

            # Takip devamliligi (detect-then-track):
            # Siki kanitla onaylanan nesne, sonraki karelerde bakis acisi/ortu
            # degisince kaniti zayiflasa da son onayli konumun yakinindaysa surdurulur.
            # Gurultu iz BASLATAMAZ (siki kapi), iz en fazla _TRACK_TTL kare tasinir.
            _TRACK_TTL = 20
            _TRACK_R   = 250.0
            if kept:
                best = max(kept, key=lambda r: r.get("conf", 0.0))
                self._track_center = ((best["top_left_x"] + best["bottom_right_x"]) / 2,
                                      (best["top_left_y"] + best["bottom_right_y"]) / 2)
                self._track_area = ((best["bottom_right_x"] - best["top_left_x"]) *
                                    (best["bottom_right_y"] - best["top_left_y"]))
                self._track_ttl = _TRACK_TTL
            elif self._track_ttl > 0 and self._track_center is not None:
                self._track_ttl -= 1
                tx, ty = self._track_center
                adaylar = []
                for b in multi:
                    cx = (b["top_left_x"] + b["bottom_right_x"]) / 2
                    cy = (b["top_left_y"] + b["bottom_right_y"]) / 2
                    if abs(cx - tx) <= _TRACK_R and abs(cy - ty) <= _TRACK_R:
                        adaylar.append((b, cx, cy))
                if adaylar:
                    b, cx, cy = max(adaylar, key=lambda t: t[0].get("conf", 0.0))
                    b_area = ((b["bottom_right_x"] - b["top_left_x"]) *
                              (b["bottom_right_y"] - b["top_left_y"]))
                    # Devam kutusu son onayli kutuya gore asiri kuculemez:
                    # ufak kirinti kutulari IoU tutturamaz, mAP'te zarar (kare bos gecilir)
                    boyut_ok = b_area >= 0.30 * getattr(self, "_track_area", 0.0)
                    # Minimal saglamlik: bir miktar geometrik iz olsun (bos doku degil)
                    if boyut_ok and _lightglue_verify(self._ref_verify_img, frame_bgr, b) >= 5:
                        kept = [b]
                        self._track_center = (cx, cy)
            multi = kept
        elif not self._is_thermal_ref and self._track_ttl > 0:
            # Hic aday yokken de iz yaslanir
            self._track_ttl -= 1
        return multi  # bos ise [] doner -> nesne bu sahnede yok
