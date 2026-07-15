# Görev 3 model artefaktları

Ağırlıklar repoya alınmaz ve çalışma zamanında indirilmez. Beklenen yerleşim:

```text
models/matching/
├── dinov2_vitb14.pth          # zorunlu DINOv2 state dict
├── aliked.ts                  # isteğe bağlı TorchScript
├── lightglue.ts               # isteğe bağlı TorchScript
└── xoftr.ts                   # çapraz-modal yol için isteğe bağlı TorchScript
```

DINOv2 kaynak deposu ayrıca `MATCHING_DINOV2_REPO_PATH` ile gösterilir. Ağırlık
yolları sırasıyla `DINOV2_MODEL_PATH`, `MATCHING_ALIKED_WEIGHTS_PATH`,
`LIGHTGLUE_MODEL_PATH` ve `XOFTR_MODEL_PATH` değişkenlerinden okunur.

ALIKED wrapper girdisi `[1,3,H,W]` RGB tensor ve çıktısı feature sözlüğüdür.
LightGlue iki feature sözlüğü alır. XoFTR iki `[1,1,H,W]` grayscale tensor alıp
`keypoints0`, `keypoints1` ve isteğe bağlı `confidence` döndürür. Gerçek artefakt
uyumluluğu, kullanılan wrapper export sözleşmesiyle ayrıca doğrulanmalıdır.
