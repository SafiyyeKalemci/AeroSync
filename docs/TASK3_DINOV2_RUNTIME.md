# Görev 3 — Yerel DINOv2 Descriptor Runtime

Bu belge Görev 3 entegrasyonunun ikinci aşamasını açıklar. Bu aşama yalnızca RGB referans ve frame görüntülerinden dense patch descriptor üretir. Eşleme, homography, bounding box ve `MatchedReferenceObject` üretimi bilinçli olarak devre dışıdır; servis doğrulanmış bir eşleme bulunmadığı için her zaman `[]` döndürür.

## Yerel ve çevrimdışı artifact politikası

DINOv2 kaynak kodu ve ağırlıkları önceden güvenilir bir kaynaktan temin edilmiş yerel artifact olmalıdır. Uygulama bunları indirmez ve uzak `torch.hub` kullanmaz. Kaynak klasörde `hubconf.py`, ağırlık yolunda `.pt`, `.pth` veya `.ckpt` dosyası beklenir.

Örnek yerleşim:

```text
models/matching/
├── dinov2/                 # Yerel DINOv2 repository
│   ├── hubconf.py
│   └── dinov2/
└── dinov2_vitb14.pth       # Yerel state-dict
```

Gerekli ayarlar:

```dotenv
MATCHING_DINOV2_ENABLED=true
MATCHING_DINOV2_REPO_PATH=models/matching/dinov2
MATCHING_DINOV2_WEIGHTS_PATH=models/matching/dinov2_vitb14.pth
MATCHING_DINOV2_MODEL_NAME=dinov2_vitb14
MATCHING_DINOV2_DEVICE=auto
MATCHING_DINOV2_MAX_LONG_EDGE=1120
MATCHING_DINOV2_PATCH_SIZE=14
MATCHING_DINOV2_DESCRIPTOR_DTYPE=float32
MATCHING_DINOV2_ALLOW_CPU_FALLBACK=true
MATCHING_DINOV2_NORMALIZE_DESCRIPTORS=true
MATCHING_DINOV2_MAX_CACHED_REFERENCES=32
MATCHING_DINOV2_TIMEOUT_SECONDS=5
MATCHING_DINOV2_CACHE_DEVICE=cpu
MATCHING_DINOV2_MAX_CACHE_MB=512
```

PyTorch checkpoint dosyaları güvenilmeyen kod/veri taşıyabilir. Yalnız güvenilir artifact kullanın ve SHA-256 değerini dağıtım manifestinizle karşılaştırın. Runtime desteklenen PyTorch sürümlerinde `weights_only=True` kullanır; eski sürüm fallback'i açık bir warning log üretir.

## Runtime yaşam döngüsü

Runtime process içinde aynı model ayarlarını kullanan session'lar arasında paylaşılır. Model uygulama importunda değil, ilk descriptor talebinde thread-safe lazy olarak yüklenir. `torch.hub.load` yalnız `source="local"` ve `pretrained=False` ile çağrılır; state-dict ayrıca yüklenip missing/unexpected key kontrolünden geçirilir. Model `eval()` ve inference `inference_mode()` ile çalışır. Tek inference lock, aynı model üzerinde eşzamanlı forward'ları güvenli biçimde seri hale getirir.

`auto` cihaz seçimi CUDA'yı tercih eder. CUDA yoksa ve fallback açıksa CPU'ya geçilir ve warning loglanır. CPU üzerinde güvenli olmayan `float16` descriptor isteği `float32`ye düşürülür. CUDA OOM, timeout veya artifact hatasında sahte descriptor üretilmez ve servis çökmeden `[]` döner.

## RGB preprocessing ve dense descriptor

Ortak preprocessing akışı şöyledir:

```text
OpenCV BGR image
→ RGB dönüşümü
→ aspect ratio korunarak uzun kenarı küçültme
→ genişlik/yüksekliği patch size katına aşağı snap
→ [0,1] dönüşümü ve ImageNet normalization
→ [1, 3, H, W] batch tensor
→ x_norm_patchtokens
→ [grid_width × grid_height, descriptor_dim]
```

Çok küçük görüntüler reddedilir. CLS token kullanılmaz. Token sayısı patch grid ile doğrulanır; descriptor dimension pozitif olmalı ve NaN/Inf bulunmamalıdır. İsteğe bağlı L2 normalization uygulanır. Typed sonuç orijinal/yeniden boyutlandırılmış ölçüleri, grid ölçülerini, ölçekleri, cihazı, dtype'ı ve kaynak hash'ini taşır.

## Referans descriptor cache

Her session kendi `ReferenceStore` nesnesine sahiptir. Cache ana geçerlilik koşulu aynı görüntü SHA-256 ve aynı model artifact SHA-256 değeridir. Cache metadata'sı shape, device, dtype, oluşturma/son erişim zamanı ve byte boyutunu içerir.

- Aynı hash/model için tekrar forward yapılmaz.
- Görüntü veya model hash'i değişince descriptor geçersizleşir.
- Incremental güncellemede değişmeyen referans korunur; yalnız yeni/değişen referans hazırlanır.
- Referans silme, session reset ve TTL eviction ilgili cache'i temizler.
- Referans başına async lock aynı descriptor'ın eşzamanlı iki kez üretilmesini önler.
- Session başına referans sayısı ve byte limiti LRU ile uygulanır; ayrıca tüm session'lar için toplam byte limiti uygulanır.
- Güvenli varsayılan cache cihazı CPU'dur. Frame descriptor cache'lenmez ve yalnız çağrı süresince yaşar.

Bir frame'de birden çok aktif referans olsa bile frame dosyası bir kez okunur, bir kez decode/preprocess edilir ve tam bir DINOv2 forward yalnız bir kez yapılır. Bu descriptor gelecek eşleme aşamasında ortak kullanılacaktır.

## Timeout ve cancellation sınırı

Descriptor işi `asyncio.wait_for(asyncio.to_thread(...))` ile sınırlandırılır. Python worker thread'i timeout anında fiziksel olarak durmayabilir. Bu nedenle store generation token ve görüntü hash'i commit sırasında yeniden doğrulanır; reset edilmiş/değişmiş session'a geç tamamlanan sonuç yazılamaz. Uzun forward için warning üretilir. `torch.cuda.empty_cache()` frame başına çağrılmaz.

## Preflight

Online RGB hazırlığında yerel DINOv2 repository ve weight zorunludur; eksikleri kritik `FAIL` üretir. Preflight repo klasörünü, `hubconf.py` import edilebilirliğini (model inference yapmadan), bağımlılıkları, weight uzantısı/boyutu/kısa SHA-256 değerini ve cihaz politikasını kontrol eder. ALIKED ve LightGlue eksikleri `WARNING`, XoFTR eksikliği RGB-only aşamada `WARNING` olarak kalır.

## Sonraki aşama

Bir sonraki planlı aşama dense descriptor'lar üzerinde mutual nearest-neighbor eşleme ve doğrulanmış MAGSAC homography'dir. Bu dosyanın kapsadığı aşamada bu algoritmaların hiçbiri ve bounding box üretimi bulunmaz.
