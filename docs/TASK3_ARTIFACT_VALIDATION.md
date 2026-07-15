# Görev 3 — DINOv2 Artifact Doğrulama ve Yerel Benchmark

`scripts.validate_dinov2_artifacts` aracı, production DINOv2 runtime ve coarse matching bileşenlerini değiştirmeden yerel artifact uyumluluğunu ve performansını manuel olarak ölçer. Araç internete bağlanmaz, model indirmez, competition runner çalıştırmaz ve prediction göndermez.

## Artifact yerleşimi

`.env` içinde en az şu değerler bulunmalıdır:

```dotenv
MATCHING_DINOV2_REPO_PATH=C:/models/dinov2
MATCHING_DINOV2_WEIGHTS_PATH=C:/models/dinov2_vitb14.pth
MATCHING_DINOV2_MODEL_NAME=dinov2_vitb14
MATCHING_DINOV2_DEVICE=auto
MATCHING_DINOV2_ALLOW_CPU_FALLBACK=true
```

Repository klasöründe `hubconf.py` bulunmalıdır. Weight uzantısı `.pt`, `.pth` veya `.ckpt` olmalıdır. Yalnız güvenilir kaynaktan alınmış weight kullanın; PyTorch artifact'ları güvenilmeyen veri/kod içerebilir. Production runtime desteklenen PyTorch sürümlerinde `weights_only=True` kullanır.

## Yalnız artifact ve model yükleme kontrolü

```powershell
python -m scripts.validate_dinov2_artifacts
```

Repo, `hubconf.py`, weight varlığı/uzantısı/boyutu/kısa SHA-256, model adı, PyTorch/CUDA ortamı ve CPU fallback politikası raporlanır. Artifact geçerliyse mevcut production `Dinov2RuntimeRegistry` kullanılarak model yüklenir; image verilmediğinde inference yapılmaz.

## Tek görüntü descriptor testi

```powershell
python -m scripts.validate_dinov2_artifacts --image C:/data/frame.webp
```

JPG/JPEG, PNG ve WEBP desteklenir. Gerçek format, çözünürlük, kanal sayısı, resized ölçüler, patch grid, descriptor sayısı/dimension/dtype, NaN/Inf ve L2 norm istatistikleri raporlanır.

## Pozitif eşleme testi

Referans nesnenin gerçekten bulunduğu bir frame kullanın:

```powershell
python -m scripts.validate_dinov2_artifacts `
  --reference C:/data/reference.png `
  --frame C:/data/positive_frame.jpg
```

## Negatif eşleme testi

Referans nesnenin bulunmadığı bir frame verin:

```powershell
python -m scripts.validate_dinov2_artifacts `
  --reference C:/data/reference.png `
  --frame C:/data/negative_frame.webp
```

Negatif testin `REJECTED` olması aracın başarısız çalıştığı anlamına gelmez; validation tamamlanmış, production threshold kapılarından geçerli eşleme çıkmamış demektir.

Rapor; correspondence ve similarity istatistiklerini, spatial coverage, homography durumu, inlier sayı/oranını, RMS reprojection error, polygon, görünürlük, raw/clipped bbox, confidence ve failure reason değerlerini içerir.

## Benchmark

```powershell
python -m scripts.validate_dinov2_artifacts `
  --reference C:/data/reference.png `
  --frame C:/data/positive_frame.jpg `
  --benchmark-runs 10
```

Model load süresi ayrı raporlanır. Bir warmup çalışmasından sonra şu sürelerin minimum, maksimum, ortalama, p50 ve p95 değerleri ölçülür:

- Preprocessing
- Descriptor forward
- Reference cache hit
- Coarse matching
- Homography/geometri
- Toplam iterasyon

CUDA kullanılıyorsa peak allocated ve reserved VRAM raporlanır. Araç aynı koşuda CPU ve GPU benchmark'ı birlikte yapmaz. Frame descriptor her iterasyonda yeniden üretilir; kalıcı frame cache yoktur.

Örnek benchmark JSON yapısı:

```json
{
  "model": {"load_seconds": 1.23, "device": "cuda"},
  "benchmark_metrics": {
    "runs": 10,
    "warmup_runs": 1,
    "timings": {
      "descriptor_forward_seconds": {
        "minimum": 0.01,
        "maximum": 0.02,
        "mean": 0.014,
        "p50": 0.013,
        "p95": 0.019
      }
    },
    "gpu_memory": {
      "peak_allocated_bytes": 123456,
      "peak_reserved_bytes": 234567
    }
  }
}
```

## Reference cache doğrulaması

Reference/frame modunda aynı referans iki kez talep edilir. İlk talep `MISS`, ikinci talep `HIT` olmalıdır. İkinci talepte production runtime `extract` tekrar çağrılmaz. Model hash veya görüntü hash değişirse cache anahtarı değişir. Bu cache yalnız doğrulama process'i süresince yaşar.

## JSON raporu

```powershell
python -m scripts.validate_dinov2_artifacts `
  --reference C:/data/reference.png `
  --frame C:/data/frame.png `
  --json-output C:/reports/dinov2-validation.json
```

JSON; timestamp, artifact/environment, image/descriptor, match, benchmark, final result ve failure reason bölümlerini içerir. Parola, token ve Authorization header eklenmez.

## İsteğe bağlı görselleştirme

```powershell
python -m scripts.validate_dinov2_artifacts `
  --reference C:/data/reference.png `
  --frame C:/data/frame.png `
  --save-visualization C:/reports/match.png
```

Yalnız bu seçenek verildiğinde frame'in bir kopyasına inlier noktaları, polygon ve doğrulanmış bbox çizilir. Kaynak görüntü değiştirilmez.

## Threshold gözlem modu

Araç threshold değiştirmez. Her kalite kapısı için `measured`, `threshold`, karşılaştırma operatörü ve `pass` ayrı raporlanır. Gerçek pozitif/negatif veri seti olmadan eşikleri düşürmeyin veya yükseltmeyin.

## LightGlue aşamasından önce önerilen kriterler

- Yerel model ve state-dict uyumluluğu hatasız olmalı.
- Descriptor'larda NaN/Inf olmamalı ve dimension kararlı olmalı.
- Reference cache ikinci talepte kesin HIT vermeli.
- Temsilî pozitif örneklerde homography ve bbox geometrik olarak doğru olmalı.
- Negatif örneklerde yanlış kabul oranı ölçülmeli.
- p95 gecikme ve peak VRAM yarışma donanımının sınırları içinde olmalı.
- Threshold değişikliği yalnız etiketli pozitif/negatif örneklerden elde edilen ölçümlerle yapılmalı.

Bu kriterlerin sağlanması LightGlue entegrasyonunu otomatik olarak başlatmaz; bu araç LightGlue, ALIKED veya XoFTR çağırmaz.
