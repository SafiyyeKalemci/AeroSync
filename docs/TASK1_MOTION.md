# Görev 1 Taşıt Hareket Analizi

Bu aşama yalnız `motion_status` üretir. Landing suitability hesaplaması içermez;
Görev 2, Görev 3 ve resmî sonuç eşleme davranışı değişmez.

## Veri akışı

1. Frame bir kez decode edilir ve YOLO bir kez çalıştırılır.
2. Geçerli kutular görüntü sınırlarına kırpılır.
3. Session'a ait önceki gri frame alınır.
4. Frame sürekliliği doğrulanır.
5. Gerekiyorsa Farnebäck dense flow frame başına bir kez hesaplanır.
6. Taşıt ve insan kutuları global akış kestiriminden maskelenir.
7. Kalan sonlu flow vektörlerinin medyanı kamera öteleme baseline'ı olur.
8. Her taşıtın iç ROI'sindeki medyan flow'dan global medyan çıkarılır.
9. Residual büyüklük config eşiğiyle karşılaştırılır.

İnsan, UAP ve UAİ için motion daima `UNKNOWN` kalır.

## Session state

Her `session_id` bağımsız olarak şunları tutar:

- video adı ve önceki frame kimliği/index'i
- önceki gri frame ve çözünürlük
- warmup sayacı
- son sonuç cache'i
- freeze sayacı
- session bazlı `asyncio.Lock`
- son erişim zamanı

Store, TTL ile eski session'ları ve kapasite dolduğunda en eski kilitlenmemiş
session'ı temizler. API session silme ve yarışma runner başlangıcı detection state'i
resetler.

## İlk frame, duplicate ve süreksizlik politikası

- İlk frame YOLO sonuçlarını üretir; bütün motion değerleri `UNKNOWN` olur ve frame
  baseline olarak saklanır.
- Aynı frame ID tekrar gelirse inference/flow tekrarlanmaz ve cache döndürülür.
- Video adı veya frame şekli değişirse mevcut frame yeni baseline olur.
- Index geriye giderse ya da izin verilen gap aşılırsa motion `UNKNOWN` olur.
- Bozuk/decode edilemeyen frame önceki state'i değiştirmez.
- Aynı görüntü farklı ID ile gelirse freeze şüphesi oluşur ve motion `UNKNOWN` olur.
- Warmup bitene kadar state ve flow hazırlanabilir, ancak motion `UNKNOWN` kalır.

## Config

```dotenv
DETECTION_MOTION_ENABLED=true
DETECTION_MOTION_THRESHOLD_PX=2.0
DETECTION_MOTION_MIN_VALID_PIXELS=25
DETECTION_MOTION_INNER_CROP_RATIO=0.15
DETECTION_MOTION_MAX_FRAME_GAP=1
DETECTION_MOTION_WARMUP_FRAMES=1
DETECTION_MOTION_FLOW_DOWNSCALE=0.5
DETECTION_MOTION_FREEZE_THRESHOLD=0.0
DETECTION_MOTION_SESSION_TTL_SECONDS=1800
DETECTION_MOTION_MAX_SESSIONS=32
```

`INNER_CROP_RATIO`, bbox'ın her kenarından atılan oranı ifade eder ve `0 <= r <
0.5` olmalıdır. Flow downscale `0 < scale <= 1` aralığındadır. Residual vektörler
karar öncesinde tekrar orijinal görüntü piksel ölçeğine dönüştürülür.

## Performans ve doğruluk sınırları

Downscale, 1920x1080 dense flow maliyetini azaltır. Flow aynı frame'deki bütün
taşıtlar tarafından paylaşılır; her taşıt için tekrar hesaplanmaz.

Global medyan yalnız baskın kamera ötelemesi için güvenli bir baseline'dır. Kamera
dönüşü, zoom, perspektif değişimi, parallax ve yetersiz zemin dokusunda uzamsal
değişen kamera hareketini temsil edemez. Güvenilir global flow üretilemezse sonuç
`UNKNOWN` olur. Gelecek aşamada, mevcut API'yi değiştirmeden feature+RANSAC tabanlı
affine veya homography modeli değerlendirilebilir.

## Test

```powershell
C:\venvs\aerosync\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp="work\pytest" tests/test_motion_analyzer.py
C:\venvs\aerosync\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp="work\pytest"
```

Testler sahte YOLO sonuçları ve deterministik flow alanları kullanır; `best.pt`
veya ağ bağlantısı gerektirmez.
