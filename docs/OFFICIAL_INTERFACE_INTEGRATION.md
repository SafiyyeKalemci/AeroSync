# TEKNOFEST Resmî Takım Bağlantı Arayüzü Entegrasyonu

## İncelenen resmî kod

Entegrasyon aşağıdaki dosyalardaki gerçek sözleşmeye dayanır:

- `main.py`: ana sıralı akış, progress ile resume, frame kapısı ve bitiş koşulu.
- `src/connection_handler.py`: `ConnectionHandler` login, progress, frame,
  translation, reference ve prediction fonksiyonları.
- `src/frame_predictions.py`: resmî sonuç zarfı ve `create_payload()`.
- `src/detected_object.py`: Görev 1 resmî alanları ve sınıf URL’si.
- `src/detected_translation.py`: Görev 2 resmî alanları.
- `src/reference_prediction.py`: Görev 3 resmî referans sonucu.
- `src/constants.py`: sınıf, iniş ve hareket kodları.
- `src/object_detection_model.py`: yalnızca resmî, token destekli
  `download_image()` yardımcı fonksiyonu. Bu dosyadaki örnek/mock `detect()` hiçbir
  zaman çağrılmaz.

Resmî endpointler yalnızca `ConnectionHandler` kodundan alınmıştır:

| İşlem | Resmî yöntem | Resmî yol |
|---|---|---|
| Login | `login()` | `auth/` |
| İlerleme/aktif oturum | `get_progress()` | `progress/` |
| Sıradaki tek frame | `get_current_frame()` | `frames/` |
| Frame translation/health | `get_current_translation()` | `translation/` |
| Referans kataloğu | `get_reference_objects()` | `reference/` |
| Sonuç gönderme | `send_prediction()` | `prediction/` |

Login `username/password` form verisi gönderir ve yanıttaki `token` alanını alır.
Sonraki çağrılar `Authorization: Token <token>` biçimini kullanır. Başarılı sonuç
gönderimi HTTP 201, önceden kabul edilmiş aynı frame HTTP 406 döndürür.

Resmî client oturum oluşturma/başlatma fonksiyonu sağlamaz. `url_session` alanı
tanımlı olsa da kullanılmaz. Bu entegrasyon endpoint uydurmaz: `progress/` ile aktif
oturumu bulur ve `SESSION_NAME` ile eşleştiğini doğrular. Aktif oturum yoksa veya ad
eşleşmiyorsa kontrollü exit code üretir.

## Adapter mimarisi

```text
Resmî ConnectionHandler + resmî payload sınıfları
  -> OfficialInterfaceAdapter
  -> frame_mapper / reference_mapper
  -> AeroSync FrameProcessor
  -> Görev 1 + Görev 2 + Görev 3 servisleri
  -> CompetitionResponse
  -> result_mapper
  -> resmî FramePredictions
  -> ConnectionHandler.send_prediction()
```

- `official_interface_adapter.py` dış klasördeki resmî sınıfları import eder.
- `frame_mapper.py` resmî frame/translation JSON’unu `FrameRequest` yapar.
- `reference_mapper.py` resmî reference URL’si ile dahili `object_id` arasında
  katalog tutar. Resmî payload’da `object_id` olmadığından katalog sırasına göre
  1’den başlayan yalnızca-dahili kimlik kullanılır.
- Sayısal `frame_start/frame_end` doğrudan korunur. Bunlar URL ise resmî ana kodun
  kullandığı `frame_start_image_url <= image_url <= frame_end_image_url` seçimi
  uygulanır; endpoint veya frame indeksi tahmin edilmez.
- `result_mapper.py` AeroSync sonucunu resmî DTO nesnelerine açık alan eşlemesiyle
  dönüştürür. JSON’u resmî `create_payload()` üretir.
- `runner.py` bir sonuç kabul edilmeden yeni frame istemez.

## Ortam kurulumu

`.env.example` dosyasını `.env` olarak kopyalayın:

```dotenv
TEAM_NAME=
PASSWORD=
EVALUATION_SERVER_URL=http://havaciliktayapayzeka.teknofest.org:1025/
SESSION_NAME=ONLINE_YARISMA_2026
OFFICIAL_INTERFACE_PATH=C:/path/to/TAKIM_BAGLANTI_ARAYUZU
```

`PASSWORD` kaynak kodda veya örnek dosyada bulunmaz. URL tam olarak bir son slash’a
normalize edilir. Runner; TEAM_NAME, PASSWORD, EVALUATION_SERVER_URL, SESSION_NAME
ve OFFICIAL_INTERFACE_PATH eksikse ağa çıkmadan startup hatası verir.

`.env` ve indirilen resmî media dizini `.gitignore` kapsamındadır. Login sırasında
resmî client’ın credential payload logu bastırılır. Password, token ve Authorization
header adapter loglarına yazılmaz.

## Çalıştırma

Yarışma günü tek komut:

```powershell
.venv\Scripts\python.exe -m competition.runner
```

Test sunucusunda `.env` içindeki `EVALUATION_SERVER_URL` ve `SESSION_NAME` resmî test
değerlerine ayarlandıktan sonra çalıştırılacak komut aynıdır:

```powershell
.venv\Scripts\python.exe -m competition.runner
```

Gerçek veya test sunucusuna bağlanmadan yerel doğrulama:

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_official_integration.py
```

## Sıralama, retry ve idempotency

- Resmî client’ın retry parametreleri merkezi config’ten geçirilir.
- Prediction gönderimi runner seviyesinde de aynı pending payload ile sınırlı,
  exponential retry uygular; retry sırasında yeni frame alınmaz.
- 201 kabul, 406 daha önce kabul edilmiş/idempotent sonuç sayılır.
- 401 sonucunda credential loglamadan yeniden login olunur ve aynı sonuç denenir.
- Resmî GET yardımcıları HTTP status döndürmediği için frame/progress alınamazsa bir
  kez güvenli re-auth yapılır; hâlâ başarısızsa açık hata ve exit code üretilir.
- Başarıyla gönderilen frame ID’leri process belleğinde tutulur. Aynı frame yeniden
  gelirse ikinci gönderim yapılmaz ve kontrollü durulur.
- Yeni runner başlangıcında matching/localization session state temizlenir.

## Resmî JSON eşlemeleri

Frame:

| Resmî alan | AeroSync alanı |
|---|---|
| `url` | `FrameRequest.url` ve değişmeden sonuç `frame` alanı |
| indirilen `image_url` | yerel `FrameRequest.image_url` |
| `video_name` | `FrameRequest.video_name` |
| progress `session_name` | `FrameRequest.session` |
| progress `frame_index` | `FrameRequest.frame_index` |
| translation `health_status` | `gps_health_status` (`0`, `1`, `null`) |
| translation `translation_x/y/z` | aynı isimler; null korunur |

Sonuç:

| AeroSync alanı | Resmî alan |
|---|---|
| `frame` | `frame` |
| `detected_objects[].cls` | `classes/{1..4}/` URL’si |
| `landing_status` | `1`, `0`, `-1` |
| `motion_status` | `moving_status`: `1`, `0`, `-1` |
| bbox koordinatları | aynı koordinat isimleri, resmî sınıf string üretir |
| `detected_translations` | aynı translation alanları |
| `detected_undefined_objects[].object_id` | katalogdaki resmî `reference` URL’si |
| Görev 3 bbox | `reference_predictions[]` |

Resmî payload confidence alanı kabul etmediğinden confidence gönderilmez. Bu sessiz
alan kaybı değil, resmî DTO sözleşmesinin açık mapping kararıdır.

## Hata ve exit code özeti

- `2`: kimlik doğrulama başarısız.
- `3`: aktif/beklenen oturum bulunamadı.
- `4`: progress/sunucu erişimi başarısız.
- `5`: aktif oturumda frame alınamadı.
- `6`: AeroSync frame task timeout.
- `7`: prediction retry sonunda kabul edilmedi; pending frame korunur.
- `8`: başarıyla gönderilmiş frame yeniden geldi; çift gönderim engellendi.
- `9`: resmî arayüz import/media/client hatası.
- `10`: eksik veya geçersiz resmî entegrasyon yapılandırması.

## Resmî arayüz güncellenirse

Önce `src/connection_handler.py`, `src/frame_predictions.py`, üç sonuç DTO dosyası,
`src/constants.py`, `src/object_detection_model.py` ve `main.py` yeniden kontrol
edilmelidir. Endpoint, status code, token biçimi, reference alanları veya payload
şekli değişirse adapter/mapping testleri güncellenmeden yarışma sunucusuna
bağlanılmamalıdır.
