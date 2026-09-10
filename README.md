# Vision

TEKNOFEST **Havacılıkta Yapay Zeka Yarışması** için geliştirilen, hava aracının
alt-görüş kamerasından gelen görüntüleri işleyerek yarışmanın üç görevini
(nesne tespiti, pozisyon kestirimi, referans nesne eşleme) gerçek zamanlı olarak
çözen FastAPI tabanlı görüntü işleme backend'i. Sunucudan gelen her görüntü
karesi için üç görevin sonucunu tek bir JSON cevapta birleştirip yarışma
sunucusuna geri gönderir.

> **Durum:** Takım olarak Çevrim İçi Yarışma Simülasyonu'na katıldık ancak
> geçiş barajını aşamadık. Proje; üç görevin de uçtan uca çalışan, test edilmiş
> ve dokümante edilmiş bir implementasyonunu içerdiği için olduğu gibi paylaşılmaktadır. Yarışma sunucusuna gönderilen gerçek sonuçlar
> yoktur; kod sahte/uydurma çıktı üretmez, çözemediği durumlarda boş sonuç döner.

## Takım

| İsim | Görev | Katkı |
| --- | --- | --- |
| [Safiyye Kalemci](https://github.com/SafiyyeKalemci) (Takım Kaptanı) | Görev 3 — Referans Nesne Eşleme | DINOv2 / ALIKED / LightGlue / XoFTR eşleme pipeline'ı; genel koordinasyon, rapor ve teknik sunum |
| [Eylül Medine Kamar](https://github.com/Eylulygia) & [Şule Demir](https://github.com/suledmr) | Görev 1 — Nesne Tespiti | YOLO tabanlı tespit, hareket/iniş durumu analizi, veri seti hazırlığı ve model eğitimi |
| [Yusuf Caymaz](https://github.com/YusufCaymazZ) | Görev 2 — Pozisyon Kestirimi | Visual odometry ve GPS ground-truth kalibrasyonu |
| [Fatma Zehra Aytaş](https://github.com/Zehrayt) | Backend & Sistem Entegrasyonu (ilk faz) | İlk API/pipeline mimarisi, ön tasarım raporuna katkı |

## İçindekiler

- [Takım](#takım)
- [Yarışma Hakkında](#yarışma-hakkında)
- [Bu Projede Ne Yaptık](#bu-projede-ne-yaptık)
  - [Görev 1 — Nesne Tespiti](#görev-1--nesne-tespiti-taşıt-insan-uap-uaİ)
  - [Görev 2 — Pozisyon Kestirimi](#görev-2--pozisyon-kestirimi-visual-odometry)
  - [Görev 3 — Referans Nesne Eşleme](#görev-3--referans-nesne-eşleme-görüntü-eşleme)
- [Mimari](#mimari)
- [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
- [Yerel API](#yerel-api)
- [Test](#test)
- [Proje Yapısı](#proje-yapısı)
- [Yarışma Sunucusu Entegrasyonu](#yarışma-sunucusu-entegrasyonu)
- [Bilinen Sınırlamalar](#bilinen-sınırlamalar)


## Yarışma Hakkında

**TEKNOFEST Havacılıkta Yapay Zeka Yarışması**, hava araçlarının alt-görüş
kamerasından aldığı görüntüleri işleyerek çevresel farkındalık kazanmasını ve
GPS'siz ortamlarda konum kestirebilmesini sağlayan algoritmalar geliştirmeyi
hedefliyor. Yarışma sunucusu, 7.5 FPS ile kaydedilmiş video karelerini sırayla
API üzerinden yarışmacılara veriyor; yarışmacılar da her kare için işledikleri
sonucu JSON olarak sunucuya geri gönderiyor. Yarışma üç ayrı görevden oluşuyor:

**Görev 1 — Nesne Tespiti:** Görüntüdeki taşıt, insan, Uçan Araba Park (UAP) ve
Uçan Ambulans İniş (UAİ) alanlarının tespit edilmesi. Taşıtlar ayrıca
hareketli/hareketsiz, UAP/UAİ alanları ise iniş için uygun/uygun değil olarak
sınıflandırılmalı.

**Görev 2 — Pozisyon Kestirimi:** Hava aracının GPS'inin güvenilmez kabul
edildiği anlarda, yalnızca kamera görüntüsünden referans koordinat sistemine
göre X/Y/Z öteleme kestirimi yapılması.

**Görev 3 — Görüntü Eşleme:** Oturum başında paylaşılan (daha önce hiç
görülmemiş) referans nesnelerin, farklı açı/irtifa/kamera modalitesinden
(RGB/termal) çekilmiş video karelerinde anlık olarak tespit edilip
konumlarının bildirilmesi.

Yarışmanın resmi şartname dokümanları bu depoya eklenmiştir; ayrıntılı kural
ve puanlama tabloları için TEKNOFEST tarafından paylaşılan PDF'lere bakılabilir.

## Bu Projede Ne Yaptık

Üç görev de gerçek algoritmalarla, birbirinden bağımsız servisler olarak
implemente edildi. Hiçbir servis, güvenilir bir sonuç üretemediğinde sahte/
uydurma veri döndürmez — boş liste veya `None` döner. Bu, yarışma
şartnamesindeki "gerçek sonuç yoksa uydurma yok" prensibiyle bilinçli bir
tasarım kararıdır.

### Görev 1 — Nesne Tespiti (Taşıt, İnsan, UAP, UAİ)

- **Tespit:** Ultralytics YOLO (`models/detection/best.pt`) ile taşıt, insan,
  UAP ve UAİ sınıflarının bounding box tespiti.
- **Hareket durumu (taşıt):** Kamera sürekli hareket ettiği için basit piksel
  farkı yetmiyor. Ardışık kareler arasında Shi–Tomasi + RANSAC ile kameranın
  kendi hareketini temsil eden bir **homografi** kestiriliyor; her taşıt kutusu
  bu homografiyle önceki kareye izdüşürülüp kalan fark (residual) eşiğe
  göre "hareketli"/"hareketsiz" etiketleniyor — kamera kaymasıyla taşıtın
  gerçek hareketi böyle ayrıştırılıyor (`DETECTION_MOTION_METHOD` ile birkaç
  strateji arasında seçim yapılabiliyor).
- **İniş uygunluğu (UAP/UAİ):** Taşıt/insan kutularıyla UAP/UAİ alanının
  kesişimi hesaplanıp yeterince kaplanmışsa veya merkezi engel içindeyse alan
  "uygun değil" sayılıyor; kare dışına taşan alan da otomatik "uygun değil".
- Session bazlı state tutulur; video değişimi, kare atlaması veya dondurulmuş
  görüntü gibi durumlarda hareket kararı güvenilmezse `UNKNOWN`'a düşülür.

Kod: [`app/services/detection/`](app/services/detection/) · Ayrıntılar:
[`docs/TASK1_MOTION.md`](docs/TASK1_MOTION.md),
[`docs/TASK1_LANDING.md`](docs/TASK1_LANDING.md)

### Görev 2 — Pozisyon Kestirimi (Visual Odometry)

- **Görüntü düzlemi hareketi:** Shi–Tomasi köşeleri piramidal Lucas–Kanade ile
  iki yönlü (forward-backward) izlenir, tutarsız izler elenir; kalan
  noktalara RANSAC + en küçük kareler ile 3 parametreli bir model
  (öteleme + yaw) fit edilir.
- **Ölçek ve eksen hizalama:** Piksel hareketi tek başına metre vermediği
  için, GPS'in sağlıklı olduğu ilk pencerede kamera adımları ile gerçek GPS
  X/Y adımları **2D Procrustes/Kabsch** ile hizalanır; ölçek ve eksen dönüşü
  medyan + MAD tabanlı aykırı değer filtresiyle robust biçimde kalibre edilir.
- **GPS kaybı simülasyonu:** GPS sağlıklıyken gerçek koordinat aynen
  döndürülür; sağlıksız olduğu anda son sağlıklı konum çapa (anchor) alınıp
  VO'nun ürettiği göreli hareket buna eklenerek konum kestirilir. Kalibrasyon
  veya devamlılık koşulları sağlanmıyorsa `None` döner — uydurma konum yok.
- Z ekseni için üç politika desteklenir; varsayılan, GPS kaybında son bilinen
  Z değerini sabit tutmaktır (`hold_last_valid_z`).

Kod: [`app/services/localization/`](app/services/localization/) · Ayrıntılar:
[`docs/TASK2_AFFINE_VO.md`](docs/TASK2_AFFINE_VO.md),
[`docs/TASK2_CALIBRATION.md`](docs/TASK2_CALIBRATION.md)

### Görev 3 — Referans Nesne Eşleme (Görüntü Eşleme)

- **Coarse eşleme:** Referans görüntü ve video karesi yerel bir **DINOv2**
  modeliyle dense patch descriptor'larına dönüştürülür; mutual
  nearest-neighbour eşleşmelerden **USAC_MAGSAC** ile homografi kestirilir.
  Inlier oranı, reprojeksiyon hatası ve geometrik tutarlılık (konveks
  dörtgen, görünür alan oranı vb.) kontrol edilerek sahte eşleşmeler elenir.
- **İnce ayar (opsiyonel, `hybrid` mod):** DINOv2'nin bulduğu aday bölge
  **ALIKED + LightGlue** ile yerel olarak yeniden eşleştirilip daha hassas bir
  bbox üretir; yetersiz kalırsa DINOv2 sonucuna geri dönülür.
- **Çapraz modalite (RGB↔termal):** Referans ve kare farklı kamera tipinden
  geldiğinde DINOv2 yerine **XoFTR** tabanlı çapraz-modal eşleme kullanılır.
- Her session kendi referans kataloğunu, aktif kare aralığını ve descriptor
  cache'ini taşır; bir referanstaki hata diğerlerini etkilemez. Confidence
  skoru inlier oranı, benzerlik, reprojeksiyon hatası, görünürlük ve uzamsal
  kapsamanın ağırlıklı ortalamasıdır; eşik altı sonuçlar reddedilir.

Kod: [`app/services/matching/`](app/services/matching/) · Ayrıntılar:
[`docs/TASK3_MATCHING.md`](docs/TASK3_MATCHING.md),
[`docs/TASK3_COARSE_MATCHING.md`](docs/TASK3_COARSE_MATCHING.md),
[`docs/TASK3_DINOV2_RUNTIME.md`](docs/TASK3_DINOV2_RUNTIME.md),
[`docs/TASK3_ARTIFACT_VALIDATION.md`](docs/TASK3_ARTIFACT_VALIDATION.md)

## Mimari

```text
Yarışma sunucusu (resmî Takım Bağlantı Arayüzü)
        │  login / progress / frame / translation / reference / prediction
        ▼
competition/runner.py ──▶ official_interface_adapter.py ──▶ frame_mapper.py / reference_mapper.py
        │                                                          │
        │                                                          ▼
        │                                          app/services/frame_processor.py
        │                                                          │
        │                              ┌───────────────────────────┼───────────────────────────┐
        │                              ▼                           ▼                           ▼
        │                    Görev 1: detection/          Görev 2: localization/       Görev 3: matching/
        │                              │                           │                           │
        │                              └───────────────────────────┼───────────────────────────┘
        │                                                          ▼
        │                                              CompetitionResponse
        ▼                                                          │
result_mapper.py  ◀───────────────────────────────────────────────┘
        │
        ▼
ConnectionHandler.send_prediction()  →  yarışma sunucusu
```

Aynı `FrameProcessor` ve üç görev servisi, yerel geliştirme/test için doğrudan
bir FastAPI uygulaması (`app/main.py`) üzerinden `POST /process_frame` ile de
çağrılabilir — resmi arayüz olmadan da tek tek test edilebilir.

## Kurulum ve Çalıştırma

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[matching,test]"
Copy-Item .env.example .env
python scripts/check_models.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Model ağırlıkları (`*.pt`, `*.pth`, `*.ckpt`) boyut ve lisans nedeniyle
depoya dahil değildir (`.gitignore`) ve ağdan otomatik indirilmez;
`.env` içindeki yollara yerel olarak yerleştirilmelidir
(bkz. [`models/matching/README.md`](models/matching/README.md)). Model
bulunamazsa ilgili görev API'yi düşürmeden devre dışı kalır ve güvenli
biçimde boş sonuç döner.

`.env.example`, üç görevi de gerçek modellerle (YOLO + homografi tabanlı
hareket/iniş analizi, Affine VO + GPS kalibrasyonu, DINOv2 + ALIKED/LightGlue
hibrit eşleme) etkinleştirilmiş şekilde gelir; herhangi bir görevi
`..._ENABLED=false` yaparak tek başına kapatmak mümkündür.

## Yerel API

- `GET /health` — her görevin etkin/devre dışı durumunu döndürür.
- `POST /process_frame` — bir görüntü karesini üç görevden de geçirip
  `CompetitionResponse` döndürür (resmi yarışma endpoint'i **değildir**, yerel
  test/entegrasyon amaçlıdır).
- `POST /sessions/{session_id}/references` — Görev 3 için referans nesne
  yükler.
- `GET /sessions/{session_id}/references` — session'daki referansları listeler.
- `DELETE /sessions/{session_id}/references/{object_id}` — tek referansı siler.
- `DELETE /sessions/{session_id}` — session'ın tüm state'ini (tespit, konum,
  eşleme) temizler.

`/health` dışındaki tüm endpoint'ler `X-API-Key` header'ı ister
(`AEROSYNC_SECRET_KEY`).

## Test

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

`tests/` altında 40'ın üzerinde dosyada; hareket/iniş analizi, VO + GPS
kalibrasyonu, DINOv2/ALIKED/LightGlue/XoFTR eşleme, resmî arayüz entegrasyonu,
preflight/connection-check araçları ve API sözleşmesi için testler bulunur.
Testler gerçek model ağırlığı veya ağ bağlantısı gerektirmez; `online`
işaretli testler varsayılan koşuda dışlanır.

## Proje Yapısı

```text
app/
├── api/            # FastAPI router + API key doğrulama
├── core/           # Settings (.env okuma/doğrulama), logging
├── schemas/        # Pydantic şemaları (yarışma JSON sözleşmesi)
├── services/
│   ├── detection/     # Görev 1: YOLO + homografi tabanlı hareket + iniş analizi
│   ├── localization/  # Görev 2: Affine VO + GPS kalibrasyonu
│   └── matching/      # Görev 3: DINOv2 + ALIKED/LightGlue + XoFTR
│       └── gorev3/    # Alternatif "gorev3" motoru (MATCHING_ENGINE=gorev3)
├── frame_processor.py # Üç görevi tek istekte orkestre eder
└── registry.py        # Config'e göre gerçek/disabled servis seçimi

competition/        # Resmî yarışma sunucusu istemcisi, adapter, runner
official_interface/ # TEKNOFEST'in paylaştığı resmî bağlantı arayüzü (değiştirilmedi)
models/              # Model ağırlıkları için yerel yerleşim (git'e dahil değil)
external/            # LightGlue, DINOv2, XoFTR üçüncü parti kaynak kodları
scripts/             # Model doğrulama, benchmark, validation, release paketleme
docs/                # Her görev için ayrıntılı tasarım dokümanları
tests/                # pytest test paketi
```

## Yarışma Sunucusu Entegrasyonu

TEKNOFEST'in bizimle paylaştığı resmî **Takım Bağlantı Arayüzü** kodu
(`official_interface/`) değiştirilmeden kullanılır; `ConnectionHandler` ile
giriş yapılır, aktif oturum bulunur, kareler sırayla istenir ve sonuçlar aynı
sözleşmeyle geri gönderilir. `competition/` klasöründeki adapter katmanı bu
resmî arayüz ile kendi `FrameProcessor`'ımız arasında köprü kurar:

- `official_interface_adapter.py` — resmî istemciyi sarmalar.
- `frame_mapper.py` / `reference_mapper.py` — resmî JSON'u kendi şemamıza,
  kendi sonucumuzu resmî DTO'lara çevirir.
- `runner.py` — bir sonuç kabul edilmeden yeni kare istemez; 401/406 gibi
  durumları, retry'ı ve idempotency'yi yönetir.
- `preflight_check.py` / `connection_check.py` — yarışma günü öncesi
  bağımlılık, model, config ve (isteğe bağlı, salt-okunur) sunucu bağlantısı
  kontrolü yapar; **hiçbir aşamada sonuç göndermez**.

Ayrıntılar: [`docs/OFFICIAL_INTERFACE_INTEGRATION.md`](docs/OFFICIAL_INTERFACE_INTEGRATION.md),
[`docs/PREFLIGHT_CHECK.md`](docs/PREFLIGHT_CHECK.md),
[`docs/CONNECTION_CHECK.md`](docs/CONNECTION_CHECK.md)

Yarışma sunucusuna bağlanmak için `.env` içinde ayrıca şu değişkenler
doldurulmalıdır:

```dotenv
TEAM_NAME=
PASSWORD=
EVALUATION_SERVER_URL=http://havaciliktayapayzeka.teknofest.org:1025/
SESSION_NAME=ONLINE_YARISMA_2026
OFFICIAL_INTERFACE_PATH=official_interface
```

Yarışma günü tek komut:

```powershell
.venv\Scripts\python.exe -m competition.runner
```

Sonuç göndermeyen, salt-okunur bağlantı kontrolü:

```powershell
.venv\Scripts\python.exe -m competition.connection_check
```

Yarışma öncesi kapsamlı kontrol (bağımlılık, model, config, testler, isteğe
bağlı online metadata kontrolü):

```powershell
.venv\Scripts\python.exe -m competition.preflight_check --strict --run-tests
```

## Bilinen Sınırlamalar

- Görev 1 hareket sınıflandırması yalnızca kamera ego-motion'ını 2D homografi
  ile modeller; roll/pitch, zoom veya güçlü parallax içeren sahnelerde hata
  payı artar.
- Görev 2'deki 3 parametreli (öteleme + yaw) model gerçek irtifa/roll/pitch
  hareketini temsil etmez; GPS kalibrasyon penceresi kalitesiz geçerse konum
  kestirimi `None` döner.
- Görev 3'te LightGlue/ALIKED ince ayarı ve XoFTR çapraz-modal yolu, ayrı
  eşik/performans doğrulaması gerektiren opsiyonel katmanlardır; varsayılan
  güvenli davranış, yetersiz kanıtta boş sonuç döndürmektir.
- Proje, Çevrim İçi Yarışma Simülasyonu'nda gerekli başarı barajını
  geçemediği için TEKNOFEST finaline kalınmamıştır.