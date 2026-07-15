# TEKNOFEST Güvenli Bağlantı Kontrolü

`competition.connection_check`, resmî Takım Bağlantı Arayüzü üzerinden kullanıcı
kontrollü ve salt-okunur bir bağlantı dry-run testi yapar. Model çalıştırmaz, tam
oturum başlatmaz ve sonuç göndermez.

## `.env` kurulumu

`.env.example` dosyasını `.env` olarak kopyalayıp şu alanları doldurun:

```dotenv
TEAM_NAME=
PASSWORD=
EVALUATION_SERVER_URL=http://havaciliktayapayzeka.teknofest.org:1025/
SESSION_NAME=ONLINE_YARISMA_2026
OFFICIAL_INTERFACE_PATH=C:/path/to/TAKIM_BAGLANTI_ARAYUZU
```

Parola, token ve Authorization header ekrana veya loglara yazılmaz. Config özeti
parola için yalnızca `configured` gösterir.

## Varsayılan güvenli test

```powershell
.venv\Scripts\python.exe -m competition.connection_check
```

Varsayılan akış yalnızca config doğrulama, login, progress ve referans metadata
okuma adımlarını yapar. Frame veya translation istemez; referans görüntülerini
indirmez ve prediction göndermez.

Referans dosyalarının formatını da doğrulamak için açıkça şunu kullanın:

```powershell
.venv\Scripts\python.exe -m competition.connection_check --fetch-references
```

## Tek frame ve translation kontrolü

```powershell
.venv\Scripts\python.exe -m competition.connection_check --fetch-frame
```

Bu seçenek:

- `get_current_frame()` fonksiyonunu tam bir kez çağırır,
- aynı frame için `get_current_translation()` fonksiyonunu tam bir kez çağırır,
- başka frame istemez,
- görüntüyü `BytesIO`, NumPy ve OpenCV ile RAM'de decode eder,
- JPEG/JPG, PNG veya WEBP magic byte ve URL uzantısını raporlar,
- genişlik, yükseklik, kanal sayısı, dtype ve OpenCV shape bilgisini raporlar,
- çözünürlüğün `1920x1080` olup olmadığını kontrol eder,
- güvenli frame ve translation metadata alanlarını gösterir,
- model, sonuç dönüştürücü veya prediction gönderim kodu çalıştırmaz.

Resmî indirme yardımcısı zorunlu olarak dosya oluşturduğu için araç bu dosyayı
yalnızca geçici taşıyıcı olarak kullanır: byte'ları hemen RAM'e okur ve başarı ya
da hata durumunda `finally` bloğunda siler. Kullanıcının çalışma dizinine frame
dosyası yazılmaz ve indirilen dosya çalışmanın sonunda tutulmaz.

Translation sunucudan sağlanmıyorsa varsayılan çıkış kodu `70` olur. Yalnızca bu
durumu uyarı kabul ederek görüntü doğrulamasını başarılı saymak için:

```powershell
.venv\Scripts\python.exe -m competition.connection_check --fetch-frame --allow-missing-translation
```

Timeout ve ayrıntılı güvenli log seçenekleri:

```powershell
.venv\Scripts\python.exe -m competition.connection_check --timeout 20 --verbose
```

## Prediction gönderiminin yapılamaması

Dry-run aracının `ReadOnlyOfficialAdapter` protokolünde prediction gönderme metodu
yoktur. Modül `send_prediction`, `FramePredictions` veya sonuç eşleme katmanını
import etmez. Login için gerekli resmî authentication çağrısı dışında bütün
sunucu okumaları GET'tir. Test doubles, frame akışında tam bir frame ve tam bir
translation istendiğini ve prediction sayısının daima sıfır kaldığını doğrular.

## Beklenen özet

```text
Dry-run summary:
  Authentication: OK
  Progress: OK
  Session match: OK
  References: 3
  Frame fetch: OK
  Translation fetch: OK
  Image decode: OK
  Resolution: OK (1920x1080)
  Frames requested: 1
  Predictions submitted: 0
  Prediction submission: DISABLED
```

Frame alınmayan varsayılan çalışmada frame, translation, decode ve resolution
alanları `skipped`, frame sayısı `0` olur. Aktif sunucu oturumu `SESSION_NAME` ile
farklıysa session durumu açıkça `WARNING` gösterir. Eksik metadata alanları
`not provided` olarak yazılır; dosya adından çıkarılan frame numarası ayrıca
`derived` etiketiyle belirtilir.

## Hata kodları

| Kod | Anlam |
|---:|---|
| `0` | Dry-run başarıyla tamamlandı |
| `10` | Config eksik veya resmî arayüz yüklenemedi |
| `20` | Login başarısız |
| `30` | Progress başarısız veya aktif oturum yok |
| `40` | Reference metadata/indirme doğrulaması başarısız |
| `50` | Tek frame alınamadı |
| `60` | Görüntü indirilemedi, bozuk veya desteklenmeyen formatta |
| `70` | Translation alınamadı veya sağlanmadı |

`1920x1080` dışındaki geçerli görüntüler açık bir `WARNING` üretir; decode
başarılı olduğu için tek başına fatal hata sayılmaz.

## Otomatik testler

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_connection_check.py
```

Testler yalnızca test doubles ve bellekte üretilen byte dizileri kullanır.
`connection_check` komutu pytest tarafından otomatik çalıştırılmaz; gerçek sunucu
bağlantısı yalnızca kullanıcı komutu açıkça çalıştırdığında oluşur.
