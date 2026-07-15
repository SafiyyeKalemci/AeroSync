# AeroSync Yarışma Öncesi Kontrolü

`competition.preflight_check`, yarışma veya test oturumundan önce proje yapısını,
bağımlılıkları, yapılandırmayı, artifact'leri, depolamayı, GPU durumunu, FastAPI
uygulamasını ve güvenlik kurallarını tek komutla denetler. Varsayılan çalışma
tamamen yereldir; model inference ve gerçek sunucu bağlantısı yapmaz.

## Yerel kontrol

```powershell
python -m competition.preflight_check
```

Her kontrol `[OK]`, `[WARNING]`, `[FAIL]` veya `[SKIPPED]` olarak raporlanır.
Parola, token, Authorization header ve gerçek credential değerleri terminale veya
JSON raporuna yazılmaz.

GPU ve model artifact kontrollerini bilinçli olarak atlamak için:

```powershell
python -m competition.preflight_check --skip-gpu --skip-models
```

## Online metadata kontrolü

Gerçek sunucuya yalnızca açık `--online` seçeneğiyle bağlanılır:

```powershell
python -m competition.preflight_check --online
```

Bu mod mevcut güvenli `competition.connection_check` akışını kullanır ve yalnızca
authentication, progress ve reference metadata okur. Frame, translation veya
prediction istemez.

## Tek frame güvenli kontrolü

```powershell
python -m competition.preflight_check --online --fetch-frame
```

Bu komut en fazla bir frame ve aynı frame'e ait bir translation alır. Sonuç
göndermez, model çalıştırmaz ve ikinci frame istemez. `--fetch-frame`, `--online`
olmadan kullanılamaz.

## Test çalıştırma

Varsayılan kontrol yalnızca pytest collection yapar. Tüm yerel testleri çalıştırmak
için:

```powershell
python -m competition.preflight_check --run-tests
```

Test alt sürecinde bytecode yazımı kapatılır ve gerçek sunucu kullanılmadığı rapora
işlenir. `online` işaretli testler dışlanır; ayrıca preflight test alt sürecinde
socket bağlantıları teknik olarak engellenir. Online bağlantı ancak ayrıca
`--online` verilirse ve test süreci tamamlandıktan sonra yapılır.

## JSON raporu

```powershell
python -m competition.preflight_check --json-output work/preflight-report.json
```

Rapor `timestamp`, `project_root`, `python_version`, `checks`, `summary`,
`readiness` ve `exit_code` alanlarını içerir. Hassas yapılandırma değerleri rapora
eklenmez.

## Strict mode

Normal modda warning durumları exit code `0` ile tamamlanabilir. Warning'lerin de
CI veya yarışma günü hazırlığını durdurması isteniyorsa:

```powershell
python -m competition.preflight_check --strict
```

Strict modda en az bir warning ve hiç fail yoksa exit code `1` döner.

## Exit kodları

| Kod | Açıklama |
|---:|---|
| `0` | Hazır veya normal modda yalnız warning var |
| `1` | Strict modda warning var |
| `10` | Kritik config/proje yapısı hatası |
| `20` | Required dependency eksik |
| `30` | Resmî arayüz eksik/geçersiz |
| `40` | Kritik model veya kamera artifact hatası |
| `50` | Uygulama import/startup/OpenAPI hatası |
| `60` | Güvenlik ihlali |
| `70` | Online bağlantı kontrolü başarısız |
| `80` | Pytest collection veya test çalıştırma başarısız |

## Yarışma günü önerilen sıra

1. Bağımlılık ve artifact kontrolü: `python -m competition.preflight_check --strict`
2. Tam yerel test: `python -m competition.preflight_check --strict --run-tests`
3. Güvenli sunucu metadata kontrolü: `python -m competition.preflight_check --strict --online`
4. Gerekliyse tek frame doğrulaması: `python -m competition.preflight_check --strict --online --fetch-frame`

Hiçbir aşamada preflight aracının prediction gönderim yolu yoktur. Gönderim yalnızca
ayrı yarışma runner akışında mümkündür.
