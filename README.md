# AeroSync Integrated

Üç AeroSync dalının kontrollü karşılaştırmasıyla oluşturulmuş bağımsız entegrasyon
projesidir. Kaynak klasörleri değiştirilmez ve uygulama sahte yarışma çıktısı üretmez.

## Görev durumu

- **Görev 1 — disabled:** Gerçek nesne tespit kodu henüz teslim edilmedi.
  `DisabledDetectionService` açıklayıcı log yazar ve boş liste döndürür.
- **Görev 2 — VO + kalibrasyon:** Session-bazlı Shi–Tomasi/LK hareketi, sağlıklı
  ground-truth ile robust 2D rotation/scale hizalaması yapar. Kalibrasyon yoksa konum uydurmaz.
- **Görev 3 — implemented:** DINOv2 dense eşleme, isteğe bağlı ALIKED/LightGlue
  refinement ve RGB/termal çapraz-modal XoFTR yolu; session izolasyonu, aktif frame
  aralığı, lazy yerel model yükleme ve doğrulanmış bbox üretimiyle sunulur. Referans
  yoksa sonuç boş listedir.

## Kurulum ve çalıştırma

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[matching,test]"
Copy-Item .env.example .env
python scripts/check_models.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Model dosyaları ağdan otomatik indirilmez. `.env` içindeki yerel yollar açıkça
verilmelidir. Model olmadan API yine başlar; eşleme güvenli biçimde boş liste döndürür.

## API

- `GET /health`
- `POST /process_frame`
- `POST /sessions/{session_id}/references`
- `GET /sessions/{session_id}/references`
- `DELETE /sessions/{session_id}/references/{object_id}`
- `DELETE /sessions/{session_id}`

`/sessions/...` endpointleri yalnızca yerel backend entegrasyon API’leridir; yarışma
sunucusu endpointi oldukları varsayılmaz. Korumalı endpointler `X-API-Key` ister.
Referans yüklenmeden `/process_frame` çağrılırsa Görev 3 boş liste döndürür. Model,
indirme veya eşleme hatalarında sahte bbox üretilmez.

## TEKNOFEST resmî arayüz entegrasyonu

Yarışma runner’ı release içindeki resmî `official_interface` klasörünü
`OFFICIAL_INTERFACE_PATH=official_interface` üzerinden import eder ve resmî `ConnectionHandler` ile
payload sınıflarını kullanır. Resmî sunucu akışı için:

```powershell
.venv\Scripts\python.exe -m competition.runner
```

Kurulum, mapping ve güvenlik ayrıntıları
[docs/OFFICIAL_INTERFACE_INTEGRATION.md](docs/OFFICIAL_INTERFACE_INTEGRATION.md)
içindedir.

Sonuç göndermeyen, kullanıcı kontrollü bağlantı kontrolü:

```powershell
.venv\Scripts\python.exe -m competition.connection_check
```

Bu komut varsayılan olarak frame almaz ve hiçbir koşulda `prediction/` çağrısı
yapmaz. Ayrıntılar [docs/CONNECTION_CHECK.md](docs/CONNECTION_CHECK.md) içindedir.

Görev 3 kurulum, artefakt ve örnekleri [docs/TASK3_MATCHING.md](docs/TASK3_MATCHING.md)
içinde; Görev 1/2 genişletme noktaları [docs/EXTENDING_TASKS.md](docs/EXTENDING_TASKS.md)
içinde; kaynak taşıma ayrıntıları [docs/MIGRATION.md](docs/MIGRATION.md) içindedir.

## Test

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

## Final Setup / Quick Start

1. Extract `AeroSync-integrated-portable-v2.zip`.
2. Install Python 3.11 or 3.12.
3. Create and activate a clean virtual environment.
4. Run `pip install -r requirements.txt` and install the local PyTorch build required by DINOv2.
5. Copy `.env.example` to `.env`.
6. Enter team credentials only in the local `.env` file.
7. Keep `OFFICIAL_INTERFACE_PATH=official_interface`; the official interface is included.
8. Run `python -m competition.preflight_check` and require `Fail=0`.
9. Run `python -m pytest -p no:cacheprovider`.
10. Start `python -m competition.runner` only in an authorized online session.

Credential-safe release package:

```powershell
python -m scripts.prepare_release_package --output-dir release_portable_v2 --zip
```

The default release excludes `.env`, `work`, `logs`, caches, and test outputs.
Use `--include-env` only when credential disclosure is explicitly intended.

On a new computer, extract the ZIP, copy `.env.example` to `.env`, enter the
credentials, create the Python environment, install dependencies, and run the
preflight. No separate official-interface download is required.
# Task 3 local refinement artifacts

The final production candidate uses `MATCHING_GEOMETRY_METHOD=hybrid`: DINOv2
provides the coarse candidate gate and ALIKED + LightGlue perform local
refinement. All three artifacts and both local source trees are included in the
portable ZIP; runtime model downloads are never performed. Startup preload and
dummy inference warmup can take several seconds on CPU, which is expected, and
reference features are prepared before frame-local refinement timeout begins.
The supported artifact contract is documented in the runtime module docstrings.
