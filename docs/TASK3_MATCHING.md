# Görev 3 — Referans Nesne Eşleme

## Akış

Her session kendi referans tablosuna, frame modalitesine ve model feature cache’ine
sahiptir. Referans resmi yüklenirken yalnızca bir kez indirilir, decode edilir ve
DINOv2 embedding’i cache’e alınır. Frame işlenirken aktif aralıktaki referanslar
seçilir; frame bir kez indirilip decode edilir ve aynı-modal eşleme gerekiyorsa frame
feature’ları bir kez hesaplanır.

- RGB→RGB ve termal→termal: DINOv2 dense eşleme, RANSAC homography ve varsa
  ALIKED/LightGlue refinement.
- RGB→termal ve termal→RGB: XoFTR çapraz-modal yolu.
- Metadata modalitesi önceliklidir. Yoksa basit kanal-istatistiği sınıflandırıcısı
  kullanılır; emin olunmayan değer `unknown` kalır.
- Referansın `active_from_frame` ve `active_until_frame` sınırları dahildir.
  Aralıklı referans için request’te `frame_index` yoksa referans aktif sayılmaz.
- Bir referansın hatası diğer referansların doğrulanmış sonuçlarını silmez.
- Timeout, eksik model, bozuk resim veya geçersiz bbox durumunda sahte sonuç yoktur.

## Yerel entegrasyon API’leri

Tüm aşağıdaki isteklerde `X-API-Key` gerekir. Bunlar yarışma sunucusu endpointleri
olarak yorumlanmamalıdır.

Referans ekleme/güncelleme:

```http
POST /sessions/flight-01/references
```

Gövde örneği: [examples/reference_set_request.json](examples/reference_set_request.json).
Session durumunu görmek için `GET /sessions/flight-01/references`, tek referansı
silmek için `DELETE /sessions/flight-01/references/42`, tüm session cache/state’ini
temizlemek için `DELETE /sessions/flight-01` kullanılır.

Frame isteğinde isteğe bağlı `frame_index` ve `image_modality` alanları bulunur:

```json
{
  "url": "frame-150",
  "image_url": "C:/data/frames/150.jpg",
  "video_name": "flight.mp4",
  "session": "flight-01",
  "frame_index": 150,
  "image_modality": "rgb",
  "translation_x": null,
  "translation_y": null,
  "translation_z": null,
  "gps_health_status": 0
}
```

## Yerel model kurulumu ve offline çalışma

1. Model yerleşimini [../models/matching/README.md](../models/matching/README.md)
   dosyasına göre hazırlayın.
2. `.env.example` dosyasını `.env` olarak kopyalayıp mutlak yerel yolları girin.
3. `python scripts/check_models.py` ile zorunlu DINOv2 repo/ağırlığını doğrulayın.
4. `MATCHING_DEVICE=auto` GPU varsa CUDA kullanır. GPU yoksa
   `MATCHING_ALLOW_CPU_FALLBACK=true` ile CPU’ya düşer; false ise eşleme güvenli
   biçimde devre dışı kalır.

Model runtime lazy ve process genelinde singleton’dır. `torch.hub.load` yalnızca
yerel repo ile `source="local"` ve `pretrained=False` kullanır. Çalışma zamanı model
indirmesi yapmaz. HTTP(S) frame/referans URL’leri seçilirse görüntü almak için ağ
doğal olarak gerekir; tamamen offline kullanımda yerel yollar kullanın.

## Testler

Model gerektirmeyen testler:

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

Gerçek model smoke testi için artefaktları yerleştirin, `scripts/check_models.py`
başarılı olduktan sonra uygulamayı başlatıp gerçek bir referans/frame çifti kullanın.
Repo gerçek ağırlıkları içermediğinden bu smoke test otomatik test paketinin parçası
değildir.
