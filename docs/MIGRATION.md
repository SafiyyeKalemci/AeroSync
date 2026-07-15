# Migration raporu

## Kaynak seçimi

### AeroSync-main

- FastAPI router, Pydantic yanıt şeması, API anahtarı ve yarışma ajanı fikirleri
  temel alındı.
- Kod doğrudan kopyalanmadı: global log dosyası, SQLite yan etkisi, mock servisler,
  süreç içi model timeout deseni ve sabit varsayılan kimlikler çıkarıldı.
- Çoklu `detected_undefined_objects` davranışı korundu.

### AeroSync-feature-safiyye

- Görev 3 için en gelişmiş yaklaşım seçildi.
- `gorev3/dinov2_matcher.py` içindeki dense DINOv2 patch eşleme, mutual nearest
  neighbor, MAGSAC homografi, ALIKED/LightGlue refinement ve XoFTR fallback
  mimarisi yeni modüllere ayrılarak taşındı.
- Eski import-anında model indirme davranışı kaldırıldı.
- `gorev3/XoFTR` eğitim deposu ve 42 MB checkpoint körü körüne kopyalanmadı.
  Bunun yerine sürümden bağımsız, yerel TorchScript XoFTR adapter sözleşmesi
  tanımlandı.
- Referans yokken dönen sabit bbox tamamen kaldırıldı.

### AeroSync-yusuf

- Stateful pozisyon kestiriminin her frame'i görmesi ve session bazında sıralı
  işlenmesi gereksinimi tasarıma alındı.
- Görev 2 tamamlanmadığı için AffineVO/DPVO kodu taşınmadı. Interface ve kilitli
  session store hazırlandı; disabled servis koordinat üretmiyor.

## Yeni dosya eşlemesi

| Yeni dosya | Kaynak/karar |
|---|---|
| `app/api/endpoints.py` | Main endpoint sözleşmesi yeniden yazıldı |
| `app/core/config.py` | Main config fikri genişletildi; tüm görev ayarları merkezileştirildi |
| `competition/*` | Main `competition_agent.py` sorumluluklara ayrıldı |
| `matching/dinov2_engine.py` | Safiyye dense DINOv2 algoritması yeniden paketlendi |
| `matching/lightglue_adapter.py` | Safiyye ALIKED/LightGlue refinement yerel artifact sözleşmesine çevrildi |
| `matching/xoftr_adapter.py` | Safiyye XoFTR wrapper yerel TorchScript sözleşmesine çevrildi |
| `localization/session_store.py` | Yusuf stateful VO gereksiniminden türetildi; algoritma taşınmadı |

Hiçbir eski projede dosya değiştirilmedi veya silinmedi.
