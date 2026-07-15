# Gerçek görevlerin eklenmesi

## Görev 1

1. `app/services/detection/interface.py` sözleşmesini uygulayan gerçek servis ekleyin.
2. `app/services/registry.py` içinde `DETECTION_ENABLED=true` durumunda bu servisi seçin.
3. Modeli `DETECTION_MODEL_PATH`, confidence ve IoU ayarlarıyla yükleyin.
4. Servis her kare için `list[DetectedObject]` döndürmelidir.

Değiştirilecek ana dosyalar: `detection/` altındaki yeni servis ve `registry.py`.

## Görev 2

1. `app/services/localization/interface.py` sözleşmesini uygulayın.
2. Verilen `LocalizationSessionState` içine modele özgü state yerleştirin.
3. Servisi her frame'de çalıştırın; GPS alanlarının `None` olabileceğini ele alın.
4. Yalnızca gerçek kestirim oluştuğunda `Translation`, aksi hâlde `None` döndürün.
5. `registry.py` içinde `LOCALIZATION_ENABLED=true` için gerçek servisi seçin.

Değiştirilecek ana dosyalar: `localization/` altındaki yeni servis ve `registry.py`.

## Görev 3

Model artifact sözleşmesi değişirse yalnızca ilgili adapter ve
`models/matching/README.md` güncellenmelidir. API ve session servisi aynı kalır.
