# Görev 1: İniş uygunluğu

Bu aşama, YOLO'nun UAP ve UAİ kutularını iniş alanı; taşıt ve insan kutularını
engel olarak değerlendirir. Kaynak prototipte `IoU` adı kullanılsa da hesaplanan
değer birleşim alanına bölünmediği için gerçek IoU değildi. Yeni bileşen açıkça
kesişim alanını ve kesişimin iniş alanına oranını hesaplar.

## Karar akışı

Ham kutu finite ve sıralı değilse ya da görünür kutu çok küçükse sonuç
`NOT_APPLICABLE` olur. Ham UAP/UAİ kutusu yapılandırılmış kenar toleransından
daha fazla frame dışına çıkıyorsa `UNSUITABLE` olur. Aksi halde her taşıt ve
insan engeli için şu politika uygulanır:

```text
occupied = intersection_area >= MIN_INTERSECTION_PIXELS
           AND (
             intersection_over_landing_area >= OCCUPANCY_RATIO
             OR enabled(center_inside)
             OR enabled(bottom_center_inside)
           )
```

Bir engel politikayı sağlarsa alan `UNSUITABLE`, hiçbir engel sağlamazsa
`SUITABLE` olur. UAP ve UAİ kutuları birbirlerini otomatik olarak engellemez.
Geometri veya analiz hataları uygunluk uydurmaz; detection korunur,
`NOT_APPLICABLE` kullanılır ve yapılandırılmış uyarı yazılır.

## Ham ve kırpılmış kutu

`raw_bbox`, modelin koordinatlarını korur ve frame içinde tam görünürlük
kontrolünde kullanılır. `clipped_bbox`, gerçek decoded görüntünün sınırlarına
kırpılır; API çıktısı, alan ve kesişim hesabında kullanılır. Kod 1920x1080
çözünürlüğüne bağlı değildir.

## Yapılandırma

- `DETECTION_LANDING_ENABLED`
- `DETECTION_LANDING_EDGE_MARGIN_PX`
- `DETECTION_LANDING_EDGE_MARGIN_RATIO`
- `DETECTION_LANDING_MIN_INTERSECTION_PIXELS`
- `DETECTION_LANDING_OCCUPANCY_RATIO`
- `DETECTION_LANDING_USE_CENTER_CHECK`
- `DETECTION_LANDING_USE_BOTTOM_CENTER_CHECK`
- `DETECTION_LANDING_MIN_AREA_PIXELS`

Mutlak ve göreli kenar toleransından büyük olan uygulanır. Tüm sayısal değerler
servis kurulurken doğrulanır.

## Sınırlamalar

Bounding box yaklaşımı nesnenin gerçek ayak izini ve iniş alanının perspektifini
yaklaşık temsil eder. Dönük veya düzensiz bölgelerde yanlış pozitif/negatif karar
oluşabilir. Gelecekte aynı karar arayüzü segmentation maskesi veya polygon
geometrisiyle geliştirilebilir.
