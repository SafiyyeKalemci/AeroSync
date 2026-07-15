# Görev 3 — RGB-RGB Coarse Matching

Bu aşama, yerel DINOv2 dense patch descriptor'ları kullanarak doğrulanmış RGB-RGB referans eşlemesi üretir. ALIKED, LightGlue, XoFTR, termal eşleme ve geçmiş frame sonucunu sürdürme bu akışın parçası değildir.

## Veri akışı

```text
Aktif referans descriptor cache'i ─┐
                                  ├─ cosine mutual NN
Frame için tek DINOv2 descriptor ─┘
→ similarity ve spatial filtreler
→ patch merkezlerini orijinal piksel koordinatlarına dönüştürme
→ USAC_MAGSAC homography
→ inlier ve RMS reprojection kontrolleri
→ referans köşelerini frame'e projekte etme
→ polygon ve görünürlük doğrulaması
→ clipped axis-aligned bbox
→ bileşen tabanlı confidence
→ MatchedReferenceObject
```

Aktif referans yoksa model çağrılmaz. Birden fazla aktif referansta referans descriptor'ları kendi cache'lerinden alınır; frame yalnız bir kez okunur, preprocess edilir ve tek DINOv2 forward çalıştırılır. Similarity ve geometri her referans için ayrı değerlendirilir. Farklı `object_id` sonuçları birbirini bastırmaz; çıktı `reference order`, ardından `object_id` düzenindedir.

## Mutual nearest-neighbour

Descriptor'lar yeniden L2-normalize edilir. Cosine similarity, reference descriptor blokları halinde hesaplanarak tam `[N_ref, N_frame]` matrisinin kalıcı olarak oluşturulması önlenir. Her reference patch'in en iyi frame patch'i ve her frame patch'in en iyi reference patch'i tutulur. Yalnız karşılıklı seçimler ve similarity eşiğini geçenler korunur; böylece iki tarafta da one-to-one correspondence elde edilir.

`MATCHING_COARSE_TOPK_PER_REFERENCE=1`, strict mutual-NN politikasını ifade eder. Daha yüksek değer yapılandırılabilse de bu aşamada nihai mutual seçim her reference için yine tek en iyi adayı tutar.

Correspondence'lar similarity'ye göre deterministik sıralanır, yakın lokal bölgelerdeki yoğun tekrarlar spatial dedup ile azaltılır ve maksimum correspondence sınırı uygulanır. Minimum sayı veya iki boyutlu spatial coverage sağlanmazsa homography çağrılmaz.

Patch koordinatı `(column + 0.5, row + 0.5)` merkezi üzerinden hesaplanır. Reference ve frame'in kendi `scale_x/scale_y` değerleriyle resized koordinattan orijinal piksel koordinatına ayrı ayrı dönülür.

## Homography kalite kapıları

Yalnız OpenCV `USAC_MAGSAC` kullanılır. En az dört correspondence gerekir. Sonuç için:

- Matris tam `3×3`, finite ve normalize edilebilir olmalıdır.
- Singular veya aşırı condition-number matris reddedilir.
- Inlier mask uzunluğu correspondence sayısıyla aynı olmalıdır.
- Minimum inlier sayısı ve oranı birlikte geçilmelidir.
- Inlier noktalarındaki RMS reprojection error yapılandırılmış üst sınırı aşmamalıdır.

Bir referansın OpenCV hatası, timeout'u veya geçersiz geometrisi diğer referansların değerlendirmesini durdurmaz.

## Polygon ve bbox doğrulaması

Referansın dört köşesi homography ile frame'e taşınır. Projected polygon finite, convex, pozitif alanlı ve self-intersection içermeyen bir dörtgen olmalıdır. Alan, minimum kenar, aspect ratio, karşılıklı kenar distortion ve frame alan oranı sınırları uygulanır.

Frame ile convex intersection alanı hesaplanır. Tamamen dışarıdaki veya görünür oranı düşük polygon reddedilir. Yeterli görünür alanı olan kısmi polygon kabul edilebilir.

Axis-aligned raw bbox polygon min/max koordinatlarından türetilir; ardından frame sınırlarına clip edilir. Minimum genişlik, yükseklik, alan ve maksimum frame alan oranı doğrulanır. Ters, boş, sıfır alanlı veya finite olmayan bbox hiçbir zaman response'a girmez.

## Confidence

Confidence sabit değildir ve kalibre edilmiş olasılık olduğu iddia edilmez. Beş normalize bileşenin ağırlıklı ortalamasıdır:

```text
confidence = (
  w_inlier      × inlier_ratio
  + w_similarity × normalized(mean_similarity, median_similarity)
  + w_reprojection × (1 - RMS / max_RMS)
  + w_visibility × visible_polygon_ratio
  + w_coverage   × normalized_spatial_coverage
) / sum(weights)
```

Sonuç `[0,1]` aralığına kırpılır. Finite olmayan veya `MATCHING_MIN_CONFIDENCE` altında kalan skor reddedilir.

## Timeout, reset ve hata izolasyonu

Her referans işi `MATCHING_COARSE_TIMEOUT_SECONDS` ve `MATCHING_REFERENCE_TIMEOUT_SECONDS` değerlerinin daha küçüğüyle sınırlandırılır. `to_thread` worker timeout sonrasında fiziksel olarak çalışmaya devam edebilir; ancak sonucu doğrudan response'a yazamaz. Session generation token her sonuç öncesi ve sonrasında doğrulanır. Reset veya referans değişikliği sonrası eski worker sonucu tüm frame çıktısından çıkarılır.

CUDA OOM, `MemoryError`, OpenCV hatası ve diğer referans-bazlı hatalar structured log üretir. Sahte bbox veya önceki frame'den taşınmış fallback sonucu oluşturulmaz.

## Neden LightGlue ve XoFTR yok?

Bu aşamanın amacı yalnız DINOv2 coarse RGB-RGB geometrisini izole biçimde doğrulamaktır. LightGlue/ALIKED refinement ve XoFTR cross-modal/termal eşleme ayrı artifact, eşik ve performans doğrulaması gerektirir; bu çağrı zincirine bağlanmamıştır.
