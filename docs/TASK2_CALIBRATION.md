# Görev 2 — GPS ground-truth kalibrasyonu

Bu aşama AffineVO'nun piksel hareketini sağlıklı sunucu ground-truth X/Y
adımlarıyla eşleştirir. İlk 450 frame yalnız beklenen kalibrasyon penceresi için
diagnostic sınırdır; örnek kabulünün asıl koşulu GPS health, finite koordinatlar,
ardışık frame ve kaliteli VO ölçümüdür. Pencere sonunda kaliteli fit yoksa kritik
uyarı yazılır, sahte dönüşüm kullanılmaz.

## Step eşleştirmesi ve trajectory

Ardışık iki sağlıklı frame arasında yaw ile başlangıç kamera eksenine döndürülmüş
kamera adımı ve aynı frame çiftinin ground-truth GPS X/Y adımı oluşturulur.
Duplicate, freeze, gap, out-of-order, reset, küçük hareket veya kalitesiz VO
örnek üretmez. Örnekler session bazlıdır ve yapılandırılmış üst sınırla tutulur.

## Rotation, scale ve kalite

Kamera adımları GPS adımlarına 2D Kabsch/Orthogonal Procrustes ile hizalanır.
Reflection kabulü açık config politikasıdır ve determinant kontrol edilir. Scale,
tek toplam yol oranından değil adım norm oranlarının median değerinden gelir;
MAD tabanlı filtre scale outlier'larını ayırır.

Kalibrasyon yalnız şu kapıları geçerse `ready` olur:

- minimum örnek ve inlier sayısı/oranı;
- finite ve config sınırları içinde pozitif scale;
- minimum yön çeşitliliği;
- reflection politikası;
- maksimum residual RMS.

Sonuç örnek/inlier sayısı, 2x2 rotation, scale, RMS, scale median/MAD, motion
span, directional diversity ve failure reason taşır.

## GPS geçişleri ve çıktı

GPS sağlıklıyken AffineVO çalışmaya devam eder ve sunucudan gelen finite X/Y/Z
değeri ground truth olarak aynen döndürülür. GPS 1→0 geçişinde hemen önceki
geçerli GPS anchor, o frame'in VO adımından önceki kamera pozu camera anchor olur
ve kaliteli kalibrasyon snapshot'ı dondurulur. Her sağlıksız frame'in geçerli VO
hareketi anchor farkından X/Y'ye çevrilir. Kalibrasyon, anchor, continuity veya
motion kalitesi eksikse sonuç `None` olur; stale veya sıfır koordinat üretilmez.

GPS 0→1 olduğunda ground truth gecikmeden döndürülür, anchor yenilenir ve recovery
loglanır. Kalibrasyon geçmişi korunur; yapılandırılmış sağlıklı frame sayısından
sonra frozen durum kaldırılır.

## Z politikası ve sınırlamalar

AffineVO gerçek Z hareketi ölçmez. Varsayılan `hold_last_valid_z`, GPS kaybında
anchor'ın son gerçek Z değerini taşır; bu bir Z tahmini değildir. Şema nullable Z
desteklemediği için `return_none_if_schema_allows` seçeneği tüm translation'ı
güvenli biçimde `None` yapar. `zero_delta_from_anchor` da açıkça sıfır Z delta
politikasıdır.

2D model irtifa değişimi, roll, pitch, perspektif ve düzlemsel olmayan sahnelerde
drift üretebilir. DPVO, termal pipeline, Essential Matrix ve full SE(3) bu aşamaya
dahil edilmemiştir.
