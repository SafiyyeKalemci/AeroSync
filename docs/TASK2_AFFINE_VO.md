# Görev 2 — Affine Visual Odometry çekirdeği

Bu belge birinci entegrasyon aşamasındaki görüntü düzlemi çekirdeğini açıklar.
GPS ölçeği ve dünya koordinat hizalaması sonraki aşamada eklenmiştir; ayrıntılar
`TASK2_CALIBRATION.md` içindedir.

## Yöntem

Önceki grayscale frame üzerinde Shi–Tomasi köşeleri seçilir. Noktalar pyramidal
Lucas–Kanade ile önce previous → current, sonra current → previous yönünde
izlenir. Forward-backward hatası eşiği aşan; NaN, sonsuz veya görüntü dışında
kalan track'ler atılır.

Geçerli track'lere kamera merkezi çevresindeki şu üç parametreli model uygulanır:

```text
du = -tx + yaw * (y - cy)
dv = -ty - yaw * (x - cx)
```

RANSAC, inlier kümesini bulur; ardından least-squares refinement yapılır.
Internal sonuç `delta_x_px`, `delta_y_px`, `delta_yaw_rad`, track/inlier sayısı,
inlier oranı, RMS residual, kalite bayrağı ve hata nedenini içerir. Bu değerler
yarışma koordinatı değildir.

## Session ve süreklilik

Her session ayrı `VisualOdometryState` taşır ve mevcut
`LocalizationSessionStore` per-session lock'u altında işlenir. State previous
frame/kimlik/index, video, şekil, modality, warmup, kümülatif piksel hareketi,
son motion sonucu, fingerprint, freeze bilgisi ve erişim zamanını saklar.

- İlk frame baseline olur.
- Aynı frame ID state'i ilerletmez.
- Out-of-order, büyük gap, eksik index, video/shape/modality değişimi baseline'ı
  sıfırlar.
- Aynı görüntünün farklı ID ile gelmesi freeze sayılır ve geçerli sıfır hareket
  olarak kabul edilmez.
- Decode hatası previous state'i değiştirmez.
- OpenCV worker state'e erişmez; state yalnız başarılı await sonrasında atomik
  commit edilir. Cancellation sonrası geç biten worker commit yapamaz.

GPS sağlıklı veya sağlıksız olsa da VO state güncellenir. GPS değerleri bu
aşamada ölçek ya da tahmin üretiminde kullanılmaz.

## Kamera modeli

RGB 1920x1080 preset'i merkezi config alanlarından gelir. JSON veya anahtar/değer
metin biçimindeki yerel kamera kalibrasyon dosyası desteklenir. Çözünürlük modelle
eşleşmiyorsa sessiz ölçekleme yapılmaz. Distorsiyon katsayıları verilmişse frame
feature çıkarımından önce undistort edilir.

## Sınırlamalar ve sonraki aşama

Üç parametreli model roll, pitch, perspektif, değişen yükseklik veya Z hareketini
temsil etmez. Piksel hareketi metrik konum değildir. Sonraki ayrı aşama, yalnız
kaliteli ve ardışık GPS ground-truth örnekleriyle scale ve kamera → dünya eksen
hizalamasını eklemelidir. DPVO ve termal pipeline bu aşamanın dışındadır.
