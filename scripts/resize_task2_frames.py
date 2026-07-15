from pathlib import Path
import cv2

# Kaynak ve hedef klasör
INPUT_DIR = Path(r"C:\Users\HP\Desktop\den\frames")
OUTPUT_DIR = Path(r"C:\Users\HP\Desktop\den\frames_1080p")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".webp"}

images = [
    p for p in INPUT_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
]

images.sort()

print(f"Bulunan görüntü sayısı: {len(images)}")

for image_path in images:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"OKUNAMADI: {image_path.name}")
        continue

    original_height, original_width = image.shape[:2]

    # TEKNOFEST RGB 1080p frame çözünürlüğü
    resized = cv2.resize(
        image,
        (1920, 1080),
        interpolation=cv2.INTER_AREA
    )

    output_path = OUTPUT_DIR / image_path.name

    success = cv2.imwrite(str(output_path), resized)

    if success:
        print(
            f"OK: {image_path.name} "
            f"{original_width}x{original_height} -> 1920x1080"
        )
    else:
        print(f"KAYDEDİLEMEDİ: {image_path.name}")

print()
print("İşlem tamamlandı.")
print(f"Çıktı klasörü: {OUTPUT_DIR}")