"""
Інференс натренованої YOLO-моделі на папці з фото.

Для кожного фото знаходить дисплей і зберігає:
  - cropped/<name>.jpg       — обрізану область дисплея
  - debug/<name>_pred.jpg    — оригінал з намальованою рамкою (для перевірки)

Запуск:
  python infer_yolo.py img runs/detect/display_detector_v1/weights/best.pt

Або зі значеннями за замовчуванням:
  python infer_yolo.py
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


# ---------- НАЛАШТУВАННЯ ----------

# Розмір вихідних обрізаних дисплеїв (буде ресайз)
TARGET_W = 400
TARGET_H = 480

# Поріг впевненості — нижче цього детектор ігнорує
CONF_THRESHOLD = 0.05

# Невеликий padding навколо bbox — щоб гарантовано не обрізати цифри
# по краях. Виражений у відсотках від розміру bbox.
PADDING_RATIO = 0.03

# ----------------------------------


def main(input_dir: str, model_path: str):
    in_path = Path(input_dir)
    out_crop = in_path.parent / "cropped"
    out_debug = in_path.parent / "debug_pred"
    out_crop.mkdir(exist_ok=True)
    out_debug.mkdir(exist_ok=True)

    model = YOLO(model_path)
    print(f"Модель: {model_path}")

    jpgs = sorted(in_path.glob("*.jpg"))
    print(f"Знайдено {len(jpgs)} фото в {in_path}")
    print(f"Cropped → {out_crop}")
    print(f"Debug → {out_debug}")
    print("=" * 60)

    found = 0
    failed = []

    for jpg in jpgs:
        img = cv2.imread(str(jpg))
        if img is None:
            print(f"  [SKIP] {jpg.name}")
            continue

        # Інференс (з діагностикою)
        results_default = model(str(jpg), verbose=False)
        results = model(str(jpg), conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            # Покажемо що модель взагалі знайшла без фільтра
            all_confs = results_default[0].boxes.conf.cpu().numpy() if len(results_default[0].boxes) else []
            print(f"  [FAIL] {jpg.name}: дисплей не знайдено. Всі confs: {all_confs.tolist() if len(all_confs) else 'пусто'}")
            failed.append(jpg.name)
            continue

        # Якщо рамок кілька — беремо ту з найвищою впевненістю
        confs = boxes.conf.cpu().numpy()
        best_idx = int(confs.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        conf = float(confs[best_idx])

        # Padding
        h, w = img.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * PADDING_RATIO)
        pad_y = int(bh * PADDING_RATIO)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        # Crop і resize до цільового розміру
        crop = img[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_crop / jpg.name), crop_resized)

        # Debug overlay з рамкою + впевненістю
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(overlay, f"display {conf:.2f}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imwrite(str(out_debug / f"{jpg.stem}_pred.jpg"), overlay)

        found += 1
        print(f"  [OK]   {jpg.name}  conf={conf:.3f}")

    print("=" * 60)
    print(f"Успіх: {found}/{len(jpgs)} ({100 * found / len(jpgs):.0f}%)")
    if failed:
        print(f"Не знайшов: {', '.join(failed)}")


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "img"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "runs/detect/display_detector_v1/weights/best.pt"
    main(input_dir, model_path)