"""
Тренування YOLOv8 для детекції цифр (0-9) на cropped дисплеях.

Після тренування:
  - попередня модель runs/detect/digit_detector_latest/ → перейменовується на digit_detector_bak/
  - нова модель → digit_detector_latest/
  - якщо bak вже існував — затирається

Запуск:
  python train_yolo_digits.py
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

# ---------- НАЛАШТУВАННЯ ----------

DATA_YAML = "dataset_digits/data.yaml"
BASE_MODEL = "yolov8n.pt"
EPOCHS = 100
IMG_SIZE = 320
BATCH = 8
PATIENCE = 30

# Тимчасова назва для нового тренування — потім перейменовується.
TEMP_NAME = "_digit_detector_tmp"

# Постійні шляхи моделей.
LATEST_NAME = "digit_detector_latest"
BAK_NAME = "digit_detector_bak"

RUNS_DIR = Path("runs/detect")

# ----------------------------------


def rotate_models():
    """Ротує: latest → bak (попередній bak видаляється), tmp → latest."""
    latest = RUNS_DIR / LATEST_NAME
    bak = RUNS_DIR / BAK_NAME
    tmp = RUNS_DIR / TEMP_NAME

    if not tmp.exists():
        print(f"Тимчасова папка {tmp} не з'явилась — тренування не дало результату.")
        return

    # Старий bak → видалити
    if bak.exists():
        print(f"Видаляю старий {bak}")
        shutil.rmtree(bak)

    # latest → bak
    if latest.exists():
        print(f"Перейменовую {latest} → {bak}")
        latest.rename(bak)

    # tmp → latest
    print(f"Перейменовую {tmp} → {latest}")
    tmp.rename(latest)


def main():
    # Видаляємо тимчасову папку якщо лишилась з попереднього невдалого запуску
    tmp_path = RUNS_DIR / TEMP_NAME
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        name=TEMP_NAME,
        device="cpu",
        flipud=0.0,
        fliplr=0.0,
        degrees=0.0,
        translate=0.05,
        scale=0.1,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        mosaic=0.5,
    )

    rotate_models()

    final = RUNS_DIR / LATEST_NAME
    print("\n" + "=" * 60)
    print(f"Готово. Активна модель: {final / 'weights' / 'best.pt'}")
    print(f"Попередня версія: {RUNS_DIR / BAK_NAME / 'weights' / 'best.pt'}")
    print(f"Графіки: {final / 'results.png'}")


if __name__ == "__main__":
    main()
