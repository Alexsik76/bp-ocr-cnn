"""
Тренування YOLOv8-nano для детекції LCD-дисплея тонометра.
Тепер з автоматичним бекапом старої моделі.
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

# ---------- НАЛАШТУВАННЯ ----------
DATA_YAML = "dataset/data.yaml"
BASE_MODEL = "yolov8n.pt"
EPOCHS = 100
IMG_SIZE = 640
BATCH = 8
PATIENCE = 30
# Назва папки, яку ми вважаємо "останньою/актуальною"
NAME = "display_detector_latest"
# ----------------------------------

def manage_model_backups(name):
    """Перейменовує існуючу папку в _bak перед новим тренуванням."""
    runs_dir = Path("runs/detect")
    target_dir = runs_dir / name
    bak_dir = runs_dir / f"{name}_bak"

    if target_dir.exists():
        if bak_dir.exists():
            shutil.rmtree(bak_dir)
        target_dir.rename(bak_dir)
        print(f"Попередня модель переміщена в: {bak_dir}")

def main():
    # 1. Готуємо місце для нової моделі
    manage_model_backups(NAME)
    
    model = YOLO(BASE_MODEL)

    # 2. Запуск тренування
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        name=NAME,
        device="cpu",
        hsv_h=0.1, hsv_s=0.9, hsv_v=0.9, bgr=0.5,
        degrees=10, translate=0.1, scale=0.3,
        flipud=0.0, fliplr=0.0, mosaic=0.5
    )

    print(f"\nГотово. Активна модель: runs/detect/{NAME}/weights/best.pt")

if __name__ == "__main__":
    main()