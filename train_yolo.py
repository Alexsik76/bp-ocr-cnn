"""
Тренування YOLOv8-nano для детекції LCD-дисплея тонометра.

Запуск:
  python train_yolo.py

Налаштування зверху файлу — змінюй як треба.

Результат:
  runs/detect/train/weights/best.pt   — найкраща модель
  runs/detect/train/weights/last.pt   — остання epoch
  runs/detect/train/results.png       — графіки тренування
  runs/detect/train/val_batch*.jpg    — приклади детекції на val
"""

from ultralytics import YOLO

# ---------- НАЛАШТУВАННЯ ----------

# Шлях до конфігу датасету (створений prepare_dataset.py)
DATA_YAML = "dataset/data.yaml"

# Базова модель. yolov8n = nano (~6 МБ), найшвидша і найменша.
# Альтернативи якщо точність недостатня: yolov8s (~22 МБ), yolov8m (~52 МБ)
BASE_MODEL = "yolov8n.pt"

# Кількість епох. На 34 фото 100 епох цілком достатньо (overfit неминучий
# з таким датасетом, але YOLO має built-in зупинку якщо метрика не росте).
EPOCHS = 100

# Розмір вхідного зображення. 640 — стандарт.
# Для нашої задачі (один великий об'єкт) можна 416 — швидше, без втрати точності.
IMG_SIZE = 640

# Batch size. На CPU великий не вийде. 4-8 нормально.
BATCH = 8

# Patience — скільки епох чекати покращення метрики перед ранньою зупинкою.
PATIENCE = 30

# Назва запуску — створиться папка runs/detect/<NAME>/
NAME = "display_detector_v1"

# ----------------------------------


def main():
    model = YOLO(BASE_MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        name=NAME,
        # device=0 для GPU, "cpu" для CPU. Auto-detect зазвичай добре працює.
        device="cpu",
        # Невелика аугментація — поможе з малим датасетом
        hsv_h=0.015,    # відтінок
        hsv_s=0.7,      # насиченість
        hsv_v=0.4,      # яскравість
        degrees=10,     # поворот (важливо для нашого кейсу — кутова зйомка)
        translate=0.1,
        scale=0.3,
        flipud=0.0,     # не перевертати догори ногами — дисплей завжди прямо
        fliplr=0.0,     # і не дзеркалити — рядки sys/dia/pul мають порядок
        mosaic=0.5,     # mozaika допомагає з малими датасетами
    )

    print("\n" + "=" * 60)
    print(f"Тренування завершено. Найкраща модель:")
    print(f"  runs/detect/{NAME}/weights/best.pt")
    print(f"Графіки: runs/detect/{NAME}/results.png")
    print(f"Приклади детекції на val: runs/detect/{NAME}/val_batch*.jpg")


if __name__ == "__main__":
    main()
