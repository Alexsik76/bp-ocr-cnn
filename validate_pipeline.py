"""
Повний пайплайн на всіх фото з оригінальної папки:
  1. YOLO #1 (display detector) → cropped дисплей
  2. YOLO #2 (digit detector) → рамки цифр
  3. Збірка → {sys, dia, pul}
  4. Порівняння з ground truth з .json

Бере КОЖНЕ фото з папки img/, для якого є парний .json.
Чи розмічене воно вручну (.txt в labels2) — не має значення для пайплайну.

Запуск:
  python validate_pipeline.py img
"""

import sys
import json
import time
from pathlib import Path

from ultralytics import YOLO

# Імпортуємо логіку з recognize_digits
from recognize_digits import boxes_from_result, group_into_rows, assemble_number, class_agnostic_nms


DISPLAY_MODEL = "runs/detect/display_detector_v1/weights/best.pt"
DIGIT_MODEL = "runs/detect/digit_detector_latest/weights/best.pt"

# Порог впевненості для YOLO #1 — низький, бо метрики train.val показали 0.06-0.11
DISPLAY_CONF = 0.05
# Для YOLO #2 — стандартний
DIGIT_CONF = 0.25

# Padding навколо bbox дисплея (як в infer_yolo.py)
PADDING_RATIO = 0.03

# Цільовий розмір cropped (як було при тренуванні YOLO #2)
TARGET_W = 400
TARGET_H = 480


def crop_display(img, display_model):
    """Знаходить дисплей через YOLO #1 і повертає cropped."""
    import cv2

    results = display_model(img, conf=DISPLAY_CONF, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None

    confs = boxes.conf.cpu().numpy()
    best_idx = int(confs.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * PADDING_RATIO)
    pad_y = int(bh * PADDING_RATIO)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


def recognize_on_cropped(cropped, digit_model):
    """YOLO #2 на cropped + збірка JSON."""
    results = digit_model(cropped, conf=DIGIT_CONF, verbose=False)
    boxes = boxes_from_result(results[0])
    if not boxes:
        return {"error": "no digits"}, boxes

    # Class-agnostic NMS
    boxes = class_agnostic_nms(boxes)

    rows = group_into_rows(boxes)
    if len(rows) != 3:
        return {"error": f"got {len(rows)} rows, need 3"}, boxes

    nums = [assemble_number(r) for r in rows]
    if any(n is None for n in nums):
        return {"error": "assemble failed"}, boxes

    return {"sys": nums[0], "dia": nums[1], "pul": nums[2]}, boxes


def main(orig_dir: str):
    import cv2
    orig = Path(orig_dir)

    display_model = YOLO(DISPLAY_MODEL)
    digit_model = YOLO(DIGIT_MODEL)

    # Знаходимо всі фото з ground truth (.json поряд)
    pairs = []
    for jpg in sorted(orig.glob("*.jpg")):
        json_path = jpg.with_suffix(".json")
        if json_path.exists():
            pairs.append((jpg, json_path))

    print(f"Знайдено {len(pairs)} фото з ground truth у {orig}")
    print(f"Моделі: {DISPLAY_MODEL}")
    print(f"        {DIGIT_MODEL}")
    print("=" * 80)
    print(f"{'file':<30} {'predicted':>15} {'truth':>15} {'time':>7}  status")
    print("-" * 80)

    n_ok = 0
    n_fail = 0
    n_error = 0
    failed_files = []
    times = []

    for jpg, json_path in pairs:
        with open(json_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        gt_str = f"{gt['sys']}/{gt['dia']}/{gt['pul']}"

        img = cv2.imread(str(jpg))
        if img is None:
            print(f"{jpg.name:<30} {'(read failed)':>15} {gt_str:>15} {'-':>7}  ERROR")
            n_error += 1
            continue

        start = time.time()
        cropped = crop_display(img, display_model)
        if cropped is None:
            elapsed = time.time() - start
            print(f"{jpg.name:<30} {'(no display)':>15} {gt_str:>15} {elapsed:>6.2f}s  ERROR")
            n_error += 1
            continue

        result, _ = recognize_on_cropped(cropped, digit_model)
        elapsed = time.time() - start
        times.append(elapsed)

        if "error" in result:
            err_short = result["error"][:13]
            print(f"{jpg.name:<30} {err_short:>15} {gt_str:>15} {elapsed:>6.2f}s  ERROR")
            n_error += 1
            failed_files.append(jpg.name)
            continue

        pred_str = f"{result['sys']}/{result['dia']}/{result['pul']}"
        if (result["sys"] == gt["sys"] and
            result["dia"] == gt["dia"] and
            result["pul"] == gt["pul"]):
            print(f"{jpg.name:<30} {pred_str:>15} {gt_str:>15} {elapsed:>6.2f}s  OK")
            n_ok += 1
        else:
            diffs = []
            if result["sys"] != gt["sys"]:
                diffs.append(f"sys")
            if result["dia"] != gt["dia"]:
                diffs.append(f"dia")
            if result["pul"] != gt["pul"]:
                diffs.append(f"pul")
            print(f"{jpg.name:<30} {pred_str:>15} {gt_str:>15} {elapsed:>6.2f}s  FAIL [{','.join(diffs)}]")
            n_fail += 1
            failed_files.append(jpg.name)

    print("=" * 80)
    total = len(pairs)
    print(f"OK:    {n_ok}/{total}  ({100 * n_ok / total:.1f}%)")
    print(f"FAIL:  {n_fail}/{total}  (помилка хоча б в одному числі)")
    print(f"ERROR: {n_error}/{total}  (пайплайн зламався)")
    if times:
        avg = sum(times) / len(times)
        print(f"Середній час: {avg:.2f}s  (min {min(times):.2f}, max {max(times):.2f})")
    if failed_files:
        print(f"\nПомилкові файли: {', '.join(failed_files)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
