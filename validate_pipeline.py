"""
Повний пайплайн на всіх фото з оригінальної папки:
  1. YOLO #1 (display detector) → cropped дисплей
  2. YOLO #2 (digit detector) → рамки цифр
  3. Збірка → {sys, dia, pul}
  4. Порівняння з ground truth з .json
"""

import argparse
import sys
import json
import time
import shutil
from pathlib import Path
import cv2
from ultralytics import YOLO

# Імпортуємо логіку з recognize_digits
from recognize_digits import boxes_from_result, group_into_rows, assemble_number, class_agnostic_nms

# ---------- НАЛАШТУВАННЯ ----------
DISPLAY_MODEL_PT = "runs/detect/display_detector_latest/weights/best.pt"
DIGIT_MODEL_PT = "runs/detect/digit_detector_latest/weights/best.pt"
# ----------------------------------

DISPLAY_CONF = 0.05
DIGIT_CONF = 0.25
PADDING_RATIO = 0.03
TARGET_W = 400
TARGET_H = 480

def crop_display(img, display_model):
    """Знаходить дисплей через YOLO #1 і повертає cropped."""
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

    boxes = class_agnostic_nms(boxes)
    # Твоя логіка фільтрації з recognize_digits.py
    max_h = max(b["h"] for b in boxes)
    boxes = [b for b in boxes if b["h"] > max_h * 0.4]
    boxes = [b for b in boxes if b["cx"] > 150]

    rows = group_into_rows(boxes)
    if len(rows) != 3:
        return {"error": f"got {len(rows)} rows, need 3"}, boxes

    nums = [assemble_number(r) for r in rows]
    if any(n is None for n in nums):
        return {"error": "assemble failed"}, boxes

    return {"sys": nums[0], "dia": nums[1], "pul": nums[2]}, boxes

def main(orig_dir: str):
    orig = Path(orig_dir)

    display_model = YOLO(DISPLAY_MODEL_PT)
    digit_model = YOLO(DIGIT_MODEL_PT)

    pairs = [(jpg, jpg.with_suffix(".json")) for jpg in orig.glob("*.jpg") if jpg.with_suffix(".json").exists()]

    print(f"Знайдено {len(pairs)} фото з ground truth у {orig}")
    print("=" * 80)
    print(f"{'file':<30} {'predicted':>15} {'truth':>15} {'status'}")
    print("-" * 80)

    n_ok = 0
    n_fail = 0
    
    for jpg, json_path in pairs:
        with open(json_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        gt_str = f"{gt['sys']}/{gt['dia']}/{gt['pul']}"

        img = cv2.imread(str(jpg))
        cropped = crop_display(img, display_model)
        
        if cropped is None:
            print(f"{jpg.name:<30} {'(no display)':>15} {gt_str:>15}  ERROR")
            continue

        result, _ = recognize_on_cropped(cropped, digit_model)

        if "error" in result:
            print(f"{jpg.name:<30} {result['error']:>15} {gt_str:>15}  ERROR")
            continue

        pred_str = f"{result['sys']}/{result['dia']}/{result['pul']}"
        if result["sys"] == gt["sys"] and result["dia"] == gt["dia"] and result["pul"] == gt["pul"]:
            print(f"{jpg.name:<30} {pred_str:>15} {gt_str:>15}  OK")
            n_ok += 1
        else:
            print(f"{jpg.name:<30} {pred_str:>15} {gt_str:>15}  FAIL")
            n_fail += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_dir")
    args = parser.parse_args()
    main(args.img_dir)