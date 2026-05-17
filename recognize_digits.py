"""
Розпізнавання цифр на cropped дисплеї через YOLO #2 + збірка у JSON.

Запуск (один файл):
  python recognize_digits.py cropped/20260516_044548.jpg

Запуск з кастомною моделлю:
  python recognize_digits.py cropped/20260516_044548.jpg runs/detect/digit_detector_v1/weights/best.pt

Вивід:
  {"sys": 125, "dia": 74, "pul": 73}
або у випадку помилки збірки:
  {"error": "got 2 rows, need 3", "raw_boxes": [...]}
"""

import sys
import json
from pathlib import Path

from ultralytics import YOLO


CONF_THRESHOLD = 0.25
DEFAULT_MODEL = "runs/detect/digit_detector_latest/weights/best.pt"
# Жорстке IoU для class-agnostic NMS — прибирає дублікати рамок різних класів,
# які YOLO залишила бо думає що це різні об'єкти (типова помилка на семисегменті).
AGGRESSIVE_NMS_IOU = 0.4


def class_agnostic_nms(boxes: list[dict], iou_threshold: float = AGGRESSIVE_NMS_IOU) -> list[dict]:
    """
    Дополнительний фільтр поверх YOLO-NMS: видаляє пари рамок з високим IoU
    незалежно від класу, лишає ту що з вищим confidence.
    Це лікує випадки коли YOLO видає `2` і `3` майже на одному й тому ж місці.
    """
    if not boxes:
        return []

    # Сортуємо за confidence спадно — найбільш впевнені залишаться першими
    sorted_boxes = sorted(boxes, key=lambda b: -b["conf"])
    kept = []

    def iou(a: dict, b: dict) -> float:
        ix1 = max(a["x1"], b["x1"])
        iy1 = max(a["y1"], b["y1"])
        ix2 = min(a["x2"], b["x2"])
        iy2 = min(a["y2"], b["y2"])
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        return inter / (area_a + area_b - inter)

    for box in sorted_boxes:
        # Якщо нова рамка надто схожа на якусь що вже залишили — пропустити
        if all(iou(box, k) < iou_threshold for k in kept):
            kept.append(box)

    return kept


def boxes_from_result(result) -> list[dict]:
    """Витягуємо рамки з YOLO-результату у звичайний формат."""
    boxes = result.boxes
    if len(boxes) == 0:
        return []
    cls_arr = boxes.cls.cpu().numpy().astype(int)
    conf_arr = boxes.conf.cpu().numpy()
    xyxy_arr = boxes.xyxy.cpu().numpy()
    out = []
    for cls, conf, (x1, y1, x2, y2) in zip(cls_arr, conf_arr, xyxy_arr):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h = y2 - y1
        out.append({"cls": int(cls), "conf": float(conf),
                    "cx": float(cx), "cy": float(cy),
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "h": float(h)})
    return out


def group_into_rows(boxes: list[dict]) -> list[list[dict]]:
    """
    Групує рамки у рівно 3 рядки за Y-координатою (зверху вниз).
    Використовує простий 1D k-means з k=3 (без зайвих залежностей).
    Усередині рядка сортує за X (зліва направо).

    Якщо рамок менше 3 — повертає менше рядків (буде помилка у виклику вище).
    """
    if not boxes:
        return []
    if len(boxes) < 3:
        # Мало рамок — просто розкладаємо як є по одній на рядок
        return [[b] for b in sorted(boxes, key=lambda b: b["cy"])]

    ys = [b["cy"] for b in boxes]
    y_min, y_max = min(ys), max(ys)
    if y_max == y_min:
        # Усі на одній висоті — поверни одним рядком
        return [sorted(boxes, key=lambda b: b["cx"])]

    # Ініціалізація центрів — три рівномірні точки між min і max Y
    centers = [
        y_min + (y_max - y_min) * 1 / 6,
        y_min + (y_max - y_min) * 3 / 6,
        y_min + (y_max - y_min) * 5 / 6,
    ]

    # Lloyd's algorithm — досить кількох ітерацій
    for _ in range(20):
        # Assign step
        clusters: list[list[dict]] = [[], [], []]
        for box in boxes:
            distances = [abs(box["cy"] - c) for c in centers]
            idx = distances.index(min(distances))
            clusters[idx].append(box)

        # Update step
        new_centers = []
        for i, cluster in enumerate(clusters):
            if cluster:
                new_centers.append(sum(b["cy"] for b in cluster) / len(cluster))
            else:
                new_centers.append(centers[i])

        if all(abs(a - b) < 1e-3 for a, b in zip(centers, new_centers)):
            break
        centers = new_centers

    # Сортуємо кластери за Y центром (зверху вниз), всередині — за X
    cluster_centers = list(zip(centers, clusters))
    cluster_centers.sort(key=lambda x: x[0])
    rows = [sorted(cluster, key=lambda b: b["cx"]) for _, cluster in cluster_centers]

    # Видаляємо порожні рядки (на випадок коли всі рамки потрапили в один кластер)
    rows = [r for r in rows if r]
    return rows


def assemble_number(row: list[dict]) -> int | None:
    """З рядка рамок збирає число (конкатенація класів)."""
    if not row:
        return None
    try:
        return int("".join(str(b["cls"]) for b in row))
    except ValueError:
        return None


def recognize(image_path: str, model_path: str = DEFAULT_MODEL) -> dict:
    """
    Розпізнає sys/dia/pul на cropped дисплеї.
    Повертає dict {sys, dia, pul} або {error, raw_boxes} у випадку помилки.
    """
    model = YOLO(model_path)
    results = model(image_path, conf=CONF_THRESHOLD, verbose=False)
    boxes = boxes_from_result(results[0])

    if not boxes:
        return {"error": "no digits detected"}

    # Class-agnostic NMS — прибирає дублікати рамок
    boxes = class_agnostic_nms(boxes)

    rows = group_into_rows(boxes)

    if len(rows) != 3:
        return {
            "error": f"got {len(rows)} rows, need 3",
            "raw_boxes": [{"cls": b["cls"], "cx": b["cx"], "cy": b["cy"]} for b in boxes],
        }

    sys_n = assemble_number(rows[0])
    dia_n = assemble_number(rows[1])
    pul_n = assemble_number(rows[2])

    if sys_n is None or dia_n is None or pul_n is None:
        return {"error": "could not assemble numbers", "rows": rows}

    return {"sys": sys_n, "dia": dia_n, "pul": pul_n}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    if not Path(image_path).exists():
        print(f"Файл не існує: {image_path}")
        sys.exit(1)

    result = recognize(image_path, model_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
