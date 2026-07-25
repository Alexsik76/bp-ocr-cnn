"""
Перевіряє відповідність розмітки YOLO (labels2) до ground truth з JSON.

Для кожного фото з cropped/ читає:
  - <name>.txt   — розмітку (рамки з класами 0-9)
  - <original_dir>/<name>.json — ground truth {sys, dia, pul}

Сортує рамки за Y (зверху вниз → 3 рядки), у кожному рядку за X (зліва направо),
збирає числа з класів і звіряє з JSON.

Запуск:
  python verify_labels.py cropped labels/labels2 img

Параметри:
  argv[1] — папка з cropped (потрібна тільки для пошуку .jpg імен)
  argv[2] — папка з розміткою .txt
  argv[3] — папка з оригіналами + .json (ground truth)
"""

import json
import sys
from pathlib import Path


def read_label(txt_path: Path) -> list[dict]:
    """Читає YOLO-розмітку. Повертає список dict з cls, cx, cy, w, h."""
    boxes = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            boxes.append({
                "cls": int(parts[0]),
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            })
    return boxes


def group_into_rows(boxes: list[dict]) -> list[list[dict]]:
    """
    Групує рамки у рівно 3 рядки за Y-координатою через 1D k-means з k=3.
    Більш надійно ніж жадібний gap-detection.
    """
    if not boxes:
        return []
    if len(boxes) < 3:
        return [[b] for b in sorted(boxes, key=lambda b: b["cy"])]

    ys = [b["cy"] for b in boxes]
    y_min, y_max = min(ys), max(ys)
    if y_max == y_min:
        return [sorted(boxes, key=lambda b: b["cx"])]

    centers = [
        y_min + (y_max - y_min) * 1 / 6,
        y_min + (y_max - y_min) * 3 / 6,
        y_min + (y_max - y_min) * 5 / 6,
    ]

    for _ in range(20):
        clusters: list[list[dict]] = [[], [], []]
        for box in boxes:
            distances = [abs(box["cy"] - c) for c in centers]
            idx = distances.index(min(distances))
            clusters[idx].append(box)
        new_centers = []
        for i, cluster in enumerate(clusters):
            if cluster:
                new_centers.append(sum(b["cy"] for b in cluster) / len(cluster))
            else:
                new_centers.append(centers[i])
        if all(abs(a - b) < 1e-3 for a, b in zip(centers, new_centers, strict=False)):
            break
        centers = new_centers

    cluster_centers = list(zip(centers, clusters, strict=False))
    cluster_centers.sort(key=lambda x: x[0])
    rows = [sorted(cluster, key=lambda b: b["cx"]) for _, cluster in cluster_centers]
    rows = [r for r in rows if r]
    return rows


def row_to_number(row: list[dict]) -> int | None:
    """З рядка рамок збирає число — конкатенація класів."""
    if not row:
        return None
    digits = [str(b["cls"]) for b in row]
    try:
        return int("".join(digits))
    except ValueError:
        return None


def verify(cropped_dir: str, labels_dir: str, orig_dir: str):
    cropped = Path(cropped_dir)
    labels = Path(labels_dir)
    orig = Path(orig_dir)

    print(f"{'file':<32} {'labeled':>20} {'ground truth':>20}   status")
    print("=" * 90)

    n_ok = 0
    n_fail = 0
    n_skip = 0

    for jpg in sorted(cropped.glob("*.jpg")):
        txt = labels / (jpg.stem + ".txt")
        json_path = orig / (jpg.stem + ".json")

        if not txt.exists():
            print(f"{jpg.name:<32} {'(no label)':>20} {'-':>20}   SKIP")
            n_skip += 1
            continue
        if not json_path.exists():
            print(f"{jpg.name:<32} {'-':>20} {'(no json)':>20}   SKIP")
            n_skip += 1
            continue

        # Ground truth
        with open(json_path, encoding="utf-8") as f:
            gt = json.load(f)
        gt_str = f"{gt['sys']}/{gt['dia']}/{gt['pul']}"

        # Зібрати з розмітки
        boxes = read_label(txt)
        if not boxes:
            print(f"{jpg.name:<32} {'(empty)':>20} {gt_str:>20}   FAIL")
            n_fail += 1
            continue

        rows = group_into_rows(boxes)
        numbers = [row_to_number(r) for r in rows]

        # Має бути рівно 3 рядки
        if len(numbers) != 3:
            label_str = "/".join(str(n) for n in numbers)
            msg = f"FAIL (got {len(numbers)} rows, need 3)"
            print(f"{jpg.name:<32} {label_str:>20} {gt_str:>20}   {msg}")
            n_fail += 1
            continue

        sys_l, dia_l, pul_l = numbers
        label_str = f"{sys_l}/{dia_l}/{pul_l}"

        if (sys_l == gt["sys"] and dia_l == gt["dia"] and pul_l == gt["pul"]):
            print(f"{jpg.name:<32} {label_str:>20} {gt_str:>20}   OK")
            n_ok += 1
        else:
            # Детальніше про різницю
            diffs = []
            if sys_l != gt["sys"]:
                diffs.append(f"sys {sys_l}≠{gt['sys']}")
            if dia_l != gt["dia"]:
                diffs.append(f"dia {dia_l}≠{gt['dia']}")
            if pul_l != gt["pul"]:
                diffs.append(f"pul {pul_l}≠{gt['pul']}")
            print(f"{jpg.name:<32} {label_str:>20} {gt_str:>20}   FAIL  [{', '.join(diffs)}]")
            n_fail += 1

    print("=" * 90)
    print(f"OK:   {n_ok}")
    print(f"FAIL: {n_fail}")
    print(f"SKIP: {n_skip}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    verify(sys.argv[1], sys.argv[2], sys.argv[3])
