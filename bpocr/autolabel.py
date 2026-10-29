"""Automatic labels: display boxes (manual or predicted), digit boxes verified by ground truth."""

import csv
from pathlib import Path

import cv2

from bpocr.config import Config
from bpocr.data import discover_photos, load_ground_truth
from bpocr.pipeline import best_display_box, crop_display, read_digits_with_boxes

REVIEW_COLUMNS = ["stem", "subfolder", "reason", "predicted", "expected"]


def to_yolo_line(cls: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    """Pixel corner box -> 'cls cx cy w h' normalised to an image of size w x h."""
    return (f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")


def from_yolo_line(line: str, w: int, h: int) -> tuple[int, float, float, float, float]:
    """'cls cx cy w h' -> (cls, x1, y1, x2, y2) in pixels of an image of size w x h."""
    parts = line.split()
    cls = int(parts[0])
    cx, cy, bw, bh = (float(v) for v in parts[1:5])
    return (cls, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)


def read_manual_box(path: Path, w: int, h: int) -> tuple[int, int, int, int] | None:
    """First box of a manual display label as pixel coordinates, or None if absent/empty."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if len(line.split()) >= 5:
            _, x1, y1, x2, y2 = from_yolo_line(line, w, h)
            return round(x1), round(y1), round(x2), round(y2)
    return None


def _fmt(values: dict) -> str:
    return f"{values['sys']}/{values['dia']}/{values['pul']}"


def autolabel(cfg: Config) -> int:
    from ultralytics import YOLO

    photos = [p for p in discover_photos(cfg.photos_dir) if p.json_path is not None]
    if not photos:
        print(f"No photos with ground truth found in {cfg.photos_dir}")
        return 1

    display_model = YOLO(str(cfg.display_model), task="detect")
    digit_model = YOLO(str(cfg.digit_model), task="detect")

    display_dir = cfg.work_dir / "labels" / "display"
    digits_dir = cfg.work_dir / "labels" / "digits"
    crops_dir = cfg.work_dir / "crops"
    for d in (display_dir, digits_dir, crops_dir):
        d.mkdir(parents=True, exist_ok=True)

    review = []
    n_manual = n_auto = n_digits = 0

    for photo in photos:
        stem = photo.stem
        expected = load_ground_truth(photo.json_path)
        # Drop outputs of previous runs so that a changed outcome leaves no stale file
        for stale in (display_dir / f"{stem}.txt", digits_dir / f"{stem}.txt",
                      crops_dir / f"{stem}.jpg"):
            stale.unlink(missing_ok=True)

        def add_review(reason: str, predicted: str, photo=photo, expected=expected):
            review.append({"stem": photo.stem, "subfolder": photo.subfolder, "reason": reason,
                           "predicted": predicted, "expected": _fmt(expected)})

        img = cv2.imread(str(photo.path))
        if img is None:
            add_review("unreadable-image", "")
            continue
        h, w = img.shape[:2]

        box = read_manual_box(cfg.manual_display_labels / f"{stem}.txt", w, h)
        if box is not None:
            n_manual += 1
        else:
            found = best_display_box(display_model(img, conf=cfg.display_conf, verbose=False)[0])
            if found is None:
                add_review("display-not-found", "")
                continue
            box = found[0]
            n_auto += 1

        (display_dir / f"{stem}.txt").write_text(
            to_yolo_line(0, *box, w, h) + "\n", encoding="utf-8")

        crop = crop_display(img, box, cfg)
        cv2.imwrite(str(crops_dir / f"{stem}.jpg"), crop)

        values, error, boxes = read_digits_with_boxes(crop, digit_model, cfg)
        if values is None:
            add_review("mismatch", error or "")
        elif values != expected:
            add_review("mismatch", _fmt(values))
        else:
            ch, cw = crop.shape[:2]
            lines = [to_yolo_line(b["cls"], b["x1"], b["y1"], b["x2"], b["y2"], cw, ch)
                     for b in boxes]
            (digits_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_digits += 1

    review_path = cfg.work_dir / "review.csv"
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(review)

    print(f"Photos with ground truth: {len(photos)}")
    print(f"Display labels: manual={n_manual} auto={n_auto}")
    print(f"Digit labels accepted: {n_digits}")
    print(f"Review: {len(review)} (see {review_path})")
    return 0
