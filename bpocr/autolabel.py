"""Automatic labels: display boxes (manual or predicted), digit boxes verified by ground truth."""

import csv
import shutil
from pathlib import Path

import cv2

from bpocr.config import Config
from bpocr.data import discover_photos, load_ground_truth
from bpocr.gemini_display import gemini_dir
from bpocr.manual import manual_crops_dir, manual_digit_labels_dir
from bpocr.pipeline import best_display_box, crop_display, read_digits_with_boxes

REVIEW_COLUMNS = ["stem", "subfolder", "reason", "predicted", "expected"]
MANUAL_CHECK_COLUMNS = ["stem", "predicted", "expected", "match"]


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


def clear_files(directory: Path) -> None:
    for f in directory.iterdir():
        if f.is_file():
            f.unlink()


def place_manual_digits(label: Path, crop: Path, digits_dir: Path, crops_dir: Path) -> bool:
    """Copies a manual digit label and its crop unchanged; False if either is missing."""
    if not (label.is_file() and crop.is_file()):
        return False
    shutil.copy2(label, digits_dir / label.name)
    shutil.copy2(crop, crops_dir / crop.name)
    return True


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


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
    # Drop outputs of previous runs so that a photo never keeps a stale automatic label
    clear_files(digits_dir)
    clear_files(crops_dir)

    manual_labels = manual_digit_labels_dir(cfg)
    manual_crops = manual_crops_dir(cfg)
    if not manual_crops.is_dir():
        print("work/manual/crops is missing - run 'bpocr restore-manual' first; "
              "continuing with automatic digit labels only")

    review = []
    manual_check = []
    n_manual = n_auto = n_gemini = n_manual_digits = n_auto_digits = 0

    for photo in photos:
        stem = photo.stem
        expected = load_ground_truth(photo.json_path)
        (display_dir / f"{stem}.txt").unlink(missing_ok=True)

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
            if found is not None:
                box = found[0]
                n_auto += 1
            else:
                # Fallback: a box suggested by Gemini (bpocr gemini-display)
                box = read_manual_box(gemini_dir(cfg) / f"{stem}.txt", w, h)
                if box is None:
                    add_review("display-not-found", "")
                    continue
                n_gemini += 1

        (display_dir / f"{stem}.txt").write_text(
            to_yolo_line(0, *box, w, h) + "\n", encoding="utf-8")

        # Manual digit labels always win over automatic ones
        if place_manual_digits(manual_labels / f"{stem}.txt", manual_crops / f"{stem}.jpg",
                               digits_dir, crops_dir):
            n_manual_digits += 1
            values, error, _ = read_digits_with_boxes(
                cv2.imread(str(crops_dir / f"{stem}.jpg")), digit_model, cfg)
            manual_check.append({"stem": stem, "predicted": _fmt(values) if values else error,
                                 "expected": _fmt(expected), "match": values == expected})
            continue

        crop = crop_display(img, box, cfg)
        values, error, boxes = read_digits_with_boxes(crop, digit_model, cfg)
        if values is None:
            add_review("mismatch", error or "")
        elif values != expected:
            add_review("mismatch", _fmt(values))
        else:
            cv2.imwrite(str(crops_dir / f"{stem}.jpg"), crop)
            ch, cw = crop.shape[:2]
            lines = [to_yolo_line(b["cls"], b["x1"], b["y1"], b["x2"], b["y2"], cw, ch)
                     for b in boxes]
            (digits_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_auto_digits += 1

    review_path = cfg.work_dir / "review.csv"
    write_csv(review_path, REVIEW_COLUMNS, review)
    write_csv(cfg.work_dir / "manual-check.csv", MANUAL_CHECK_COLUMNS, manual_check)

    print(f"Photos with ground truth: {len(photos)}")
    print(f"Display labels: manual={n_manual} auto={n_auto} gemini={n_gemini}")
    print(f"Digit labels: manual={n_manual_digits} auto-accepted={n_auto_digits}")
    print(f"Review: {len(review)} (see {review_path})")
    return 0
