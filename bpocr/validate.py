"""End-to-end evaluation on all photos with ground truth."""

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2

from bpocr.config import Config
from bpocr.data import discover_photos, load_ground_truth, load_test_split, split_of
from bpocr.pipeline import recognize

BACKENDS = ("pt", "onnx", "int8")
SPLITS = ("all", "train", "test")
COLUMNS = ["stem", "subfolder", "split", "gt", "pred", "status", "error",
           "display_conf", "display_width_frac"]
BINS = ("<0.30", "0.30-0.60", ">0.60", "n/a")
STATUSES = ("ok", "fail", "error")


def backend_model_path(pt_path: Path, backend: str) -> Path:
    if backend == "pt":
        return pt_path
    suffix = "best.onnx" if backend == "onnx" else "best_int8.onnx"
    return pt_path.with_name(suffix)


def width_bin(frac: float | None) -> str:
    if frac is None:
        return "n/a"
    if frac < 0.30:
        return "<0.30"
    if frac <= 0.60:
        return "0.30-0.60"
    return ">0.60"


def _load_model(path: Path):
    from ultralytics import YOLO

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return YOLO(str(path), task="detect")


def _fmt(values: dict) -> str:
    return f"{values['sys']}/{values['dia']}/{values['pul']}"


def print_summary(rows: list[dict]) -> None:
    def line(label: str, subset: list[dict]) -> str:
        c = Counter(r["status"] for r in subset)
        return (f"  {label:<12} total={len(subset):<4} ok={c['ok']:<4} "
                f"fail={c['fail']:<4} error={c['error']:<4}")

    print("\nTotal")
    print(line("all", rows))
    print("\nBy display width (box width / image width)")
    for b in BINS:
        print(line(b, [r for r in rows if width_bin(_frac(r)) == b]))
    print("\nBy split")
    for s in ("train", "test"):
        print(line(s, [r for r in rows if r["split"] == s]))


def _frac(row: dict) -> float | None:
    v = row["display_width_frac"]
    return None if v == "" else float(v)


def validate(cfg: Config, backend: str, split: str) -> int:
    photos = [p for p in discover_photos(cfg.photos_dir) if p.json_path is not None]
    test_entries = load_test_split(cfg.test_split)
    items = [(p, split_of(p, test_entries)) for p in photos]
    if split != "all":
        items = [(p, s) for p, s in items if s == split]
    if not items:
        print(f"No photos with ground truth found in {cfg.photos_dir} (split={split}).")
        return 1

    display_model = _load_model(backend_model_path(cfg.display_model, backend))
    digit_model = _load_model(backend_model_path(cfg.digit_model, backend))
    print(f"Backend: {backend}, photos: {len(items)}, split: {split}")

    rows = []
    for photo, photo_split in items:
        gt = load_ground_truth(photo.json_path)
        row = {"stem": photo.stem, "subfolder": photo.subfolder, "split": photo_split,
               "gt": _fmt(gt), "pred": "", "status": "error", "error": "",
               "display_conf": "", "display_width_frac": ""}

        img = cv2.imread(str(photo.path))
        if img is None:
            row["error"] = "cannot read image"
        else:
            rec = recognize(img, display_model, digit_model, cfg)
            if rec.display_conf is not None:
                row["display_conf"] = f"{rec.display_conf:.4f}"
                row["display_width_frac"] = f"{rec.display_width_frac:.4f}"
            if rec.values is None:
                row["error"] = rec.error or ""
            else:
                row["pred"] = _fmt(rec.values)
                row["status"] = "ok" if rec.values == gt else "fail"
        rows.append(row)
        print(f"  {photo.stem:<20} {row['pred'] or '-':>15} {row['gt']:>15}  "
              f"{row['status'].upper()} {row['error']}")

    reports = cfg.work_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report = reports / f"validate-{backend}-{stamp}.csv"
    with open(report, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows)
    print(f"\nReport: {report}")
    return 0
