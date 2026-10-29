"""Verifies YOLO digit labels against the ground truth json of the original photos."""

from pathlib import Path

from bpocr.config import Config
from bpocr.data import discover_photos, find_images, load_ground_truth
from bpocr.pipeline import assemble_number, group_into_rows


def read_label(txt_path: Path) -> list[dict]:
    """Reads a YOLO label file into dicts with cls, cx, cy, w, h."""
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


def verify(cfg: Config, cropped_dir: Path, labels_dir: Path) -> int:
    json_by_stem = {p.stem: p.json_path for p in discover_photos(cfg.photos_dir) if p.json_path}
    images = find_images(cropped_dir)
    if not images:
        print(f"No images found in {cropped_dir}")
        return 1

    print(f"{'file':<32} {'labeled':>20} {'ground truth':>20}   status")
    print("=" * 90)

    n_ok = n_fail = n_skip = 0

    for img in images:
        txt = labels_dir / (img.stem + ".txt")
        json_path = json_by_stem.get(img.stem)

        if not txt.exists():
            print(f"{img.name:<32} {'(no label)':>20} {'-':>20}   SKIP")
            n_skip += 1
            continue
        if json_path is None:
            print(f"{img.name:<32} {'-':>20} {'(no json)':>20}   SKIP")
            n_skip += 1
            continue

        gt = load_ground_truth(json_path)
        gt_str = f"{gt['sys']}/{gt['dia']}/{gt['pul']}"

        boxes = read_label(txt)
        if not boxes:
            print(f"{img.name:<32} {'(empty)':>20} {gt_str:>20}   FAIL")
            n_fail += 1
            continue

        numbers = [assemble_number(r) for r in group_into_rows(boxes)]

        if len(numbers) != 3:
            label_str = "/".join(str(n) for n in numbers)
            msg = f"FAIL (got {len(numbers)} rows, need 3)"
            print(f"{img.name:<32} {label_str:>20} {gt_str:>20}   {msg}")
            n_fail += 1
            continue

        sys_l, dia_l, pul_l = numbers
        label_str = f"{sys_l}/{dia_l}/{pul_l}"

        if (sys_l, dia_l, pul_l) == (gt["sys"], gt["dia"], gt["pul"]):
            print(f"{img.name:<32} {label_str:>20} {gt_str:>20}   OK")
            n_ok += 1
        else:
            diffs = [
                f"{name} {got}≠{gt[name]}"
                for name, got in (("sys", sys_l), ("dia", dia_l), ("pul", pul_l))
                if got != gt[name]
            ]
            print(f"{img.name:<32} {label_str:>20} {gt_str:>20}   FAIL  [{', '.join(diffs)}]")
            n_fail += 1

    print("=" * 90)
    print(f"OK:   {n_ok}")
    print(f"FAIL: {n_fail}")
    print(f"SKIP: {n_skip}")
    return 0
