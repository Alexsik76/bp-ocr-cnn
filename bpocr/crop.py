"""Crops displays out of photos and writes debug overlays."""

from pathlib import Path

import cv2

from bpocr.config import Config
from bpocr.data import discover_photos, find_images
from bpocr.pipeline import best_display_box, crop_display, padded_box


def crop_photos(cfg: Config, input_dir: Path | None, out_dir: Path | None) -> int:
    from ultralytics import YOLO

    if input_dir is None:
        images = [p.path for p in discover_photos(cfg.photos_dir)]
        source = cfg.photos_dir
    else:
        images = find_images(input_dir)
        source = input_dir
    if not images:
        print(f"No images found in {source}")
        return 1

    out_dir = out_dir or cfg.work_dir
    out_crop = out_dir / "cropped"
    out_debug = out_dir / "debug_pred"
    out_crop.mkdir(parents=True, exist_ok=True)
    out_debug.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(cfg.display_model))
    print(f"Model: {cfg.display_model}")
    print(f"Found {len(images)} photos in {source}")
    print(f"Cropped -> {out_crop}")
    print(f"Debug -> {out_debug}")
    print("=" * 60)

    found = 0
    failed = []
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [SKIP] {path.name}")
            continue

        best = best_display_box(model(img, conf=cfg.display_conf, verbose=False)[0])
        if best is None:
            print(f"  [FAIL] {path.name}: display not found")
            failed.append(path.name)
            continue
        box, conf = best

        cv2.imwrite(str(out_crop / f"{path.stem}.jpg"), crop_display(img, box, cfg))

        x1, y1, x2, y2 = padded_box(box, img.shape[1], img.shape[0], cfg.padding_ratio)
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(overlay, f"display {conf:.2f}", (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imwrite(str(out_debug / f"{path.stem}_pred.jpg"), overlay)

        found += 1
        print(f"  [OK]   {path.name}  conf={conf:.3f}")

    print("=" * 60)
    print(f"Success: {found}/{len(images)} ({100 * found / len(images):.0f}%)")
    if failed:
        print(f"Not found: {', '.join(failed)}")
    return 0
