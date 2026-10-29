"""Two-stage recognition: display detector -> crop -> digit detector -> sys/dia/pul."""

from dataclasses import dataclass

import cv2

from bpocr.config import Config


@dataclass
class Recognition:
    values: dict | None = None  # {"sys", "dia", "pul"} on success
    error: str | None = None
    display_box: tuple[int, int, int, int] | None = None  # unpadded x1, y1, x2, y2
    display_conf: float | None = None
    display_width_frac: float | None = None  # box width / image width


def best_display_box(result) -> tuple[tuple[int, int, int, int], float] | None:
    """Highest-confidence box of a display detector result."""
    boxes = result.boxes
    if len(boxes) == 0:
        return None
    confs = boxes.conf.cpu().numpy()
    best_idx = int(confs.argmax())
    x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[best_idx].cpu().numpy().astype(int))
    return (x1, y1, x2, y2), float(confs[best_idx])


def padded_box(
    box: tuple[int, int, int, int], img_w: int, img_h: int, padding_ratio: float
) -> tuple[int, int, int, int]:
    """Grows the box by padding_ratio of its size on every side, clamped to the image."""
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(img_w, x2 + pad_x),
        min(img_h, y2 + pad_y),
    )


def crop_display(img, box: tuple[int, int, int, int], cfg: Config):
    """Crops the padded display box and resizes it to the configured size."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = padded_box(box, w, h, cfg.padding_ratio)
    return cv2.resize(
        img[y1:y2, x1:x2], (cfg.crop_width, cfg.crop_height), interpolation=cv2.INTER_AREA
    )


def boxes_from_result(result) -> list[dict]:
    """Extracts boxes from a YOLO result into plain dicts."""
    boxes = result.boxes
    if len(boxes) == 0:
        return []
    cls_arr = boxes.cls.cpu().numpy().astype(int)
    conf_arr = boxes.conf.cpu().numpy()
    xyxy_arr = boxes.xyxy.cpu().numpy()
    out = []
    for cls, conf, (x1, y1, x2, y2) in zip(cls_arr, conf_arr, xyxy_arr, strict=False):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h = y2 - y1
        out.append({"cls": int(cls), "conf": float(conf),
                    "cx": float(cx), "cy": float(cy),
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "h": float(h)})
    return out


def _iou(a: dict, b: dict) -> float:
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


def class_agnostic_nms(boxes: list[dict], iou_threshold: float = 0.4) -> list[dict]:
    """
    Extra filter on top of the YOLO NMS: drops boxes that overlap a more confident
    box above the IoU threshold, regardless of class. Fixes cases where YOLO emits
    e.g. `2` and `3` at almost the same place.
    """
    if not boxes:
        return []

    kept = []
    for box in sorted(boxes, key=lambda b: -b["conf"]):
        if all(_iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def filter_by_height(boxes: list[dict], min_height_ratio: float = 0.4) -> list[dict]:
    """Drops boxes shorter than min_height_ratio of the tallest box."""
    if not boxes:
        return []
    max_h = max(b["h"] for b in boxes)
    return [b for b in boxes if b["h"] > max_h * min_height_ratio]


def group_into_rows(boxes: list[dict]) -> list[list[dict]]:
    """
    Groups boxes into 3 rows by Y (top to bottom) with a plain 1D k-means, k=3.
    Boxes inside a row are sorted by X (left to right).

    With fewer than 3 boxes fewer rows are returned (the caller reports the error).
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

    # Drop empty rows (all boxes may have landed in one cluster)
    return [r for r in rows if r]


def assemble_number(row: list[dict]) -> int | None:
    """Concatenates the classes of a row into a number."""
    if not row:
        return None
    try:
        return int("".join(str(b["cls"]) for b in row))
    except ValueError:
        return None


def read_digits_with_boxes(
    cropped, digit_model, cfg: Config
) -> tuple[dict | None, str | None, list[dict]]:
    """Digit detector on a cropped display. Returns (values, error, kept boxes)."""
    results = digit_model(cropped, conf=cfg.digit_conf, verbose=False)
    boxes = boxes_from_result(results[0])
    if not boxes:
        return None, "no digits", []

    boxes = class_agnostic_nms(boxes, cfg.nms_iou)
    boxes = filter_by_height(boxes, cfg.min_height_ratio)

    rows = group_into_rows(boxes)
    if len(rows) != 3:
        return None, f"got {len(rows)} rows, need 3", boxes

    nums = [assemble_number(r) for r in rows]
    if any(n is None for n in nums):
        return None, "assemble failed", boxes

    return {"sys": nums[0], "dia": nums[1], "pul": nums[2]}, None, boxes


def read_digits(cropped, digit_model, cfg: Config) -> tuple[dict | None, str | None]:
    """Digit detector on a cropped display. Returns (values, error)."""
    values, error, _ = read_digits_with_boxes(cropped, digit_model, cfg)
    return values, error


def recognize(img, display_model, digit_model, cfg: Config) -> Recognition:
    """Full pipeline on an original photo (BGR array)."""
    results = display_model(img, conf=cfg.display_conf, verbose=False)
    found = best_display_box(results[0])
    if found is None:
        return Recognition(error="no display")

    box, conf = found
    width_frac = (box[2] - box[0]) / img.shape[1]
    rec = Recognition(display_box=box, display_conf=conf, display_width_frac=width_frac)

    cropped = crop_display(img, box, cfg)
    rec.values, rec.error = read_digits(cropped, digit_model, cfg)
    return rec
