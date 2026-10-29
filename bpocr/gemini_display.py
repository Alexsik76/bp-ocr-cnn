"""Display boxes from Gemini for photos where the display model finds nothing."""

import base64
import json
import os
from pathlib import Path

import requests

from bpocr.config import Config
from bpocr.data import discover_photos

PROMPT = (
    "The photo shows a blood pressure monitor. Find its LCD screen: the rectangular "
    "display area that shows the digits (systolic, diastolic, pulse), without the plastic "
    "case. Return JSON {\"box_2d\": [ymin, xmin, ymax, xmax]} with coordinates normalised "
    "to 0-1000. If no display is visible, return {\"box_2d\": null}."
)


def gemini_dir(cfg: Config) -> Path:
    return cfg.work_dir / "gemini" / "display"


def ask_box(image_path: Path, api_key: str, model: str) -> list[int] | None:
    """Gemini box as [ymin, xmin, ymax, xmax] on a 0-1000 scale, or None."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": base64.b64encode(image_path.read_bytes()).decode()}},
            {"text": PROMPT},
        ]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    # Key in a header, not in the URL, so it never appears in error messages
    r = requests.post(url, headers={"x-goog-api-key": api_key}, json=body, timeout=60)
    r.raise_for_status()
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    data = json.loads(text)
    if isinstance(data, list):
        data = data[0] if data else {}
    box = data.get("box_2d")
    if not box or len(box) != 4:
        return None
    return [int(v) for v in box]


def to_yolo(box: list[int]) -> str:
    ymin, xmin, ymax, xmax = (v / 1000 for v in box)
    return f"0 {(xmin + xmax) / 2:.6f} {(ymin + ymax) / 2:.6f} {xmax - xmin:.6f} {ymax - ymin:.6f}"


def gemini_display(cfg: Config, only_review: bool = True) -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set")
        return 1
    model = os.environ.get("BPOCR_GEMINI_MODEL") or "gemini-flash-latest"

    stems = None
    if only_review:
        review = cfg.work_dir / "review.csv"
        if not review.is_file():
            print("work/review.csv is missing - run 'bpocr autolabel' first")
            return 1
        lines = review.read_text(encoding="utf-8").splitlines()[1:]
        stems = {ln.split(",")[0] for ln in lines if ",display-not-found," in ln}

    out = gemini_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    n_ok = n_none = 0
    for photo in discover_photos(cfg.photos_dir):
        if photo.json_path is None or (stems is not None and photo.stem not in stems):
            continue
        target = out / f"{photo.stem}.txt"
        if target.is_file():
            continue
        box = ask_box(photo.path, api_key, model)
        if box is None:
            n_none += 1
            print(f"{photo.stem}: no display")
            continue
        target.write_text(to_yolo(box) + "\n", encoding="utf-8")
        n_ok += 1
        print(f"{photo.stem}: {box}")
    print(f"Gemini display boxes: {n_ok} written, {n_none} not found (model {model})")
    return 0
