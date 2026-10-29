"""Imports manually captured photos into photos_dir (the only code that writes there)."""

import csv
import json
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

from bpocr.config import Config
from bpocr.data import IMAGE_EXTENSIONS

MAX_LONG_EDGE = 1024
JPEG_QUALITY = 92
LOG_NAME = "import-log.csv"
LOG_COLUMNS = ["session", "original", "target"]
EXIF_IFD = 0x8769
EXIF_DATETIME_ORIGINAL = 0x9003

RANGES = {"sys": (40, 300), "dia": (20, 200), "pul": (30, 250)}


def parse_values(text: str) -> dict:
    """Parses the single 'sys dia pul' line of values.txt; raises ValueError if invalid."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one line, got {len(lines)}")
    parts = lines[0].split()
    if len(parts) != 3:
        raise ValueError(f"expected 3 integers, got {len(parts)} fields")
    try:
        values = dict(zip(("sys", "dia", "pul"), (int(p) for p in parts), strict=True))
    except ValueError:
        raise ValueError(f"not integers: {lines[0].strip()!r}") from None

    for name, (lo, hi) in RANGES.items():
        if not lo <= values[name] <= hi:
            raise ValueError(f"{name}={values[name]} out of range {lo}..{hi}")
    if values["sys"] <= values["dia"]:
        raise ValueError(f"sys ({values['sys']}) must be greater than dia ({values['dia']})")
    return values


def capture_time(image: Image.Image, path: Path) -> datetime:
    """EXIF DateTimeOriginal, falling back to the file modification time."""
    raw = image.getexif().get_ifd(EXIF_IFD).get(EXIF_DATETIME_ORIGINAL)
    if raw:
        try:
            return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def target_path(photos_dir: Path, when: datetime, taken: set[Path] | None = None) -> Path:
    """<photos_dir>/<YYYY-MM>/<yyyyMMdd_HHmmss>[_N].jpg that does not exist yet."""
    folder = photos_dir / when.strftime("%Y-%m")
    base = when.strftime("%Y%m%d_%H%M%S")
    taken = taken or set()
    n = 1
    while True:
        name = base if n == 1 else f"{base}_{n}"
        candidate = folder / f"{name}.jpg"
        if candidate not in taken and not candidate.exists() \
                and not candidate.with_suffix(".json").exists():
            return candidate
        n += 1


def prepare_image(image: Image.Image) -> Image.Image:
    """Applies EXIF orientation and downscales so the long edge is at most MAX_LONG_EDGE."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    long_edge = max(image.size)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def build_json(values: dict, when: datetime, session: str) -> dict:
    return {
        "sys": values["sys"],
        "dia": values["dia"],
        "pul": values["pul"],
        "timestamp": when.isoformat(),
        "device_model": "Paramed Expert-X",
        "source": "manual_capture",
        "corrected_by_user": False,
        "gemini_suggested": None,
        "notes": None,
        "quality_flags": None,
        "session": session,
    }


def _read_log(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["session"], r["original"]) for r in csv.DictReader(f)}


def _append_log(path: Path, session: str, original: str, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([session, original, target.name])


def import_photos(cfg: Config, raw_dir: Path, dry_run: bool) -> int:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    log_path = cfg.work_dir / LOG_NAME
    imported = _read_log(log_path)
    taken: set[Path] = set()
    n_written = n_skipped = 0

    for session_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        session = session_dir.name
        values_file = session_dir / "values.txt"
        if not values_file.is_file():
            print(f"[SKIP] {session}: values.txt is missing")
            continue
        try:
            values = parse_values(values_file.read_text(encoding="utf-8"))
        except ValueError as e:
            print(f"[SKIP] {session}: invalid values.txt: {e}")
            continue

        photos = sorted(p for p in session_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not photos:
            print(f"[SKIP] {session}: no photos")
            continue

        for photo in photos:
            if (session, photo.name) in imported:
                print(f"[SKIP] {session}/{photo.name}: already imported")
                n_skipped += 1
                continue
            with Image.open(photo) as src:
                when = capture_time(src, photo)
                image = prepare_image(src)
            dst = target_path(cfg.photos_dir, when, taken)
            taken.add(dst)

            if dry_run:
                print(f"[DRY]  {session}/{photo.name} -> {dst} "
                      f"({values['sys']}/{values['dia']}/{values['pul']})")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                image.save(dst, "JPEG", quality=JPEG_QUALITY)
                dst.with_suffix(".json").write_text(
                    json.dumps(build_json(values, when, session), indent=2) + "\n",
                    encoding="utf-8")
                _append_log(log_path, session, photo.name, dst)
                imported.add((session, photo.name))
                print(f"[OK]   {session}/{photo.name} -> {dst}")
            n_written += 1

    verb = "would be imported" if dry_run else "imported"
    print(f"\n{n_written} photos {verb}, {n_skipped} already imported")
    return 0
