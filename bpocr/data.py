"""Photo discovery, ground truth and test split loading."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUBFOLDER_RE = re.compile(r"^\d{4}-\d{2}$")


@dataclass(frozen=True)
class Photo:
    path: Path
    subfolder: str
    json_path: Path | None

    @property
    def stem(self) -> str:
        return self.path.stem


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def find_images(root: Path) -> list[Path]:
    """All images under root (recursive, case-insensitive extensions), sorted."""
    return sorted(p for p in root.rglob("*") if is_image(p))


def sibling_json(image: Path) -> Path | None:
    for suffix in (".json", ".JSON"):
        candidate = image.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def discover_photos(photos_dir: Path) -> list[Photo]:
    """Photos from the YYYY-MM subfolders of photos_dir (other folders are ignored)."""
    if not photos_dir.is_dir():
        raise FileNotFoundError(f"Photos directory not found: {photos_dir}")
    photos = []
    for sub in sorted(photos_dir.iterdir()):
        if not (sub.is_dir() and SUBFOLDER_RE.match(sub.name)):
            continue
        for image in find_images(sub):
            photos.append(Photo(image, sub.name, sibling_json(image)))
    return photos


def load_ground_truth(json_path: Path) -> dict:
    """Reads {sys, dia, pul} from a ground truth json."""
    with open(json_path, encoding="utf-8") as f:
        gt = json.load(f)
    return {k: int(gt[k]) for k in ("sys", "dia", "pul")}


def load_session(json_path: Path | None) -> str | None:
    """The 'session' field of a ground truth json, if present."""
    if json_path is None:
        return None
    with open(json_path, encoding="utf-8") as f:
        session = json.load(f).get("session")
    return str(session) if session else None


def load_test_split(path: Path) -> set[str]:
    """Stems or subfolder names listed one per line; lines starting with # are comments."""
    if not path.is_file():
        return set()
    entries = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.add(line)
    return entries


def split_of(photo: Photo, test_entries: set[str]) -> str:
    if photo.stem in test_entries or photo.subfolder in test_entries:
        return "test"
    return "train"
