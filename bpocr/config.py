"""Configuration loaded from bpocr.toml."""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_CONFIG = Path("bpocr.toml")


@dataclass(frozen=True)
class Config:
    photos_dir: Path
    work_dir: Path
    display_model: Path
    digit_model: Path
    test_split: Path
    manual_display_labels: Path
    display_conf: float = 0.05
    digit_conf: float = 0.25
    padding_ratio: float = 0.03
    crop_width: int = 400
    crop_height: int = 480
    nms_iou: float = 0.4
    min_height_ratio: float = 0.4


_PATH_KEYS = ("photos_dir", "work_dir", "display_model", "digit_model", "test_split",
              "manual_display_labels")


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    """Load the config; relative paths are resolved against the config file's directory."""
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {', '.join(sorted(unknown))}")

    base = path.resolve().parent
    for key in _PATH_KEYS:
        if key not in raw:
            raise ValueError(f"Missing key '{key}' in {path}")
        # An absolute right-hand operand replaces the base
        raw[key] = base / raw[key]
    return Config(**raw)
