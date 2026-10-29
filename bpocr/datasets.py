"""Builds the display and digit datasets from autolabel output (test split excluded)."""

import shutil
from pathlib import Path

from bpocr.config import Config
from bpocr.data import discover_photos, load_session, load_test_split, split_of
from bpocr.train import _split_and_copy, _write_data_yaml


def session_group(pair: tuple[Path, Path], sessions: dict[str, str | None]) -> str:
    """Group key of a pair: its session, or the photo itself when there is none."""
    stem = pair[0].stem
    session = sessions.get(stem)
    return f"session:{session}" if session else f"photo:{stem}"


def build_datasets(cfg: Config) -> int:
    test_entries = load_test_split(cfg.test_split)
    photos = [p for p in discover_photos(cfg.photos_dir)
              if p.json_path is not None and split_of(p, test_entries) != "test"]
    sessions = {p.stem: load_session(p.json_path) for p in photos}

    labels = cfg.work_dir / "labels"
    crops = cfg.work_dir / "crops"
    display_pairs = [(p.path, labels / "display" / f"{p.stem}.txt") for p in photos
                     if (labels / "display" / f"{p.stem}.txt").is_file()]
    digit_pairs = [(crops / f"{p.stem}.jpg", labels / "digits" / f"{p.stem}.txt")
                   for p in photos
                   if (labels / "digits" / f"{p.stem}.txt").is_file()
                   and (crops / f"{p.stem}.jpg").is_file()]

    print(f"Photos outside the test split: {len(photos)}")
    if not display_pairs or not digit_pairs:
        print("No labels found - run 'bpocr autolabel' first.")
        return 1

    def group_of(pair):
        return session_group(pair, sessions)

    root = cfg.work_dir / "datasets"
    for name, pairs, nc, names in (
        ("display", display_pairs, 1, ["display"]),
        ("digits", digit_pairs, 10, [str(i) for i in range(10)]),
    ):
        out = root / name
        if out.exists():
            shutil.rmtree(out)
        n_train, n_val = _split_and_copy(pairs, out, group_of)
        _write_data_yaml(out, f"bpocr build-datasets ({name})", nc, names)
        print(f"{name}: train={n_train} val={n_val} -> {out}")
    return 0
