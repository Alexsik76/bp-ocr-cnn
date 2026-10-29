"""Restores the manually labelled display crops that were removed from the repository."""

import subprocess
from pathlib import Path

from bpocr.config import Config
from bpocr.labels import verify

# The commit that deleted cropped/; its parent still holds the crops
DELETING_COMMIT = "d799d32"
CROPS_TREE = "cropped/"


def manual_digit_labels_dir(cfg: Config) -> Path:
    """labels/labels5 next to the manual display labels (labels/labels1)."""
    return cfg.manual_display_labels.parent / "labels5"


def manual_crops_dir(cfg: Config) -> Path:
    return cfg.work_dir / "manual" / "crops"


def repo_root(cfg: Config) -> Path:
    """Repository root: the parent of labels/."""
    return cfg.manual_display_labels.parent.parent


def git_output(repo: Path, *args: str) -> bytes:
    """Runs a read-only git command in repo and returns its raw stdout."""
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def restore_crops(repo: Path, out_dir: Path, rev: str = f"{DELETING_COMMIT}^") -> list[Path]:
    """Writes every file of <rev>:cropped/ into out_dir via git show; returns the paths."""
    listing = git_output(repo, "ls-tree", "--name-only", rev, CROPS_TREE)
    names = [n for n in listing.decode().splitlines() if n]
    if not names:
        raise RuntimeError(f"No files under {CROPS_TREE} in {rev}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        dst = out_dir / Path(name).name
        dst.write_bytes(git_output(repo, "show", f"{rev}:{name}"))
        written.append(dst)
    return written


def restore_manual(cfg: Config) -> int:
    out_dir = manual_crops_dir(cfg)
    written = restore_crops(repo_root(cfg), out_dir)
    print(f"Restored {len(written)} crops into {out_dir}")

    labels_dir = manual_digit_labels_dir(cfg)
    missing = [t.stem for t in sorted(labels_dir.glob("*.txt"))
               if not (out_dir / f"{t.stem}.jpg").is_file()]
    if missing:
        print(f"Labels without a crop: {', '.join(missing)}")
    else:
        print(f"Every label in {labels_dir} has a crop")

    print()
    verify(cfg, out_dir, labels_dir)
    return 1 if missing else 0
