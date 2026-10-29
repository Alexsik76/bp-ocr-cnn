import os
import subprocess
from pathlib import Path

import pytest

from bpocr.autolabel import clear_files, place_manual_digits
from bpocr.manual import restore_crops


def _git(repo: Path, *args: str) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)


def test_restore_crops_from_git_history(tmp_path):
    repo = tmp_path / "repo"
    (repo / "cropped").mkdir(parents=True)
    _git(repo, "init", "-q")
    payload = {"a.jpg": bytes(range(256)), "b.jpg": b"\xff\xd8\x00\r\n\x00end"}
    for name, data in payload.items():
        (repo / "cropped" / name).write_bytes(data)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add crops")
    _git(repo, "rm", "-q", "-r", "cropped")
    _git(repo, "commit", "-q", "-m", "delete crops")

    out = tmp_path / "out"
    written = restore_crops(repo, out, rev="HEAD^")

    assert sorted(p.name for p in written) == ["a.jpg", "b.jpg"]
    for name, data in payload.items():
        assert (out / name).read_bytes() == data
    assert not (repo / "cropped").exists()  # working tree untouched


def test_restore_crops_missing_tree_reports_error(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "x.txt").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    with pytest.raises(RuntimeError):
        restore_crops(repo, tmp_path / "out", rev="HEAD")


def test_manual_digit_labels_override_stale_automatic_ones(tmp_path):
    digits, crops = tmp_path / "digits", tmp_path / "crops"
    digits.mkdir()
    crops.mkdir()
    (digits / "p.txt").write_text("auto label")
    (crops / "p.jpg").write_bytes(b"auto crop")

    clear_files(digits)
    clear_files(crops)
    assert list(digits.iterdir()) == [] and list(crops.iterdir()) == []

    label = tmp_path / "labels5" / "p.txt"
    crop = tmp_path / "manual" / "p.jpg"
    label.parent.mkdir()
    crop.parent.mkdir()
    label.write_text("manual label")
    crop.write_bytes(b"manual crop")

    assert place_manual_digits(label, crop, digits, crops) is True
    assert (digits / "p.txt").read_text() == "manual label"
    assert (crops / "p.jpg").read_bytes() == b"manual crop"


def test_place_manual_digits_needs_both_files(tmp_path):
    digits, crops = tmp_path / "d", tmp_path / "c"
    digits.mkdir()
    crops.mkdir()
    label = tmp_path / "p.txt"
    label.write_text("x")
    assert place_manual_digits(label, tmp_path / "missing.jpg", digits, crops) is False
    assert list(digits.iterdir()) == []
