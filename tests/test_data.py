import json

import pytest

from bpocr.data import (
    discover_photos,
    load_ground_truth,
    load_test_split,
    split_of,
)


def touch(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_discover_photos_mixed_extensions_and_missing_json(tmp_path):
    touch(tmp_path / "2026-05" / "a.jpg")
    touch(tmp_path / "2026-05" / "a.json", json.dumps({"sys": 1, "dia": 2, "pul": 3}))
    touch(tmp_path / "2026-05" / "b.JPEG")
    touch(tmp_path / "2026-05" / "c.PNG")
    touch(tmp_path / "2026-05" / "c.json", "{}")
    touch(tmp_path / "2026-06" / "nested" / "d.Jpg")
    touch(tmp_path / "2026-05" / "notes.txt")
    touch(tmp_path / "labels" / "e.jpg")  # not a YYYY-MM folder
    touch(tmp_path / "top.jpg")

    photos = discover_photos(tmp_path)

    assert [p.stem for p in photos] == ["a", "b", "c", "d"]
    by_stem = {p.stem: p for p in photos}
    assert by_stem["a"].json_path == tmp_path / "2026-05" / "a.json"
    assert by_stem["b"].json_path is None
    assert by_stem["c"].json_path is not None
    assert by_stem["d"].subfolder == "2026-06"


def test_discover_photos_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_photos(tmp_path / "nope")


def test_load_ground_truth(tmp_path):
    touch(tmp_path / "a.json", json.dumps({"sys": 125, "dia": 74, "pul": 73, "x": 1}))
    assert load_ground_truth(tmp_path / "a.json") == {"sys": 125, "dia": 74, "pul": 73}


def test_load_test_split_parsing(tmp_path):
    f = tmp_path / "test.txt"
    touch(f, "# comment\n\n20260503_043525\n  20260504_053951  \n# 20260505_000000\n2026-06\n")
    assert load_test_split(f) == {"20260503_043525", "20260504_053951", "2026-06"}


def test_load_test_split_missing_file(tmp_path):
    assert load_test_split(tmp_path / "missing.txt") == set()


def test_split_of_by_stem_or_subfolder(tmp_path):
    touch(tmp_path / "2026-05" / "a.jpg")
    touch(tmp_path / "2026-06" / "b.jpg")
    touch(tmp_path / "2026-07" / "c.jpg")
    a, b, c = discover_photos(tmp_path)
    entries = {"a", "2026-06"}
    assert [split_of(p, entries) for p in (a, b, c)] == ["test", "test", "train"]
