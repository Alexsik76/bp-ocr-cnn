import json
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from bpocr.config import Config
from bpocr.importer import import_photos, parse_values, prepare_image, target_path


@pytest.mark.parametrize("text,expected", [
    ("125 74 73\n", {"sys": 125, "dia": 74, "pul": 73}),
    ("\n  140\t90   60  \n\n", {"sys": 140, "dia": 90, "pul": 60}),
])
def test_parse_values_ok(text, expected):
    assert parse_values(text) == expected


@pytest.mark.parametrize("text", [
    "", "125 74", "125 74 73 1", "a b c", "125 74 73\n130 80 70",
    "39 20 60", "301 80 60", "120 19 60", "120 201 60", "120 80 29", "120 80 251",
    "100 100 60", "90 100 60",
])
def test_parse_values_invalid(text):
    with pytest.raises(ValueError):
        parse_values(text)


def test_target_path_clash_suffixes(tmp_path):
    when = datetime(2026, 5, 3, 4, 35, 25)
    first = target_path(tmp_path, when)
    assert first == tmp_path / "2026-05" / "20260503_043525.jpg"

    first.parent.mkdir(parents=True)
    first.touch()
    second = target_path(tmp_path, when)
    assert second.name == "20260503_043525_2.jpg"

    second.touch()
    assert target_path(tmp_path, when).name == "20260503_043525_3.jpg"


def test_target_path_respects_taken_set(tmp_path):
    when = datetime(2026, 5, 3, 4, 35, 25)
    taken = {tmp_path / "2026-05" / "20260503_043525.jpg"}
    assert target_path(tmp_path, when, taken).name == "20260503_043525_2.jpg"


def test_exif_orientation_applied():
    raw = Image.new("RGB", (200, 100), "red")
    raw.getexif()[0x0112] = 6  # orientation: rotate 90 degrees to display
    assert prepare_image(raw).size == (100, 200)


def test_prepare_image_downscales_long_edge():
    out = prepare_image(Image.new("RGB", (4000, 3000)))
    assert max(out.size) == 1024
    assert out.size == (1024, 768)
    small = prepare_image(Image.new("RGB", (800, 600)))
    assert small.size == (800, 600)


def _config(tmp_path: Path) -> Config:
    return Config(
        photos_dir=tmp_path / "photos", work_dir=tmp_path / "work",
        display_model=Path("d.pt"), digit_model=Path("g.pt"),
        test_split=tmp_path / "test.txt", manual_display_labels=tmp_path / "labels1",
    )


def test_import_end_to_end_with_dedup(tmp_path):
    raw = tmp_path / "raw"
    good = raw / "s1"
    good.mkdir(parents=True)
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0x9003] = "2026:06:01 08:15:30"
    Image.new("RGB", (2000, 1000), "blue").save(good / "a.JPG", exif=exif)
    (good / "values.txt").write_text("130 80 70\n")
    bad = raw / "s2"
    bad.mkdir()
    Image.new("RGB", (10, 10)).save(bad / "b.png")
    (bad / "values.txt").write_text("nonsense")

    cfg = _config(tmp_path)
    import_photos(cfg, raw, dry_run=True)
    assert not cfg.photos_dir.exists()

    import_photos(cfg, raw, dry_run=False)
    jpg = cfg.photos_dir / "2026-06" / "20260601_081530.jpg"
    assert jpg.is_file()
    assert max(Image.open(jpg).size) == 1024
    meta = json.loads(jpg.with_suffix(".json").read_text())
    assert meta["sys"] == 130 and meta["session"] == "s1"
    assert meta["timestamp"] == "2026-06-01T08:15:30"
    assert meta["source"] == "manual_capture" and meta["corrected_by_user"] is False

    import_photos(cfg, raw, dry_run=False)  # second run imports nothing new
    assert len(list(cfg.photos_dir.rglob("*.jpg"))) == 1
