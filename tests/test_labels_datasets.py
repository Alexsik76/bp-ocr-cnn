import json
from pathlib import Path

import pytest

from bpocr.autolabel import from_yolo_line, read_manual_box, to_yolo_line
from bpocr.config import Config
from bpocr.datasets import build_datasets
from bpocr.train import split_pairs


def test_yolo_label_round_trip():
    line = to_yolo_line(7, 100, 50, 300, 250, 400, 480)
    cls, x1, y1, x2, y2 = from_yolo_line(line, 400, 480)
    assert cls == 7
    assert (x1, y1, x2, y2) == pytest.approx((100, 50, 300, 250), abs=1e-3)
    assert line == "7 0.500000 0.312500 0.500000 0.416667"


def test_read_manual_box(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(to_yolo_line(0, 100, 200, 300, 600, 1000, 1000) + "\n")
    assert read_manual_box(f, 1000, 1000) == (100, 200, 300, 600)
    assert read_manual_box(tmp_path / "missing.txt", 1000, 1000) is None


def test_split_pairs_keeps_groups_together():
    pairs = [(Path(f"s{g}_{i}.jpg"), Path(f"s{g}_{i}.txt")) for g in range(10) for i in range(3)]
    train, val = split_pairs(pairs, group_of=lambda p: p[0].name.split("_")[0])
    assert len(train) + len(val) == 30
    assert {p[0].name.split("_")[0] for p in train}.isdisjoint(
        {p[0].name.split("_")[0] for p in val})
    assert val  # at least one group goes to val


def _config(tmp_path: Path) -> Config:
    return Config(
        photos_dir=tmp_path / "photos", work_dir=tmp_path / "work",
        display_model=Path("d.pt"), digit_model=Path("g.pt"),
        test_split=tmp_path / "test.txt", manual_display_labels=tmp_path / "labels1",
    )


def test_build_datasets_groups_sessions_and_excludes_test(tmp_path):
    cfg = _config(tmp_path)
    folder = cfg.photos_dir / "2026-05"
    folder.mkdir(parents=True)
    (cfg.work_dir / "labels" / "display").mkdir(parents=True)
    (cfg.work_dir / "labels" / "digits").mkdir(parents=True)
    (cfg.work_dir / "crops").mkdir(parents=True)

    stems = []
    for i in range(20):
        stem = f"202605{i + 1:02d}_000000"
        stems.append(stem)
        (folder / f"{stem}.jpg").write_bytes(b"x")
        meta = {"sys": 120, "dia": 80, "pul": 60}
        if i < 12:
            meta["session"] = f"s{i // 3}"  # four sessions of three photos
        (folder / f"{stem}.json").write_text(json.dumps(meta))
        (cfg.work_dir / "labels" / "display" / f"{stem}.txt").write_text("0 .5 .5 .1 .1\n")
        (cfg.work_dir / "labels" / "digits" / f"{stem}.txt").write_text("1 .5 .5 .1 .1\n")
        (cfg.work_dir / "crops" / f"{stem}.jpg").write_bytes(b"x")

    test_stems = {stems[0], stems[15]}
    cfg.test_split.write_text("# test\n" + "\n".join(sorted(test_stems)) + "\n")

    assert build_datasets(cfg) == 0

    for name in ("display", "digits"):
        root = cfg.work_dir / "datasets" / name
        assert (root / "data.yaml").is_file()
        train = {p.stem for p in (root / "images" / "train").iterdir()}
        val = {p.stem for p in (root / "images" / "val").iterdir()}
        assert train.isdisjoint(val)
        assert (train | val) == set(stems) - test_stems
        for s in range(4):
            members = {st for i, st in enumerate(stems) if i // 3 == s and i < 12}
            members -= test_stems
            assert members <= train or members <= val


def test_build_datasets_excludes_session_in_test_split(tmp_path):
    cfg = _config(tmp_path)
    folder = cfg.photos_dir / "2026-05"
    folder.mkdir(parents=True)
    (cfg.work_dir / "labels" / "display").mkdir(parents=True)
    (cfg.work_dir / "labels" / "digits").mkdir(parents=True)
    (cfg.work_dir / "crops").mkdir(parents=True)

    stems = ["photo1", "photo2", "photo3", "photo4"]
    sessions = ["s04", "s04", "s05", None]
    for stem, session in zip(stems, sessions, strict=True):
        (folder / f"{stem}.jpg").write_bytes(b"x")
        meta = {"sys": 120, "dia": 80, "pul": 60}
        if session:
            meta["session"] = session
        (folder / f"{stem}.json").write_text(json.dumps(meta))
        (cfg.work_dir / "labels" / "display" / f"{stem}.txt").write_text("0 .5 .5 .1 .1\n")
        (cfg.work_dir / "labels" / "digits" / f"{stem}.txt").write_text("1 .5 .5 .1 .1\n")
        (cfg.work_dir / "crops" / f"{stem}.jpg").write_bytes(b"x")

    cfg.test_split.write_text("s04\n")
    assert build_datasets(cfg) == 0

    for name in ("display", "digits"):
        root = cfg.work_dir / "datasets" / name
        assert (root / "data.yaml").is_file()
        train = {p.stem for p in (root / "images" / "train").iterdir()}
        val = {p.stem for p in (root / "images" / "val").iterdir()}
        assert train.isdisjoint(val)
        assert "photo1" not in (train | val)
        assert "photo2" not in (train | val)
        assert (train | val) == {"photo3", "photo4"}
