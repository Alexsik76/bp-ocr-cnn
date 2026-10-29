"""Command line interface: bpocr <command>."""

import argparse
import sys
from pathlib import Path

from bpocr.config import DEFAULT_CONFIG, load_config
from bpocr.validate import BACKENDS, SPLITS


def build_parser() -> argparse.ArgumentParser:
    # SUPPRESS keeps the value given before the subcommand when it is not repeated after it
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=argparse.SUPPRESS,
                        help=f"path to the config file (default: {DEFAULT_CONFIG})")

    parser = argparse.ArgumentParser(prog="bpocr", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"path to the config file (default: {DEFAULT_CONFIG})")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", parents=[common],
                       help="end-to-end accuracy on all photos with ground truth")
    p.add_argument("--backend", choices=BACKENDS, default="pt")
    p.add_argument("--split", choices=SPLITS, default="all")

    p = sub.add_parser("import", parents=[common],
                       help="import captured sessions (photos + values.txt) into photos_dir")
    p.add_argument("raw_dir", type=Path, help="folder with one subfolder per session")
    p.add_argument("--dry-run", action="store_true", help="print what would be written")

    sub.add_parser("restore-manual", parents=[common],
                   help="restore the manually labelled display crops from git history")
    sub.add_parser("autolabel", parents=[common],
                   help="generate display and digit labels verified by ground truth")
    sub.add_parser("gemini-display", parents=[common],
                   help="ask Gemini for display boxes of photos with display-not-found")
    sub.add_parser("build-datasets", parents=[common],
                   help="build train/val datasets from autolabel output")

    p = sub.add_parser("crop", parents=[common], help="crop displays out of photos")
    p.add_argument("input", nargs="?", type=Path,
                   help="folder with images (default: photos_dir from config)")
    p.add_argument("--out", type=Path, help="output folder (default: work_dir)")

    p = sub.add_parser("verify-labels", parents=[common],
                       help="compare digit labels with ground truth")
    p.add_argument("--cropped", type=Path, help="cropped images (default: <work_dir>/cropped)")
    p.add_argument("--labels", type=Path, default=Path("labels/labels5"))

    p = sub.add_parser("prepare-display", parents=[common], help="build the display dataset")
    p.add_argument("--labels", type=Path, default=Path("labels/labels1"))
    p.add_argument("--out", type=Path, help="default: <work_dir>/dataset")

    p = sub.add_parser("prepare-digits", parents=[common], help="build the digit dataset")
    p.add_argument("--cropped", type=Path, help="cropped images (default: <work_dir>/cropped)")
    p.add_argument("--labels", type=Path, default=Path("labels/labels5"))
    p.add_argument("--out", type=Path, help="default: <work_dir>/dataset_digits")

    p = sub.add_parser("train-display", parents=[common], help="train the display detector")
    p.add_argument("--data", type=Path, help="default: <work_dir>/datasets/display/data.yaml")

    p = sub.add_parser("train-digits", parents=[common], help="train the digit detector")
    p.add_argument("--data", type=Path, help="default: <work_dir>/datasets/digits/data.yaml")

    sub.add_parser("export", parents=[common], help="export both models to ONNX and int8")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        return run(args, cfg) or 0
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def run(args: argparse.Namespace, cfg) -> int | None:
    work = cfg.work_dir
    match args.command:
        case "validate":
            from bpocr.validate import validate
            return validate(cfg, args.backend, args.split)
        case "import":
            from bpocr.importer import import_photos
            return import_photos(cfg, args.raw_dir, args.dry_run)
        case "restore-manual":
            from bpocr.manual import restore_manual
            return restore_manual(cfg)
        case "gemini-display":
            from bpocr.gemini_display import gemini_display
            return gemini_display(cfg)
        case "autolabel":
            from bpocr.autolabel import autolabel
            return autolabel(cfg)
        case "build-datasets":
            from bpocr.datasets import build_datasets
            return build_datasets(cfg)
        case "crop":
            from bpocr.crop import crop_photos
            return crop_photos(cfg, args.input, args.out)
        case "verify-labels":
            from bpocr.labels import verify
            return verify(cfg, args.cropped or work / "cropped", args.labels)
        case "prepare-display":
            from bpocr.train import prepare_display
            return prepare_display(cfg, args.labels, args.out or work / "dataset")
        case "prepare-digits":
            from bpocr.train import prepare_digits
            return prepare_digits(args.cropped or work / "cropped", args.labels,
                                  args.out or work / "dataset_digits")
        case "train-display":
            from bpocr.train import train_display
            train_display(cfg, args.data or work / "datasets" / "display" / "data.yaml")
        case "train-digits":
            from bpocr.train import train_digits
            train_digits(cfg, args.data or work / "datasets" / "digits" / "data.yaml")
        case "export":
            from bpocr.export import export_all
            export_all(cfg)


if __name__ == "__main__":
    sys.exit(main())
