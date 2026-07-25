"""
Exports display_detector and digit_detector to ONNX (fp32, opset=17).

Produces best.onnx alongside best.pt in each weights/ directory.
Does not modify or replace existing .pt files.

Usage:
  python export_onnx.py
"""

from pathlib import Path

from ultralytics import YOLO

MODELS = [
    "runs/detect/display_detector_v1/weights/best.pt",
    "runs/detect/digit_detector_latest/weights/best.pt",
]


def export_to_onnx(pt_path: str) -> Path:
    src = Path(pt_path)
    if not src.exists():
        raise FileNotFoundError(f"Model not found: {src}")

    model = YOLO(str(src))
    model.export(format="onnx", dynamic=False, simplify=True, opset=17)

    onnx_path = src.with_suffix(".onnx")
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"  {src.name} → {onnx_path.name}  ({size_mb:.2f} MB)")
    return onnx_path


def main():
    print("Exporting to ONNX (fp32, opset=17, static shapes)...")
    for pt in MODELS:
        export_to_onnx(pt)
    print("\nDone. Run validate_pipeline.py with --backend onnx to verify accuracy.")


if __name__ == "__main__":
    main()
