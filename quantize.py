"""
Динамічна int8-квантизація ONNX-моделей (weight-only, без калібрування).

Створює best_int8.onnx поряд з best.onnx — оригінал не чіпає.

Usage:
  python quantize.py
"""

from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType


MODELS = [
    "runs/detect/display_detector_v1/weights/best.onnx",
    "runs/detect/digit_detector_latest/weights/best.onnx",
]


def quantize(onnx_path: str) -> Path:
    src = Path(onnx_path)
    if not src.exists():
        raise FileNotFoundError(f"ONNX not found: {src}  (run export_onnx.py first)")
    dst = src.with_name(src.stem + "_int8.onnx")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    orig_mb = src.stat().st_size / 1024 / 1024
    int8_mb = dst.stat().st_size / 1024 / 1024
    ratio = int8_mb / orig_mb * 100
    print(f"  {src.name} ({orig_mb:.2f} MB) -> {dst.name} ({int8_mb:.2f} MB, {ratio:.0f}%)")
    return dst


def main():
    print("Quantizing to int8 (dynamic, weight-only)...")
    for m in MODELS:
        quantize(m)
    print("\nDone. Run validate_pipeline.py with --backend int8 to check accuracy.")


if __name__ == "__main__":
    main()
