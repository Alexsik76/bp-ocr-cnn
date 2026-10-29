"""ONNX export (fp32, opset 17, static shapes) and dynamic int8 quantization."""

from pathlib import Path

from bpocr.config import Config


def export_onnx(pt_path: Path) -> Path:
    from ultralytics import YOLO

    if not pt_path.exists():
        raise FileNotFoundError(f"Model not found: {pt_path}")

    YOLO(str(pt_path)).export(format="onnx", dynamic=False, simplify=True, opset=17)

    onnx_path = pt_path.with_suffix(".onnx")
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"  {pt_path.name} -> {onnx_path.name}  ({size_mb:.2f} MB)")
    return onnx_path


def quantize_int8(onnx_path: Path) -> Path:
    """Dynamic weight-only int8 quantization, no calibration."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    quantize_dynamic(str(onnx_path), str(dst), weight_type=QuantType.QUInt8)
    orig_mb = onnx_path.stat().st_size / 1024 / 1024
    int8_mb = dst.stat().st_size / 1024 / 1024
    print(f"  {onnx_path.name} ({orig_mb:.2f} MB) -> {dst.name} ({int8_mb:.2f} MB, "
          f"{int8_mb / orig_mb * 100:.0f}%)")
    return dst


def export_all(cfg: Config) -> None:
    for pt in (cfg.display_model, cfg.digit_model):
        quantize_int8(export_onnx(pt))
