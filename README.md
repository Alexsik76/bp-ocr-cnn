[README українською](README-UKR.md)

# bp-ocr-cnn

![CI](https://github.com/Alexsik76/bp-ocr-cnn/actions/workflows/ci.yml/badge.svg)

Tool for training and developing ML models to recognize digits from the LCD display of the **Paramed Expert-X** blood pressure monitor. Ready-to-use models are copied to [aivm-photo-api](https://github.com/Alexsik76/aivm-photo-api) for production deployment.

## What it does

Takes a photo of a blood pressure monitor and returns a JSON object with blood pressure values:

```
20260516_044548.jpg  →  {"sys": 125, "dia": 74, "pul": 73}
```

End-to-end check: **42/42 photos parsed correctly**. Note that these 42 photos are also the training set — the dataset is too small for a separate held-out split. The number shows the pipeline works end to end, not how it generalises to unseen devices. Inference speed: **~50 ms on CPU** (Ryzen 7 5700X3D).

## Pipeline Architecture

Two-stage YOLOv8:

```mermaid
flowchart TD
    A[Original photo<br/>~1080×1920] --> B[YOLO #1<br/>display_detector<br/>finds display]
    B --> C[cropped 400×480]
    C --> D[YOLO #2<br/>digit_detector<br/>finds digits 0–9]
    D --> E[class_agnostic_nms<br/>removes duplicate boxes]
    E --> F[K-means with k=3<br/>groups boxes into 3 rows]
    F --> G["{sys, dia, pul}"]
```

**Why two YOLO models instead of one:** Easier to train and debug, and simpler to retrain independently. The first model rarely needs retraining (the display layout stays consistent), while the second model can be retrained as new dataset photos arrive.

**Why YOLOv8 nano:** ~50 ms for the entire pipeline on CPU. Both models combined are ~12 MB — easy to version control in Git.

## Screenshots

### Stage 1 — display detection

![Display detector on photos with different lighting, angle and background](docs/img/detector_conditions.jpg)

The first-stage detector locates the monitor screen on a full photo.
The examples show different lighting, camera angles and backgrounds.

### Stage 2 — digit recognition

![Digit detector on cropped displays, with class and confidence](docs/img/digit_recognition.jpg)

The second-stage detector reads each digit on the cropped display.
Every box shows the predicted class and the confidence score.

## Results

| Model | Precision | Recall | mAP50 | mAP50-95 | Epochs |
|---|---|---|---|---|---|
| Display detector | 0.990 | 1.000 | 0.995 | 0.946 | 100 |
| Digit detector | 0.992 | 1.000 | 0.995 | 0.856 | 37 |

These scores come from a small and uniform dataset: photos of one device model, taken indoors by one person. The numbers show that the task is narrow, not that the models are universal. A different device or a wider range of conditions would need new training data.

## Project Structure

```
bp-ocr-cnn/
├── cropped/                # cropped displays 400×480 (YOLO #1 output)
├── docs/
│   ├── img/                # documentation screenshots
│   │   ├── detector_conditions.jpg
│   │   └── digit_recognition.jpg
│   └── bp-ocr-cnn_PLAN.md  # project roadmap
├── img_test/               # sample photo + ground truth for testing
├── labels/
│   ├── labels1/            # YOLO display labels (1 class)
│   └── labels5/            # YOLO digit labels (10 classes), active batch
├── latest_models/          # exported ONNX models
├── runs/detect/
│   ├── display_detector_v1/weights/
│   │   ├── best.pt         # display model (stable)
│   │   ├── best.onnx       # fp32 ONNX
│   │   └── best_int8.onnx  # dynamic int8
│   ├── digit_detector_latest/weights/
│   │   ├── best.pt         # active digit model
│   │   ├── best.onnx       # fp32 ONNX
│   │   └── best_int8.onnx  # dynamic int8
│   └── digit_detector_bak/weights/
│       └── best.pt         # previous version backup
├── export_onnx.py
├── infer_yolo.py
├── prepare_dataset.py
├── prepare_dataset_digits.py
├── pyproject.toml
├── quantize.py
├── recognize_digits.py
├── requirements.txt
├── train_yolo.py
├── train_yolo_digits.py
├── validate_pipeline.py
├── verify_labels.py
├── README-UKR.md
└── README.md
```

## Scripts Directory

| Script | Purpose |
|---|---|
| `prepare_dataset.py` | Prepares dataset for 1 class (display) for YOLO #1 |
| `train_yolo.py` | Trains YOLO #1 |
| `infer_yolo.py` | Runs YOLO #1, crops input photos into `cropped/` |
| `prepare_dataset_digits.py` | Prepares dataset for 10 classes (digits 0-9) for YOLO #2 |
| `train_yolo_digits.py` | Trains YOLO #2 with latest/bak rotation |
| `verify_labels.py` | Compares YOLO labels with ground truth from .json |
| `recognize_digits.py` | YOLO #2 inference + JSON assembly on a single cropped photo |
| `validate_pipeline.py` | End-to-end evaluation on all photos in a given directory (`img_test/` for a quick check), comparing with ground truth; supports `--backend pt\|onnx\|int8\|int8-display` |
| `export_onnx.py` | Exports `.pt` → `.onnx` (fp32, opset 17) for both models |
| `quantize.py` | Quantizes `.onnx` → `_int8.onnx` (dynamic weight-only int8) |

## Local Development

Note that `img/` contains the author's private photos and is not committed to Git; `img_test/` contains a sample photo with its ground truth so the pipeline can be verified after cloning.

Create environment and install dependencies:

```bash
python -m venv venv

# Linux/macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Run evaluation commands:

```bash
# Check current model performance on test photo (PyTorch)
python validate_pipeline.py img_test

# Check ONNX fp32
python validate_pipeline.py img_test --backend onnx

# Check int8 (browser version)
python validate_pipeline.py img_test --backend int8

# Update ONNX after retraining
python export_onnx.py && python quantize.py
```

Requirements: Python 3.12+, PyTorch (CPU is sufficient), ultralytics, onnx, onnxruntime. Dependencies are listed in `requirements.txt`.

## How It Works — Full Development Cycle

### Initial Training (Done once)

1. Collected 42 monitor photos with manually recorded real values in `.json`.
2. Labeled display bounding boxes on all 42 photos (`labels/labels1/`) → trained YOLO #1.
3. Ran `infer_yolo.py` → produced 42 cropped display images.
4. Labeled digits on 20 out of 42 photos (`labels/labels5/`) → trained YOLO #2.
5. End-to-end check: 42/42 photos parsed correctly.

### Retraining on New Photos

New photos from real users accumulate on the NAS via `aivm-photo-api`. When +N new photos are available (where N is usually 20–50):

```bash
# Linux/macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1

# 1. Copy new photos from NAS into img/
# (new .jpg + .json pairs compatible with aivm-photo-api format)

# 2. Crop displays via YOLO #1
python infer_yolo.py img

# 3. Label new cropped images using https://www.makesense.ai/
#    Save into labels/labelsN/ (new batch, do not overwrite previous)

# 4. Verify labels against ground truth
python verify_labels.py cropped labels/labelsN img

# 5. Re-prepare the dataset
python prepare_dataset_digits.py cropped labels/labelsN dataset_digits

# 6. Training — automatically rotates latest → bak
python train_yolo_digits.py

# 7. Validate on all photos
python validate_pipeline.py img

# 8. If the new model is better than bak — copy best.pt to aivm-photo-api,
#    redeploy the container. Otherwise — revert to bak.
```

## ONNX Export and Quantization

Models are exported to ONNX format for browser execution via `onnxruntime-web` — part of moving OCR processing to the client side.

### ONNX Accuracy Check (42 training photos)

| Backend | End-to-End Check | display_detector | digit_detector | Total Size |
|---|---|---|---|---|
| PyTorch `.pt` | 42/42 (100%) | ~6 MB | ~6 MB | ~12 MB |
| ONNX fp32 | 42/42 (100%) | 11.7 MB | 11.6 MB | 23.3 MB |
| ONNX int8 | 42/42 (100%) | 3.2 MB | 3.1 MB | **6.3 MB** |

Quantization is dynamic (weight-only), requiring no calibration dataset. Accuracy does not drop on the 42 training photos; verify again whenever new photos are added.

### How to Update ONNX Files After Retraining

```bash
# Linux/macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1

# After train_yolo_digits.py (or train_yolo.py):
python export_onnx.py       # best.pt -> best.onnx
python quantize.py          # best.onnx -> best_int8.onnx

# Verify that accuracy did not decrease
python validate_pipeline.py img --backend int8
```

### `--backend` Options in `validate_pipeline.py`

| Option | Models |
|---|---|
| `pt` | both `.pt` (default) |
| `onnx` | both fp32 `.onnx` |
| `int8` | both `_int8.onnx` |
| `int8-display` | display int8 + digit fp32 (for isolated testing) |

## Labeling Procedure

**Website:** https://www.makesense.ai/

1. Get Started → drag and drop photos from `cropped/` (from `cropped`, not `img` — digits are larger and easier to select).
2. Select Object Detection.
3. Create 10 classes: `0`, `1`, `2`, … `9` **in this exact order** (important: class ID must match the digit).
4. Labeling each photo:
   - Draw a bounding box around each digit individually.
   - **Box = "slot"**, not digit contour. Box width and height within the same row should be equal for all digits — `1` uses the same box dimensions as `8`.
   - DO NOT label: `mmHg`, `SYS/DIA/PUL`, icons, color bars, `Expert-X`.
   - SYS — top row (3 digits), DIA — middle row (2), PUL — bottom row (2).
5. Actions → Export Annotations → "A .zip package containing files in YOLO format".
6. Extract into `labels/labelsN/` (create a new folder for each batch, do not overwrite previous ones).

## Architectural Decisions

- **"Slot" Labeling:** The model learns seven-segment digit positions rather than tight contours. This improves detection accuracy for narrow digits (`1`, `7`) next to wide digits (`8`).
- **No Flip/Rotate in YOLO #2:** `2↔5` are mirrored, `6↔9` are inverted. Rotation augmentation is set to ±10° for YOLO #1 only (since cameras may be slightly tilted); YOLO #2 uses no rotations.
- **Class-agnostic NMS:** Removes duplicate overlapping bounding boxes that YOLO might produce when predicting different classes for the same location (e.g., `2` and `3`). Standard YOLO NMS does not remove such pairs.
- **K-means instead of Gap-detection:** Guarantees 3 rows even when Y-coordinates are close. Greedy gap-detection failed on tilted images.
- **digit_detector_bak/:** After each training run, the previous model is renamed rather than deleted. If the new model performs worse, you can revert by renaming it back.

## Common Errors & Troubleshooting

| Type | Symptom | Cause | Solution |
|---|---|---|---|
| 1 | `1233/78/77` | YOLO detects 2 boxes for one digit | NMS (fixed) |
| 2 | `13/77/73` | Missing digit | Add more data |
| 3 | `137/79/73` instead of `137/79/78` | Class confusion (e.g. `8↔3`) | Add more data |
| 4 | `got 2 rows, need 3` | Gap-detection error | K-means (fixed) |

## What NOT to Do

- Do not modify the `cropped/` structure — it serves as input for YOLO #2 and associated labels.
- Do not pass images from `cropped/` through YOLO #1 again.
- Do not train YOLO #2 with flip augmentation.
- Do not mix `labels1` (display) and `labels5` (digits) folders — they represent different tasks.
- Do not delete `digit_detector_bak/` — it serves as a safety fallback.

## Integration with Other System Components

- **aivm-photo-api** — consumer of trained models. When deploying a new model, copy it to [aivm-photo-api](https://github.com/Alexsik76/aivm-photo-api) and redeploy the container.
- **[bptracker-backend-fastapi](https://github.com/Alexsik76/bptracker-backend-fastapi)** — calls aivm-photo-api for recognition and receives JSON.
- **Project plan** — see [bp-ocr-cnn_PLAN.md](docs/bp-ocr-cnn_PLAN.md) for roadmap details.

## Future Plans

See [bp-ocr-cnn_PLAN.md](docs/bp-ocr-cnn_PLAN.md) — tasks for model training, extra annotations, and architecture experiments.
