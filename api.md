# OneOCR Python API & Execution Providers Guide

This document outlines the Python API for the `oneocr` library and describes how to configure different hardware acceleration backends (CPU, NVIDIA GPU/CUDA, AMD/Intel/NVIDIA DirectML, etc.) using ONNX Runtime.

> [!IMPORTANT]
> **Platform Requirements**:
> * **Preparation & Decryption** (`prepare()`, `download_and_extract()`, `decrypt_and_extract()`): Must be executed on a **Windows** system because they call Windows APIs and execute native `.dll` binaries.
> * **OCR Engine & Inference** (`OneOCR`): Once the sub-models are extracted into the `models/` folder, they are standard ONNX networks. The main `OneOCR` engine runs on **Windows, macOS, Linux, and Docker** without any OS-specific dependencies.

---

## 1. Installation Requirements

Depending on your target hardware, install **one** of the following `onnxruntime` packages:

```bash
# For CPU execution (default)
pip install onnxruntime pillow numpy

# For Windows Hardware Acceleration (DirectML - works on AMD, Intel, NVIDIA)
pip install onnxruntime-directml pillow numpy

# For NVIDIA GPU Acceleration (CUDA - requires CUDA and cuDNN installed)
pip install onnxruntime-gpu pillow numpy
```

---

## 2. API Reference

The main entry point is the `OneOCR` class, which manages the pipeline of text detection, classification, character recognition, and orientation correction.

### `OneOCR` Class

```python
from oneocr import OneOCR

# Standard instantiation (auto-detects models directory in project root)
ocr = OneOCR()
```

#### Constructor Arguments:
*   `config_dir` (`str | Path`, optional): Custom directory containing the `models/` directory. Defaults to checking the project root, then `~/.config/oneocr/`.
*   `max_lines` (`int`, optional): Maximum number of text lines to process per image. Defaults to `1000`.
*   `use_gpu` (`bool`, optional): If `True`, automatically searches for and enables available GPU/acceleration execution providers (CUDA, DirectML, ROCm) in order of preference, falling back to CPU. Defaults to `False`.
*   `providers` (`list[str]`, optional): Explicit list of ONNX Runtime execution providers to use. Overrides `use_gpu`.
*   `default_language` (`str`, optional): Default language code (e.g. `'ru'`, `'en'`) or script name (e.g. `'cyrillic'`, `'latin'`). Specifying this bypasses the script classifier to improve performance. Defaults to `None`.
*   `default_rotation` (`int`, optional): Default fixed rotation angle (`0`, `90`, `180`, `270`) to skip auto-orientation. Defaults to `None`.
*   `max_side` (`int`, optional): Resize image's longest side to this limit for detection. Defaults to `1536`.
*   `score_threshold` (`float`, optional): Text line detection confidence threshold. Defaults to `0.5`.
*   `link_threshold` (`float`, optional): Text line character linkage threshold. Defaults to `0.0`.

#### `ocr.recognize(...)` & `ocr.recognize_file(...)` Arguments:
You can pass the following optional parameters to override the default settings per call:
*   `language` (`str`, optional): Specific language code or script group name. Bypasses the script classifier.
*   `rotation` (`int`, optional): Explicit rotation angle (`0`, `90`, `180`, `270`). Bypasses orientation detection.
*   `max_side` (`int`, optional): Overrides image resizing limit.
*   `score_threshold` (`float`, optional): Overrides line detection confidence threshold.
*   `link_threshold` (`float`, optional): Overrides line linkage threshold.

---

### Basic Usage Example

```python
from oneocr import OneOCR
from PIL import Image

# 1. Initialize the engine with custom defaults
with OneOCR(default_language="ru", default_rotation=0) as ocr:
    # 2. Perform OCR on an image file overriding max_side limit
    result = ocr.recognize_file("screenshot.png", max_side=2048)
    
    # 3. Access full extracted text
    print("--- Extracted Text ---")
    print(result.full_text)
    
    # 4. Access individual layout lines and words
    print("\n--- Detailed Layout Analysis ---")
    print(f"Detected rotation angle: {result.image_angle}°")
    
    for line in result.lines:
        print(f"\n[Line] '{line.text}'")
        x, y, w, h = line.bbox.as_rect()
        print(f"  Position: x={x:.0f}, y={y:.0f}, width={w:.0f}, height={h:.0f}")
```


---

### Programmatic Model Preparation API

You can programmatically trigger the download, extraction, and decryption of the models directly inside your Python code using the library's built-in preparation helpers.

```python
import oneocr

# 1. Run the entire preparation process sequentially (downloads, extracts, decrypts, and organizes)
oneocr.prepare(bin_dir="my_bin", models_dir="my_models")

# 2. Or call the two steps individually if you need custom handling:

# Step A: Download and extract Microsoft DLLs + encrypted container from MS Store
oneocr.download_and_extract(output_dir="my_bin", target_arch="x64")

# Step B: Hook and decrypt the container, extract vocabs, and sort the models
oneocr.decrypt_and_extract(bin_dir="my_bin", models_dir="my_models")
```

---

## 3. Data Structures

The OCR results are structured into the following dataclasses:

### `OcrResult`
*   `.lines` (`list[Line]`): List of detected text lines, ordered logically by layout (top-to-bottom for horizontal, right-to-left for vertical CJK text).
*   `.image_angle` (`float`): Auto-detected image rotation angle (`0.0`, `90.0`, `180.0`, or `270.0` degrees).
*   `.full_text` (`str`, property): All lines combined together into a single string, using appropriate separators (spaces for Latin/Cyrillic, empty string for CJK).

### `Line`
*   `.text` (`str`): The full recognized text of the line.
*   `.bbox` (`Quad`): The 4-point bounding quad in the original image coordinate space.
*   `.style` (`int`): Text layout direction (`0` for horizontal, `1` for vertical).
*   `.words` (`list[Word]`): Individual words within this line.

### `Word`
*   `.text` (`str`): The word text.
*   `.bbox` (`Quad`): Bounding quad of the word.
*   `.confidence` (`float`): Recognition confidence score between `0.0` and `1.0`.

### `Quad`
*   `.x1`, `.y1`, `.x2`, `.y2`, `.x3`, `.y3`, `.x4`, `.y4`: Clockwise corner coordinates starting from top-left.
*   `as_rect() -> (x, y, width, height)`: Returns the axis-aligned bounding box (AABB) of the quad.

---

## 4. Hardware Acceleration (ONNX Runtime Providers)

You can customize execution provider settings for different devices:

### CPU-only (Standard)
```python
# Runs strictly on CPU
ocr = OneOCR(use_gpu=False)
```

### Auto GPU Detection
```python
# Automatically picks the best available GPU provider (CUDA, DirectML, ROCm)
# and falls back to CPU if none are found.
ocr = OneOCR(use_gpu=True)
```

### Explicit Provider Selection
If you have multiple providers or want to enforce a specific hardware backend, use the `providers` list parameter:

#### DirectML (Recommended for Windows AMD/Intel/NVIDIA)
DirectML runs on DirectX 12, allowing hardware acceleration on almost any modern Windows GPU without installing complex CUDA toolkits.
```python
ocr = OneOCR(providers=["DmlExecutionProvider", "CPUExecutionProvider"])
```

#### CUDA (Recommended for Linux/NVIDIA servers)
```python
ocr = OneOCR(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

#### OpenVINO (Optimized for Intel CPUs/iGPUs)
```python
# Requires pip install onnxruntime-openvino
ocr = OneOCR(providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"])
```

---

## 5. Command Line Interface (CLI)

You can run OCR directly from the command line using `python -m oneocr`.

### Basic Usage
```bash
python -m oneocr path/to/image.png
```

### Advanced Arguments:
*   `-l`, `--lang` / `--language`: Force specific language or script (e.g. `ru`, `en`, `cyrillic`, `latin`) to bypass classification.
*   `-r`, `--rotation`: Force fixed rotation angle (`0`, `90`, `180`, `270`) to bypass auto-rotation estimation.
*   `-o`, `--output`: Write OCR results to a file instead of stdout.
*   `--gpu`: Force enable GPU acceleration.
*   `--max-side`: Max image dimensions for detection (default: 1536).
*   `--score-threshold`: Set custom score threshold (default: 0.5).
*   `--link-threshold`: Set custom link threshold (default: 0.0).
*   `--json`: Output results in full JSON format (includes word-level bounding boxes and confidences).

#### Examples:
```bash
# Force Russian recognition on an unrotated image:
python -m oneocr screenshot.png -l ru -r 0

# Extract details in JSON format and save to out.json:
python -m oneocr screenshot.png -l en -r 0 --json -o out.json
```
