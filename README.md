# OneOCR ONNX Python

A clean, modular Python runtime pipeline and wrapper for the **Windows 11 Snipping Tool** OCR engine (`oneocr.onemodel`).

> **Note**: The DLL binaries and model files are Microsoft intellectual property.  
> This repository contains only the extraction tooling — the binaries are fetched
> from the official Microsoft Store at runtime and never redistributed.

---

## Quick Start

```bash
# 1. Run the automated script to download, extract, decrypt, and organize everything:
python prepare_files.py

# 2. Run OCR on any image using the pure Python/ONNX pipeline:
python tests/test_real_image.py path/to/screenshot.png
```

For detailed documentation on configuration, hardware acceleration, and the full Python class interface, see [api.md](api.md).

> [!IMPORTANT]
> **Platform Requirements**:
> * **Preparation & Decryption**: The extraction and decryption step (`prepare_files.py`) utilizes Windows APIs and native DLLs, meaning it **must be run on a Windows machine**.
> * **OCR Engine & Inference**: Once the sub-models are extracted into the `models/` folder, they are standard platform-independent ONNX files. The core `oneocr` Python package runs on **Windows, macOS, Linux, and Docker** using only standard Python libraries and `onnxruntime`.

Or use the library directly in your own code:

```python
from oneocr import OneOCR

with OneOCR() as ocr:
    result = ocr.recognize_file("screenshot.png")
    print(result.full_text)
    print(f"Image angle: {result.image_angle}°")

    for line in result.lines:
        print(f"\n[Line] {line.text}")
        x, y, w, h = line.bbox.as_rect()
        print(f"  Position: x={x:.0f} y={y:.0f} w={w:.0f} h={h:.0f}")

        for word in line.words:
            print(f"  '{word.text}'  conf={word.confidence:.3f}")
```

---

## File Structure

```
OneOCR_Deobfuscated/
│
├── oneocr/                            ← Python package (Modular, pure ONNX Runtime)
│   ├── __init__.py                    ← Package entry point
│   ├── common.py                      ← Quad, Word, Line, OcrResult classes
│   ├── vocab.py                       ← Vocabulary loading
│   ├── detector.py                    ← TextDetector class
│   ├── classifier.py                  ← ScriptClassifier class
│   ├── recognizer.py                  ← TextRecognizer class
│   └── corrector.py                   ← OrientationCorrector class
│
├── download_and_extract_oneocr.py     ← Downloads + extracts DLLs from Microsoft Store
├── extract_all.py                     ← Automated memory patch decryptor & vocab extractor
├── test_real_image.py                 ← Example OCR test script
├── README.md                          ← This file
│
├── bin/                               ← Raw DLL binaries (gitignored)
│   ├── oneocr.dll
│   ├── oneocr.onemodel                ← Encrypted model container
│   └── onnxruntime.dll
│
└── models/                            ← Decrypted & organized ONNX sub-models
    ├── detector/
    │   └── text_detector.onnx         ← FPN text region detector (~11 MB)
    ├── classifier/
    │   └── script_classifier.onnx     ← Script/language/orientation classifier (~3.3 MB)
    └── recognizers/                   ← 9 CTC text recognizers (named by script)
        ├── recognizer_cjk.onnx        ← Chinese / Japanese / Korean (~12.8 MB)
        ├── recognizer_cyrillic.onnx   ← Cyrillic + extended Latin (~2.0 MB)
        ├── recognizer_latin.onnx      ← Extended Latin block (~6.3 MB)
        ├── recognizer_arabic.onnx     ← (~3.3 MB)
        ├── recognizer_devanagari.onnx ← (~3.5 MB)
        ├── recognizer_hebrew.onnx     ← (~3.5 MB)
        ├── recognizer_thai.onnx       ← (~3.5 MB)
        ├── recognizer_bengali.onnx    ← (~1.7 MB)
        └── recognizer_greek.onnx      ← (~1.9 MB)
```

---

## Pipeline Architecture

```
                ┌──────────────────────────────────────────┐
                │          oneocr Package Pipeline         │
                │                                          │
                │  1. text_detector.onnx                   │
                │     FPN detector → bounding quads        │
                │                   │                      │
                │  2. script_classifier.onnx               │
                │     Per-crop: script ID (10 classes)     │
                │              orientation / flip          │
                │                   │                      │
                │  3. recognizer_<lang>.onnx  (CRNN+CTC)   │
                │     Input: [1, 3, 60, W]                 │
                │     Output: logsoftmax over vocab        │
                │                   │                      │
                │  4. CTC Decode & Word alignment (Python) │
                └────────────────────┬─────────────────────┘
                                     │
                             Lines + Words + BBox
```

### Model roles

| Model      | File                       |   Vocab | Role                        |
| ---------- | -------------------------- | ------: | --------------------------- |
| Detector   | `text_detector.onnx`       |       — | Finds text regions via FPN  |
| Classifier | `script_classifier.onnx`   |      10 | Script type + orientation   |
| CJK        | `recognizer_cjk.onnx`      |  32 632 | Chinese / Japanese / Korean |
| Cyrillic   | `recognizer_cyrillic.onnx` |     548 | Cyrillic + extended         |
| Latin+     | `recognizer_latin.onnx`    |     415 | Extended Latin block        |
| Others     | `recognizer_*.onnx`        | 179–244 | Smaller script families     |

---

## Library API

### `OneOCR(config_dir=None, max_lines=1000)`

Main engine class. Auto-detects `models/` or `~/.config/oneocr/`.

```python
engine = OneOCR()                        # auto-detect
engine = OneOCR(config_dir="path/to")   # explicit path
engine = OneOCR(max_lines=50)            # limit output lines
```

### `engine.recognize(image: PIL.Image) → OcrResult`

Run OCR on a PIL Image object (any mode, any size 50–10 000 px).

### `engine.recognize_file(path) → OcrResult`

Convenience wrapper that opens the file and calls `recognize()`.

### `OcrResult`

| Field          | Type             | Description               |
| -------------- | ---------------- | ------------------------- |
| `.lines`       | `list[Line]`     | All recognized text lines |
| `.image_angle` | `float`          | Page rotation in degrees  |
| `.full_text`   | `str` (property) | All lines joined by `\n`  |

### `Line`

| Field    | Type         | Description                  |
| -------- | ------------ | ---------------------------- |
| `.text`  | `str`        | Full text of the line        |
| `.bbox`  | `Quad`       | Quadrilateral bounding box   |
| `.style` | `int`        | 0 = horizontal, 1 = vertical |
| `.words` | `list[Word]` | Individual words             |

### `Word`

| Field         | Type    | Description                |
| ------------- | ------- | -------------------------- |
| `.text`       | `str`   | Word text                  |
| `.bbox`       | `Quad`  | Quadrilateral bounding box |
| `.confidence` | `float` | Recognition confidence 0–1 |

### `Quad`

Four corner points `(x1,y1)…(x4,y4)` clockwise from top-left.

```python
quad.as_rect()  # → (x, y, width, height)  axis-aligned AABB
```

---

## How the decryption works

`oneocr.onemodel` is an encrypted container. The DLL decrypts it internally using a hardcoded key before loading each sub-model via ONNX Runtime's `CreateSessionFromArray` C API. For a detailed step-by-step reverse engineering analysis of how this memory hooking and decryption works, see [WRITEUP.md](WRITEUP.md).

`extract_all.py` intercepts this at runtime by:

1. Loading `onnxruntime.dll` into the Python process.
2. Patching the API table of ONNX Runtime in-memory to hook `CreateSessionFromArray` and `Run`.
3. Loading `oneocr.dll` to trigger model decryption.
4. Saving the decrypted model buffers and automatically reconstructing vocabulary files.

This technique requires no manual disassembly and works against future updates.

---

## References & Prior Work

- [b1tg's Research Post on Win11 OneOCR](https://b1tg.github.io/post/win11-oneocr) — Detailed reverse engineering writeup of the OneOCR DLL engine.
- [b1tg's win11-oneocr Repository](https://github.com/b1tg/win11-oneocr) — Original C++ proof-of-concept DLL hooking and extraction codebase.
- [AuroraWright's oneocr Wrapper](https://github.com/AuroraWright/oneocr) — Python-based OneOCR pipeline loader and model wrapper.

---

## License & Legal Disclaimer

* **Software License**: The Python codebase, scripts, and documentation in this repository are licensed under the [MIT License](LICENSE).
* **Legal Disclaimer**: This project is created for educational and research purposes only. This repository **does not host or distribute** any Microsoft proprietary binaries, DLLs, or neural network models. All proprietary assets must be acquired and decrypted locally by the end-user. All trademarks and technologies remain the property of their respective owners.


