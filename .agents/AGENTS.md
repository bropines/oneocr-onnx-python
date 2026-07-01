# OneOCR Deobfuscated Project Rules & Context

This file provides context and rules for AI agents working on the **OneOCR Deobfuscated** repository. Read this before proposing modifications or analyzing the codebase.

## Repository Overview

This project is a clean, modular Python wrapper and inference pipeline for the Windows 11 Snipping Tool OCR engine (`oneocr.onemodel`).

### Core Components

1. **Extraction & Decryption (`oneocr/extractor.py`)**:
   * Intercepts `onnxruntime.dll` API table in-memory (using `ctypes` and memory patching) at the moment `oneocr.dll` decrypts and loads the models.
   * Pulls the raw decrypted ONNX model files and automatically reconstructs the language vocabularies using a feedback loop running OCR on synthetic text layouts.
   * **Note**: Preparation/extraction requires a Windows machine as it relies on Windows DLL APIs.

2. **Python OCR Engine (`oneocr/engine.py` & package)**:
   * **Pure Python & ONNX**: Once models are extracted, the OCR engine runs on Windows, macOS, Linux, and Docker using only standard libraries (`onnxruntime`, `pillow`, `numpy`).
   * **Pipeline steps**:
     1. `OrientationCorrector`: Estimates page angle (0, 90, 180, 270) using the detector and classifier, and rotates the image upright.
     2. `TextDetector`: Finds bounding boxes of text lines using an FPN (Feature Pyramid Network) model.
     3. `ScriptClassifier`: Determines script class (10 classes: CJK, Latin, Cyrillic, etc.) and text line orientation.
     4. `TextRecognizer`: Executes CTC-based text recognition for the corresponding language script and decodes character sequences.

3. **Data Structures (`oneocr/common.py`)**:
   * Uses python `dataclasses`: `Quad` (4-point bounding box), `Word`, `Line`, and `OcrResult`.

---

## Behavioral Rules for Agents

### 1. Verification & Commands
* **No Unnecessary Testing**: Do not execute OCR test scripts or run Python tests (`test_real_image.py`, etc.) unless specifically asked to do so by the user.
* **GPU Providers**: The project supports CUDA, DirectML, and OpenVINO acceleration. When modifying the engine constructor, respect the order of execution provider evaluation in `oneocr/engine.py`.

### 2. Code Modifications
* **Keep Dependencies Minimal**: The core `oneocr` inference package should only depend on `numpy`, `pillow`, and `onnxruntime`. Do not add heavy dependencies.
* **Keep Code Cross-Platform**: Ensure all changes inside `oneocr/engine.py`, `detector.py`, `classifier.py`, `recognizer.py`, `corrector.py`, and `common.py` remain fully cross-platform. Platform-specific Windows APIs are strictly limited to `downloader.py` and `extractor.py`.
* **Clean Typings**: Use Python 3.10+ type annotations, dataclasses, and standard Python conventions.

### 3. File Paths
* Use `Path` objects from `pathlib` for file system interactions.
* Resolved model path targets:
  * Detector: `models/detector/text_detector.onnx`
  * Classifier: `models/classifier/script_classifier.onnx`
  * Recognizers: `models/recognizers/recognizer_<lang_name>.onnx`
  * Vocabularies: `models/vocab/vocab_<lang_name>.txt`
