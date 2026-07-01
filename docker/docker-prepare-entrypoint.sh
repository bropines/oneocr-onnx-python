#!/bin/bash
set -e

OUTPUT_DIR="/output"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  OneOCR Model Preparation (Wine + Docker)   "
echo "=============================================="

# Start virtual display (Wine needs it on headless systems)
Xvfb :0 -screen 0 1024x768x16 &
export DISPLAY=:0
sleep 1

echo ""
echo "[1/2] Downloading Snipping Tool package from Microsoft Store..."
echo "    (This step uses native Python - no Wine needed)"
python3 utils/download_and_extract_oneocr.py

echo ""
echo "[2/2] Decrypting ONNX models via Windows DLL interception (Wine)..."
wine python utils/extract_all.py

echo ""
echo "Copying organized models to output directory..."
cp -r /app/models/detector "$OUTPUT_DIR/" 2>/dev/null || true
cp -r /app/models/classifier "$OUTPUT_DIR/" 2>/dev/null || true
cp -r /app/models/recognizers "$OUTPUT_DIR/" 2>/dev/null || true
cp -r /app/models/vocab "$OUTPUT_DIR/" 2>/dev/null || true

echo ""
echo "=============================================="
echo "SUCCESS! Decrypted models are saved in: $OUTPUT_DIR"
echo ""
echo "Contents:"
find "$OUTPUT_DIR" -type f | sort
echo "=============================================="
