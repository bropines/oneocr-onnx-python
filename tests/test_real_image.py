# /// script
# dependencies = [
#   "onnxruntime>=1.16.0",
#   "numpy>=1.20.0",
#   "pillow>=9.0.0",
# ]
# ///
"""
Example: run OneOCR on a real screenshot using the oneocr.py wrapper.
Usage:
    python test_real_image.py [image_path]
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Import the local wrapper (same directory)
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from oneocr import OneOCR

IMAGE_PATH = (sys.argv[1] if len(sys.argv) > 1
              else r"F:\bropi\Documents\ShareX\Screenshots\2026-06\tailscale-ipn_uVzgDard51.png")

print(f"Running OneOCR on: {IMAGE_PATH}\n")

with OneOCR() as ocr:
    result = ocr.recognize_file(IMAGE_PATH)

print(f"Image angle : {result.image_angle:.1f}°")
print(f"Lines found : {len(result.lines)}")
print("=" * 60)

for i, line in enumerate(result.lines, 1):
    x, y, w, h = line.bbox.as_rect() if line.bbox else (0, 0, 0, 0)
    print(f"\nLine {i}: \"{line.text}\"")
    print(f"  Position : x={x:.0f} y={y:.0f}  size={w:.0f}×{h:.0f}")

    for word in line.words:
        wx, wy, ww, wh = word.bbox.as_rect() if word.bbox else (0, 0, 0, 0)
        print(f"  '{word.text}'  "
              f"x={wx:.0f} y={wy:.0f}  "
              f"conf={word.confidence:.3f}")

print("\n" + "=" * 60)
print("FULL TEXT:")
print(result.full_text)
