# /// script
# dependencies = [
#   "onnxruntime>=1.16.0",
#   "numpy>=1.20.0",
#   "pillow>=9.0.0",
# ]
# ///
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from oneocr import OneOCR

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_mask.py <image_path>")
        sys.exit(1)
        
    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"File not found: {img_path}")
        sys.exit(1)
        
    print(f"Extracting text mask for: {img_path}...")
    with OneOCR() as ocr:
        img = Image.open(img_path)
        mask = ocr.get_text_mask(img)
        
        output_path = img_path.parent / f"mask_{img_path.stem}.png"
        mask.save(output_path)
        print(f"Text mask saved to: {output_path}")

if __name__ == "__main__":
    main()
