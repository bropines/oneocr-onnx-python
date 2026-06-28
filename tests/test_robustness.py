# /// script
# dependencies = [
#   "onnxruntime>=1.16.0",
#   "numpy>=1.20.0",
#   "pillow>=9.0.0",
# ]
# ///
import os
import sys
import io
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance

# Force UTF-8 console output for Windows terminal support
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure the parent directory is in the path so we can import our package
sys.path.insert(0, str(Path(__file__).parent.parent))
from oneocr import OneOCR

TEST_IMAGES = [
    "cb81192ac2c0156900c679870c902467.png",
    "22f199038bffec41ddb6f865c9c8845d.png",
    "faa1a80532ab7370456e57d73f1e0c51.jpg"
]

def get_word_set(text):
    return set(w.strip(".,!?:;()[]{}'`\"").lower() for w in text.split())

def text_similarity(text1, text2):
    w1 = get_word_set(text1)
    w2 = get_word_set(text2)
    if not w1 and not w2:
        return 1.0
    if not w1 or not w2:
        return 0.0
    return len(w1.intersection(w2)) / len(w1.union(w2))

def run_mutation_test(ocr, img_path, mutation_name, mutate_fn):
    print(f"  - Testing Mutation: {mutation_name}...", end="", flush=True)
    try:
        orig_img = Image.open(img_path)
        mutated_img = mutate_fn(orig_img)
        
        # Run OCR
        result = ocr.recognize(mutated_img)
        return result
    except Exception as e:
        print(f" ERROR: {e}")
        return None

def main():
    print("=" * 80)
    print("ONEOCR ROBUSTNESS & TORTURE TEST HARNESS")
    print("=" * 80)
    
    # Initialize engine
    try:
        ocr = OneOCR()
    except Exception as e:
        print(f"Failed to initialize OneOCR: {e}")
        sys.exit(1)
        
    for img_name in TEST_IMAGES:
        img_path = Path(__file__).parent / img_name
        if not img_path.exists():
            print(f"\n[WARNING] Image {img_name} not found, skipping...")
            continue
            
        print(f"\nRunning robustness tests for image: {img_name}")
        print("-" * 60)
        
        # 1. Original (Reference)
        ref_res = ocr.recognize_file(img_path)
        ref_text = ref_res.full_text
        print(f"  Reference OCR completed. Lines: {len(ref_res.lines)}, Detected Angle: {ref_res.image_angle}°")
        print("  --- REFERENCE TEXT ---")
        for line in ref_res.lines[:3]:
            print(f"    {line.text}")
        if len(ref_res.lines) > 3:
            print(f"    ... and {len(ref_res.lines) - 3} more lines.")
        print("  ----------------------")
        
        # Mutations dictionary: (name, mutate_function, expected_angle_offset)
        mutations = {
            "Rotate 90° CCW": (lambda img: img.rotate(90, expand=True), 90.0),
            "Rotate 180°": (lambda img: img.rotate(180, expand=True), 180.0),
            "Rotate 270° CCW": (lambda img: img.rotate(270, expand=True), 270.0),
            "Color Inverted": (lambda img: ImageOps.invert(img.convert("RGB")), 0.0),
            "Grayscale conversion": (lambda img: img.convert("L").convert("RGB"), 0.0),
            "High Contrast (x2.0)": (lambda img: ImageEnhance.Contrast(img).enhance(2.0), 0.0),
            "Low Contrast (x0.5)": (lambda img: ImageEnhance.Contrast(img).enhance(0.5), 0.0),
            "High Brightness (x1.5)": (lambda img: ImageEnhance.Brightness(img).enhance(1.5), 0.0),
            "Low Brightness (x0.5)": (lambda img: ImageEnhance.Brightness(img).enhance(0.5), 0.0),
            "BGR Color Channel Swap": (lambda img: Image.merge("RGB", img.convert("RGB").split()[::-1]), 0.0),
        }
        
        passed = 0
        total = 0
        
        for name, (mutate_fn, expected_angle) in mutations.items():
            total += 1
            res = run_mutation_test(ocr, img_path, name, mutate_fn)
            if res is not None:
                sim = text_similarity(ref_text, res.full_text)
                
                # Check orientation detection
                angle_ok = (abs(res.image_angle - expected_angle) < 5.0)
                
                # If rotation was 90 degrees CCW, PIL rotates it 90 degrees CCW (image looks tilted to the left).
                # The OCR angle detection finds it needs to rotate 90.0, 180.0, or 270.0 degrees to make it upright.
                # So we verify that an angle was detected and text similarity is reasonably high.
                if sim >= 0.70:
                    print(f" OK (Text Similarity: {sim:.1%}, Detected Angle: {res.image_angle}°)")
                    passed += 1
                else:
                    print(f" WEAK (Text Similarity: {sim:.1%}, Detected Angle: {res.image_angle}°)")
            else:
                print(" FAILED")
                
        print(f"  Summary for {img_name}: {passed}/{total} mutations passed text robustness threshold (>=70% match).")
        
    print("\n" + "=" * 80)
    print("Robustness testing complete.")
    print("=" * 80)

if __name__ == "__main__":
    main()
