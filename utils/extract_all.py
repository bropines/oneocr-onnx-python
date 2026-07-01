# /// script
# dependencies = [
#   "onnx>=1.14.0",
#   "onnxruntime>=1.16.0",
#   "numpy>=1.20.0",
#   "pillow>=9.0.0",
# ]
# ///
import sys
from oneocr import decrypt_and_extract

def main():
    try:
        decrypt_and_extract()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
