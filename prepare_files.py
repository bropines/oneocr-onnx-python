import sys
import os
from pathlib import Path

def main():
    print("=" * 80)
    print("OneOCR Automated Preparation Script")
    print("=" * 80)
    
    # 1. Import and run download/extraction script
    print("\n--- STEP 1: Downloading & extracting Snipping Tool OCR components ---")
    try:
        import download_and_extract_oneocr
        download_and_extract_oneocr.main()
    except Exception as e:
        print(f"\n[ERROR] Step 1 failed: {e}")
        sys.exit(1)
        
    # 2. Import and run decryption/vocabulary extraction script
    print("\n--- STEP 2: Decrypting ONNX models & reconstructing vocabularies ---")
    try:
        import extract_all
        extract_all.main()
    except Exception as e:
        print(f"\n[ERROR] Step 2 failed: {e}")
        sys.exit(1)
        
    # 3. Final Summary
    base_dir = Path(__file__).parent.absolute()
    print("\n" + "=" * 80)
    print("PREPARATION SUCCESSFUL!")
    print("=" * 80)
    print(f"All files have been successfully processed and placed in the following folders:")
    print(f"  • Microsoft DLLs & container (Git-ignored):")
    print(f"    - {base_dir / 'bin'}")
    print(f"  • Decrypted ONNX models (Git-ignored):")
    print(f"    - {base_dir / 'models' / 'detector'}")
    print(f"    - {base_dir / 'models' / 'classifier'}")
    print(f"    - {base_dir / 'models' / 'recognizers'}")
    print(f"  • Reconstructed vocabularies:")
    print(f"    - {base_dir / 'models' / 'vocab'}")
    print(f"  • In-memory vocabulary bin buffers (Git-ignored):")
    print(f"    - {base_dir / 'vocab_buffers'}")
    print("\nYou can now run 'python tests/test_real_image.py' to verify the installation.")
    print("=" * 80)

if __name__ == '__main__':
    main()
