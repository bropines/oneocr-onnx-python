import sys
from pathlib import Path
from oneocr import prepare

def main():
    print("=" * 80)
    print("OneOCR Automated Preparation Script")
    print("=" * 80)
    
    try:
        prepare()
    except Exception as e:
        print(f"\n[ERROR] Preparation failed: {e}")
        sys.exit(1)
        
    # Final Summary
    base_dir = Path(__file__).parent.parent.absolute()
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
