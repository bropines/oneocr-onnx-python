# /// script
# dependencies = [
#   "requests>=2.28.0",
#   "packaging>=23.0",
# ]
# ///
import sys
from oneocr import download_and_extract

def main():
    try:
        download_and_extract()
        print("\nAll done! You can now run the decryption script.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
