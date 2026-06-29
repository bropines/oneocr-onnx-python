from .common import Quad, Word, Line, OcrResult
from .engine import OneOCR
from .downloader import download_and_extract
from .extractor import decrypt_and_extract

def prepare(bin_dir=None, models_dir=None, target_arch="x64"):
    """Unified setup helper to download, extract, decrypt, and sort OneOCR models."""
    print("Initializing OneOCR preparation...")
    # 1. Download and extract Microsoft Store package
    final_bin_dir = download_and_extract(output_dir=bin_dir, target_arch=target_arch)
    # 2. Decrypt ONNX models and extract character vocabularies
    decrypt_and_extract(bin_dir=final_bin_dir, models_dir=models_dir)
