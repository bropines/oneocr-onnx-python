import os
from pathlib import Path

class Vocabulary:
    """Manages loading and indexing of script vocabularies."""
    @staticmethod
    def load_clean(vocab_path: Path) -> dict[int, str]:
        vocab = {}
        if not vocab_path.exists():
            return vocab
        raw_list = []
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split(":", 1)
                if len(parts) == 2:
                    idx = int(parts[0])
                    char = parts[1]
                    if char.startswith(" "):
                        char = char[1:]
                    raw_list.append((idx, char))
        
        raw_list.sort(key=lambda x: x[0])
        
        # Bypass clean mapping for CJK (large vocabulary)
        if len(raw_list) > 1000:
            for idx, char in raw_list:
                vocab[idx] = char
            return vocab
            
        # Omit blank and separator tokens for exact alignment in smaller scripts
        clean_idx = 1
        for idx, char in raw_list:
            if idx in [0, 13, 16, 28, 33]:
                continue
            vocab[clean_idx] = char
            clean_idx += 1
        return vocab
