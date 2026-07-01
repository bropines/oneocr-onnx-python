from dataclasses import dataclass, field
from typing import Optional

SCRIPT_METADATA = {
    1: {"name": "cjk",        "vocab_size": 32632},
    2: {"name": "cyrillic",   "vocab_size": 548},
    3: {"name": "latin",      "vocab_size": 415},
    4: {"name": "arabic",     "vocab_size": 221},
    5: {"name": "devanagari", "vocab_size": 237},
    6: {"name": "hebrew",     "vocab_size": 244},
    7: {"name": "thai",       "vocab_size": 199},
    8: {"name": "bengali",    "vocab_size": 201},
    9: {"name": "greek",      "vocab_size": 179}
}

SCRIPT_BY_NAME = {info["name"]: (script_id - 1, info["vocab_size"]) for script_id, info in SCRIPT_METADATA.items()}

LANG_TO_SCRIPT = {
    # CJK
    "zh": "cjk", "cn": "cjk", "tw": "cjk", "hk": "cjk", "ja": "cjk", "jp": "cjk", "ko": "cjk", "kr": "cjk",
    "chinese": "cjk", "japanese": "cjk", "korean": "cjk",
    # Cyrillic
    "ru": "cyrillic", "bg": "cyrillic", "uk": "cyrillic", "be": "cyrillic", "sr": "cyrillic", "mk": "cyrillic",
    "kk": "cyrillic", "ky": "cyrillic", "tg": "cyrillic", "mn": "cyrillic", "russian": "cyrillic", "bulgarian": "cyrillic",
    "ukrainian": "cyrillic", "belarussian": "cyrillic", "serbian": "cyrillic", "macedonian": "cyrillic",
    # Latin
    "en": "latin", "de": "latin", "fr": "latin", "es": "latin", "it": "latin", "pt": "latin", "pl": "latin",
    "tr": "latin", "vi": "latin", "nl": "latin", "sv": "latin", "no": "latin", "da": "latin", "fi": "latin",
    "cs": "latin", "sk": "latin", "hu": "latin", "ro": "latin", "ca": "latin", "hr": "latin", "id": "latin",
    "ms": "latin", "english": "latin", "german": "latin", "french": "latin", "spanish": "latin", "italian": "latin",
    "portuguese": "latin", "polish": "latin", "turkish": "latin", "vietnamese": "latin", "dutch": "latin",
    "swedish": "latin", "norwegian": "latin", "danish": "latin", "finnish": "latin", "czech": "latin",
    "slovak": "latin", "hungarian": "latin", "romanian": "latin", "catalan": "latin", "croatian": "latin",
    "indonesian": "latin", "malay": "latin",
    # Arabic
    "ar": "arabic", "fa": "arabic", "ur": "arabic", "ps": "arabic", "ku": "arabic", "arabic": "arabic",
    "persian": "arabic", "urdu": "arabic", "pashto": "arabic", "kurdish": "arabic",
    # Hebrew
    "he": "hebrew", "iw": "hebrew", "yi": "hebrew", "hebrew": "hebrew", "yiddish": "hebrew",
    # Thai
    "th": "thai", "thai": "thai",
    # Devanagari
    "hi": "devanagari", "mr": "devanagari", "ne": "devanagari", "sa": "devanagari", "hindi": "devanagari",
    "marathi": "devanagari", "nepali": "devanagari", "sanskrit": "devanagari",
    # Bengali
    "bn": "bengali", "as": "bengali", "bengali": "bengali", "assamese": "bengali",
    # Greek
    "el": "greek", "gr": "greek", "greek": "greek"
}

@dataclass
class Quad:
    x1: float; y1: float
    x2: float; y2: float
    x3: float; y3: float
    x4: float; y4: float

    def as_rect(self) -> tuple[float, float, float, float]:
        xs = (self.x1, self.x2, self.x3, self.x4)
        ys = (self.y1, self.y2, self.y3, self.y4)
        x, y = min(xs), min(ys)
        return x, y, max(xs) - x, max(ys) - y

    def __repr__(self) -> str:
        return (f"Quad(({self.x1:.0f},{self.y1:.0f}) "
                f"({self.x2:.0f},{self.y2:.0f}) "
                f"({self.x3:.0f},{self.y3:.0f}) "
                f"({self.x4:.0f},{self.y4:.0f}))")

@dataclass
class Word:
    text: str
    bbox: Optional[Quad]
    confidence: float

@dataclass
class Line:
    text: str
    bbox: Optional[Quad]
    style: int
    words: list[Word] = field(default_factory=list)

@dataclass
class OcrResult:
    lines: list[Line]
    image_angle: float

    @property
    def full_text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)
