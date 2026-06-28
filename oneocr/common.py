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
