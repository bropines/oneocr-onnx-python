import os
from pathlib import Path
from PIL import Image
from typing import Optional
import numpy as np
import onnxruntime as ort

from .common import SCRIPT_METADATA, Quad, Word, Line, OcrResult
from .vocab import Vocabulary
from .detector import TextDetector
from .classifier import ScriptClassifier
from .recognizer import TextRecognizer
from .corrector import OrientationCorrector

class OneOCR:
    """The main drop-in replacement OneOCR orchestrator engine class."""
    def __init__(self, config_dir: Optional[str | Path] = None, max_lines: int = 1000, use_gpu: bool = False):
        # Resolve config/models directory path
        candidates = []
        if config_dir is not None:
            candidates.append(Path(config_dir))
            candidates.append(Path(config_dir) / "models")
        candidates.extend([
            Path(__file__).parent.parent / "models",
            Path.home() / ".config" / "oneocr" / "models",
            Path.home() / ".config" / "oneocr",
        ])
        
        self.models_dir = None
        for p in candidates:
            if (p / "detector" / "text_detector.onnx").exists():
                self.models_dir = p
                break
        if self.models_dir is None:
            raise FileNotFoundError(
                "OneOCR ONNX models not found. Please place decrypted models folder."
            )
            
        self.max_lines = max_lines
        
        # Configure providers
        if use_gpu:
            available = ort.get_available_providers()
            gpu_providers = []
            if "CUDAExecutionProvider" in available:
                gpu_providers.append("CUDAExecutionProvider")
            if "DmlExecutionProvider" in available:
                gpu_providers.append("DmlExecutionProvider")
            if "ROCMExecutionProvider" in available:
                gpu_providers.append("ROCMExecutionProvider")
            self.providers = gpu_providers + ["CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]
        
        # Load vocabularies
        self.vocabs = {}
        for script_id, info in SCRIPT_METADATA.items():
            lang_name = info["name"]
            vocab_path = self.models_dir / "vocab" / f"vocab_{lang_name}.txt"
            self.vocabs[lang_name] = Vocabulary.load_clean(vocab_path)
            
        # Initialize modules
        self.detector = TextDetector(self.models_dir / "detector" / "text_detector.onnx", providers=self.providers)
        self.classifier = ScriptClassifier(self.models_dir / "classifier" / "script_classifier.onnx", providers=self.providers)
        self.recognizer = TextRecognizer(self.models_dir, self.vocabs, providers=self.providers)

    def __enter__(self) -> "OneOCR":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        pass

    @staticmethod
    def _map_point_back(x, y, angle, W, H):
        if angle == 0.0:
            return x, y
        elif angle == 90.0:
            return y, H - x
        elif angle == 180.0:
            return W - x, H - y
        elif angle == 270.0:
            return W - y, x
        return x, y

    def _map_box_back(self, box, angle, W, H):
        x1, y1 = self._map_point_back(box[0], box[1], angle, W, H)
        x2, y2 = self._map_point_back(box[2], box[3], angle, W, H)
        x3, y3 = self._map_point_back(box[4], box[5], angle, W, H)
        x4, y4 = self._map_point_back(box[6], box[7], angle, W, H)
        return [x1, y1, x2, y2, x3, y3, x4, y4]

    def recognize(self, image: Image.Image) -> OcrResult:
        if any(d < 50 or d > 10000 for d in image.size):
            raise ValueError(f"Image dimensions {image.size} out of supported range (50-10000 px).")
            
        orig_w, orig_h = image.size
        angle = OrientationCorrector.estimate(image, self.detector, self.classifier)
        
        upright_img = image if angle == 0.0 else image.rotate(angle, expand=True)
        sh, sv, lh, lv, sx, sy = self.detector.run(upright_img)
        
        # Auto-detect vertical vs horizontal text layout
        is_vertical = np.sum(sv > 0.5) > np.sum(sh > 0.5)
        
        if is_vertical:
            lines_list = self.detector.get_segmented_lines(sv, lv, sx, sy)
            # Sort vertical lines from right to left (standard Japanese reading order)
            lines_list.sort(key=lambda item: item[0], reverse=True)
        else:
            lines_list = self.detector.get_segmented_lines(sh, lh, sx, sy)
            
        lines_out = []
        for x, y, w, h in lines_list[:self.max_lines]:
            pad = 2
            crop_x1 = max(0, int(x) - pad)
            crop_y1 = max(0, int(y) - pad)
            crop_x2 = min(upright_img.width, int(x + w) + pad)
            crop_y2 = min(upright_img.height, int(y + h) + pad)
            
            crop = upright_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            
            # Rotate crop by 90 degrees CW if layout is vertical (stacks to horizontal read)
            if is_vertical:
                processed_crop = crop.rotate(90, expand=True)
            else:
                processed_crop = crop
                
            script_id, _ = self.classifier.classify(processed_crop)
            metadata = SCRIPT_METADATA.get(script_id + 1, SCRIPT_METADATA[3])
            lang_name = metadata["name"]
            vocab_size = metadata["vocab_size"]
            
            # Use raw size for CJK CTC mapping bounds
            blank_idx = vocab_size - 1
            
            words_data, total_preds = self.recognizer.recognize_line(processed_crop, lang_name, vocab_size)
            
            words_list = []
            line_text_parts = []
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            
            # Use total width of processed (rotated or raw) crop
            proc_w = processed_crop.width
            
            for w_chars in words_data:
                w_text = "".join(c['char'] for c in w_chars)
                w_conf = sum(c['prob'] for c in w_chars) / len(w_chars)
                
                t_start, t_end = w_chars[0]['t'], w_chars[-1]['t']
                
                # Bounding box calculation for the horizontal text inside the processed crop
                wx1 = t_start * (proc_w / total_preds)
                wx2 = (t_end + 1) * (proc_w / total_preds)
                
                if is_vertical:
                    # In 90 CW rotated crop:
                    # x_rot spans along the vertical height of original upright crop
                    # y_rot spans along horizontal width of original crop
                    wy1_upright = crop_y1 + wx1
                    wy2_upright = crop_y1 + wx2
                    
                    wx_box = [
                        crop_x1, wy1_upright,
                        crop_x2, wy1_upright,
                        crop_x2, wy2_upright,
                        crop_x1, wy2_upright
                    ]
                else:
                    wx_box = [
                        crop_x1 + wx1, crop_y1,
                        crop_x1 + wx2, crop_y1,
                        crop_x1 + wx2, crop_y2,
                        crop_x1 + wx1, crop_y2
                    ]
                    
                w_box_orig = self._map_box_back(wx_box, angle, orig_w, orig_h)
                words_list.append(Word(
                    text=w_text,
                    bbox=Quad(
                        x1=w_box_orig[0], y1=w_box_orig[1],
                        x2=w_box_orig[2], y2=w_box_orig[3],
                        x3=w_box_orig[4], y3=w_box_orig[5],
                        x4=w_box_orig[6], y4=w_box_orig[7]
                    ),
                    confidence=float(w_conf)
                ))
                line_text_parts.append(w_text)
                
            full_line_text = "".join(line_text_parts) if lang_name == "cjk" else " ".join(line_text_parts)
            if not full_line_text.strip():
                continue
                
            line_box_orig = self._map_box_back([x, y, x + w, y, x + w, y + h, x, y + h], angle, orig_w, orig_h)
            lines_out.append(Line(
                text=full_line_text,
                bbox=Quad(
                    x1=line_box_orig[0], y1=line_box_orig[1],
                    x2=line_box_orig[2], y2=line_box_orig[3],
                    x3=line_box_orig[4], y3=line_box_orig[5],
                    x4=line_box_orig[6], y4=line_box_orig[7]
                ),
                style=1 if is_vertical else 0,
                words=words_list
            ))
            
        return OcrResult(lines=lines_out, image_angle=float(angle))

    def recognize_file(self, path: str | Path) -> OcrResult:
        return self.recognize(Image.open(path))
