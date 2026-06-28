from PIL import Image
import numpy as np
import onnxruntime as ort
from .common import Word

class TextRecognizer:
    """Loads script recognizers dynamically and runs CTC decoding."""
    def __init__(self, models_dir, vocabs, providers=None):
        self.models_dir = models_dir
        self.vocabs = vocabs
        self.providers = providers if providers is not None else ['CPUExecutionProvider']
        self.sessions = {}

    def _get_session(self, lang_name: str) -> ort.InferenceSession:
        if lang_name not in self.sessions:
            path = self.models_dir / "recognizers" / f"recognizer_{lang_name}.onnx"
            self.sessions[lang_name] = ort.InferenceSession(str(path), providers=self.providers)
        return self.sessions[lang_name]

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def recognize_line(self, crop: Image.Image, lang_name: str, vocab_size: int):
        aspect = crop.width / crop.height
        new_width = max(16, int(60 * aspect))
        resized_crop = crop.resize((new_width, 60))
        
        rec_sess = self._get_session(lang_name)
        rec_inp = np.array(resized_crop.convert("RGB")).astype(np.float32)[:, :, ::-1] / 255.0
        rec_chw = np.transpose(rec_inp, (2, 0, 1))
        rec_data = np.expand_dims(rec_chw, axis=0).astype(np.float32)
        seq_lengths = np.array([new_width // 4], dtype=np.int32)
        
        rec_out = rec_sess.run(None, {'data': rec_data, 'seq_lengths': seq_lengths})
        logits = rec_out[0]
        probs = self._softmax(logits[:, 0, :])
        preds = np.argmax(probs, axis=-1)
        
        # CTC Decode
        blank = vocab_size - 1
        char_runs = []
        prev = -1
        for t, pred in enumerate(preds):
            if pred != prev and pred != blank:
                char = " " if pred == 0 else self.vocabs[lang_name].get(pred, "")
                char_runs.append({'char': char, 'prob': probs[t, pred], 't': t})
            prev = pred
            
        words_data = []
        current_word = []
        for run in char_runs:
            if run['char'] == " " or not run['char']:
                if current_word:
                    words_data.append(current_word)
                    current_word = []
            else:
                current_word.append(run)
        if current_word:
            words_data.append(current_word)
            
        return words_data, len(preds)
