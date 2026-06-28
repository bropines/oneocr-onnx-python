from PIL import Image
import numpy as np
import onnxruntime as ort

class ScriptClassifier:
    """Identifies the script type and orientation properties of cropped zones."""
    def __init__(self, model_path, providers=None):
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(str(model_path), providers=providers)

    def classify(self, crop: Image.Image) -> tuple[int, float]:
        resized = crop.resize((200, 60)).convert("RGB")
        # Convert to BGR and normalize to [0, 1] range
        cls_inp = np.array(resized).astype(np.float32)[:, :, ::-1] / 255.0
        cls_chw = np.transpose(cls_inp, (2, 0, 1))
        cls_data = np.expand_dims(cls_chw, axis=0)
        
        cls_out = self.sess.run(None, {'data': cls_data})
        script_scores = cls_out[3].flatten()
        script_id = np.argmax(script_scores)
        flip_score = cls_out[5].flatten()[0]
        return script_id, flip_score
