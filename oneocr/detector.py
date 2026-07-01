from PIL import Image
import numpy as np
import onnxruntime as ort

class TextDetector:
    """Handles text region detection using the FPN ONNX model."""
    def __init__(self, model_path, providers=None):
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(str(model_path), providers=providers)

    def run(self, image: Image.Image, max_side: int = 1536):
        orig_w, orig_h = image.size
        scale = min(max_side / max(orig_w, orig_h), 1.0)
        target_w = int(round(orig_w * scale / 32.0) * 32)
        target_h = int(round(orig_h * scale / 32.0) * 32)
        target_w = max(32, target_w)
        target_h = max(32, target_h)
        
        resized_img = image.resize((target_w, target_h))
        img_np = np.array(resized_img.convert("RGB")).astype(np.float32)
        chw = np.transpose(img_np, (2, 0, 1))
        inp = np.expand_dims(chw, axis=0).astype(np.float32)
        im_info = np.array([[float(target_h), float(target_w), 1.0]], dtype=np.float32)
        
        outputs = self.sess.run(None, {'data': inp, 'im_info': im_info})
        return outputs[0][0, 0], outputs[2][0, 0], outputs[4][0], outputs[5][0], target_w / orig_w, target_h / orig_h

    def get_segmented_lines(self, scores, links, sx, sy, score_threshold: float = 0.5, link_threshold: float = 0.0):
        H_lvl, W_lvl = scores.shape
        stride = 4
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1),   (1, 1),  (1, 0),
            (1, -1),  (0, -1)
        ]
        
        active_pixels = set()
        for r in range(H_lvl):
            for c in range(W_lvl):
                if scores[r, c] > score_threshold:
                    active_pixels.add((r, c))
                    
        parent = {p: p for p in active_pixels}
        
        def find(p):
            if parent[p] == p:
                return p
            parent[p] = find(parent[p])
            return parent[p]
            
        def union(p1, p2):
            r1, r2 = find(p1), find(p2)
            if r1 != r2:
                parent[r1] = r2
                
        for r, c in active_pixels:
            for n_idx, (dy, dx) in enumerate(neighbors):
                nr, nc = r + dy, c + dx
                if (nr, nc) in active_pixels:
                    if links[n_idx, r, c] > link_threshold:
                        union((r, c), (nr, nc))
                        
        groups = {}
        for p in active_pixels:
            root = find(p)
            if root not in groups:
                groups[root] = []
            groups[root].append(p)
            
        lines = []
        for root, pixels in groups.items():
            if len(pixels) < 5:
                continue
            rows, cols = [p[0] for p in pixels], [p[1] for p in pixels]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            
            x = (min_c * stride) / sx
            y = (min_r * stride) / sy
            w = ((max_c - min_c + 1) * stride) / sx
            h = ((max_r - min_r + 1) * stride) / sy
            
            if w < 10 or h < 10:
                continue
            lines.append((x, y, w, h))
            
        lines.sort(key=lambda item: item[1])
        return lines

    def get_text_mask(self, image: Image.Image, max_side: int = 1536, score_threshold: float = 0.5) -> Image.Image:
        """Extract a pixel-level binary text mask (0=bg, 255=text) from the FPN network."""
        sh, sv, _, _, sx, sy = self.run(image, max_side=max_side)
        combined = np.maximum(sh, sv)
        mask_np = (combined > score_threshold).astype(np.uint8) * 255
        
        # Convert to L image and scale back to original size
        mask_img = Image.fromarray(mask_np, mode="L")
        return mask_img.resize(image.size, resample=Image.NEAREST)
