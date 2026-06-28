import numpy as np

class OrientationCorrector:
    """Determines the optimal upright correction angle (0, 90, 180, 270)."""
    @staticmethod
    def estimate(image, detector, classifier) -> float:
        best_angle = 0.0
        best_quality = -9999.0
        
        for angle in [0.0, 90.0, 180.0, 270.0]:
            candidate_img = image if angle == 0.0 else image.rotate(angle, expand=True)
            sh, sv, lh, lv, sx, sy = detector.run(candidate_img)
            active_pixels = np.argwhere(sh > 0.5)
            if len(active_pixels) == 0:
                continue
                
            scores_active = sh[sh > 0.5]
            sorted_indices = np.argsort(scores_active)[::-1]
            
            flip_scores = []
            valid_scripts = 0
            
            for idx in sorted_indices[:3]:
                r, c = active_pixels[idx]
                stride = 4
                cx, cy = c * stride + 2.0, r * stride + 2.0
                x1 = max(0, int((cx - 60) / sx))
                y1 = max(0, int((cy - 20) / sy))
                x2 = min(candidate_img.width, int((cx + 60) / sx))
                y2 = min(candidate_img.height, int((cy + 20) / sy))
                
                crop = candidate_img.crop((x1, y1, x2, y2))
                script_id, flip_score = classifier.classify(crop)
                flip_scores.append(flip_score)
                if script_id != 0:
                    valid_scripts += 1
                    
            avg_flip = np.mean(flip_scores) if flip_scores else -300.0
            quality = (valid_scripts * 1000.0) + avg_flip
            
            if quality > best_quality:
                best_quality = quality
                best_angle = angle
                
        return best_angle
