import cv2
import numpy as np
import base64
from typing import List, Dict, Any

class XAIService:
    def generate_heatmap_overlay(self, image_path: str, predictions: List[Dict[str, Any]]) -> str:
        """
        Generates a Grad-CAM / SHAP style explainability overlay on the image.
        Uses Gaussian heat spots centered on predictions to simulate activation areas.
        Returns a base64 encoded JPEG string.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return ""
            
            h, w, c = img.shape
            
            # Create blank single-channel heatmap accumulator
            heatmap_acc = np.zeros((h, w), dtype=np.float32)
            
            for pred in predictions:
                coords = pred.get("coordinates")
                if not coords:
                    continue
                
                # Get center coordinates
                box = coords.get("box")
                if box:
                    cx = int(box["x"] + box["w"] / 2)
                    cy = int(box["y"] + box["h"] / 2)
                    radius = int(max(box["w"], box["h"]) * 1.5)
                elif coords.get("points"):
                    pts = coords.get("points")
                    cx = int(sum(p["x"] for p in pts) / len(pts))
                    cy = int(sum(p["y"] for p in pts) / len(pts))
                    radius = 80
                else:
                    continue
                
                # Draw Gaussian activation blob on the accumulator
                # Create coordinate grids
                grid_y, grid_x = np.ogrid[-cy:h-cy, -cx:w-cx]
                mask = grid_x*grid_x + grid_y*grid_y <= radius*radius
                
                # Activation values drop off towards the edges
                dist_sq = grid_x*grid_x + grid_y*grid_y
                spot = np.exp(-dist_sq / (2.0 * (radius / 2.0)**2))
                spot[~mask] = 0
                
                # Accumulate
                heatmap_acc = np.maximum(heatmap_acc, spot)
                
            # Normalize to 0-255
            heatmap_acc = (heatmap_acc * 255).astype(np.uint8)
            
            # Apply COLORMAP_JET to get typical red/blue activation heatmap
            color_heatmap = cv2.applyColorMap(heatmap_acc, cv2.COLORMAP_JET)
            
            # Overlay heatmap with original image (alpha blend)
            overlay = cv2.addWeighted(img, 0.6, color_heatmap, 0.4, 0)
            
            # Encode to JPEG base64
            _, buffer = cv2.imencode('.jpg', overlay)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_str}"
            
        except Exception as e:
            return ""

xai_service = XAIService()
