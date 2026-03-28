import cv2
import numpy as np
from pathlib import Path

def create_static_comparison(base_dir):
    out_dir = base_dir / "docs/demo_content"
    before_path = out_dir / "slider_before.png"
    after_path = out_dir / "slider_after.png"
    
    if not before_path.exists() or not after_path.exists():
        print("Source images not found!")
        return
        
    img_before = cv2.imread(str(before_path))
    img_after = cv2.imread(str(after_path))
    
    # We want a side-by-side comparison
    h1, w1 = img_before.shape[:2]
    h2, w2 = img_after.shape[:2]
    
    # Make them same height
    min_h = min(h1, h2)
    img_before = img_before[:min_h, :]
    img_after = img_after[:min_h, :]
    
    h, w = img_before.shape[:2]
    
    # Create a blank canvas
    padding = 10
    total_w = w * 2 + padding
    canvas = np.zeros((h, total_w, 3), dtype=np.uint8)
    canvas.fill(255) # white background/divider
    
    # Place images
    canvas[:, :w] = img_before
    canvas[:, w+padding:] = img_after
    
    out_path = out_dir / "readme_comparison.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"Created {out_path}")

if __name__ == "__main__":
    base_dir = Path("/home/nky/BadmintonGRF")
    create_static_comparison(base_dir)
