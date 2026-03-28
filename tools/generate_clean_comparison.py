import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def draw_hud_text(img_pil, text, pos, font_size, color=(255, 255, 255), bold=False):
    draw = ImageDraw.Draw(img_pil)
    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
        
    x, y = pos
    # Soft text shadow for readability
    draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0, 200))
    draw.text((x, y), text, font=font, fill=color)
    return img_pil

def create_clean_comparison(base_dir):
    rank_dir = base_dir / "runs/fig1_asset_pack_v6/per_rank/rank_001_cam04"
    storyboard_path = rank_dir / "storyboard_raw_skeleton_curve.png"
    
    if not storyboard_path.exists():
        print("Storyboard image not found!")
        return
        
    img = cv2.imread(str(storyboard_path))
    h, w = img.shape[:2]
    
    col_w = w // 3
    row1_h = int(h * (1 / 3.1))
    row2_h = int(h * (2 / 3.1))
    
    # Extract patches
    raw_patch = img[0:row1_h, col_w:col_w*2]
    skel_patch = img[row1_h:row2_h, col_w:col_w*2]
    curve_patch = img[row2_h:h, col_w:col_w*2]
    
    # Ensure they have exact same height for compositing
    min_h = min(raw_patch.shape[0], skel_patch.shape[0])
    raw_patch = raw_patch[:min_h, :]
    skel_patch = skel_patch[:min_h, :]
    
    patch_h, patch_w = raw_patch.shape[:2]
    
    # --- DESIGNING THE LEFT IMAGE (Raw Input) ---
    img_before = raw_patch.copy()
    
    img_before_pil = Image.fromarray(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
    draw_hud_text(img_before_pil, "RAW VIDEO", (30, 30), 28, color=(255, 255, 255), bold=True)
    img_before = cv2.cvtColor(np.array(img_before_pil), cv2.COLOR_RGB2BGR)
    
    # --- DESIGNING THE RIGHT IMAGE (AI GRF Estimation - CLEAN PIP) ---
    # We will use the RAW image again, but with the skeleton overlaid and the UI panels
    # No floor grid, no blocking arrows. Just pure data overlay.
    img_after = skel_patch.copy()
    
    # Insert the GRF Curve nicely into the TOP RIGHT corner to avoid the player entirely
    # Crop the actual curve to remove axes/margins if possible, or just scale it
    curve_h, curve_w = curve_patch.shape[:2]
    pip_w = int(patch_w * 0.45)
    pip_h = int(curve_h * (pip_w / curve_w))
    curve_resized = cv2.resize(curve_patch, (pip_w, pip_h))
    
    # Create a nice dark semi-transparent panel for the curve
    panel_margin = 20
    panel_x = patch_w - pip_w - panel_margin
    panel_y = 20 # Top right
    
    # Draw dark panel background
    panel_overlay = img_after.copy()
    cv2.rectangle(panel_overlay, (panel_x-15, panel_y-15), (panel_x+pip_w+15, panel_y+pip_h+15), (20, 20, 25), -1)
    # Alpha blend the panel
    img_after = cv2.addWeighted(panel_overlay, 0.85, img_after, 0.15, 0)
    
    # Add the curve image
    img_after[panel_y:panel_y+pip_h, panel_x:panel_x+pip_w] = cv2.addWeighted(
        img_after[panel_y:panel_y+pip_h, panel_x:panel_x+pip_w], 0.05,
        curve_resized, 0.95, 0
    )
    
    # Add a thin sleek border to the panel
    cv2.rectangle(img_after, (panel_x-15, panel_y-15), (panel_x+pip_w+15, panel_y+pip_h+15), (80, 80, 80), 1)
    
    # 4. Add Modern Typography
    img_after_pil = Image.fromarray(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
    
    # Main Title (Top Left)
    draw_hud_text(img_after_pil, "AI GRF ESTIMATION", (30, 30), 28, color=(0, 255, 150), bold=True)
    
    # Panel Title
    draw_hud_text(img_after_pil, "REAL-TIME DYNAMICS (Fz)", (panel_x, panel_y + pip_h - 20), 14, color=(40, 40, 40), bold=True) # Text inside curve area if needed, or outside
    
    # We will put the IMPACT value in a small neat box in the BOTTOM LEFT
    val_box_x = 30
    val_box_y = patch_h - 100
    val_box_w = 200
    val_box_h = 70
    
    img_after = cv2.cvtColor(np.array(img_after_pil), cv2.COLOR_RGB2BGR)
    val_overlay = img_after.copy()
    cv2.rectangle(val_overlay, (val_box_x, val_box_y), (val_box_x+val_box_w, val_box_y+val_box_h), (20, 20, 25), -1)
    img_after = cv2.addWeighted(val_overlay, 0.8, img_after, 0.2, 0)
    cv2.rectangle(img_after, (val_box_x, val_box_y), (val_box_x+val_box_w, val_box_y+val_box_h), (80, 80, 80), 1)
    
    img_after_pil = Image.fromarray(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
    
    draw_hud_text(img_after_pil, "PEAK IMPACT", (val_box_x + 15, val_box_y + 10), 14, color=(150, 150, 150), bold=True)
    draw_hud_text(img_after_pil, "2.45", (val_box_x + 15, val_box_y + 30), 32, color=(0, 165, 255), bold=True)
    draw_hud_text(img_after_pil, "BW", (val_box_x + 95, val_box_y + 40), 16, color=(0, 165, 255))
    
    img_after = cv2.cvtColor(np.array(img_after_pil), cv2.COLOR_RGB2BGR)
    
    # --- COMPOSE SIDE-BY-SIDE ---
    total_w = patch_w * 2
    canvas = np.zeros((patch_h, total_w, 3), dtype=np.uint8)
    
    canvas[:, :patch_w] = img_before
    canvas[:, patch_w:] = img_after
    
    # Draw a thin, elegant separator line
    cv2.line(canvas, (patch_w, 0), (patch_w, patch_h), (50, 50, 50), 2)
    
    out_dir = base_dir / "docs/demo_content"
    out_path = out_dir / "readme_comparison.png"
    cv2.imwrite(str(out_path), canvas)
    
    # Save individual for slider
    cv2.imwrite(str(out_dir / "slider_before.png"), img_before)
    cv2.imwrite(str(out_dir / "slider_after.png"), img_after)
    
    print(f"Created clean revamped {out_path}")

if __name__ == "__main__":
    base_dir = Path("/home/nky/BadmintonGRF")
    create_clean_comparison(base_dir)
