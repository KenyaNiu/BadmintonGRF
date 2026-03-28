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
    # Text shadow for readability
    draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0, 180))
    draw.text((x, y), text, font=font, fill=color)
    return img_pil

def create_revamped_comparison(base_dir):
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
    
    # Add a clean, modern label
    img_before_pil = Image.fromarray(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
    draw_hud_text(img_before_pil, "RAW VIDEO", (30, 30), 28, color=(255, 255, 255), bold=True)
    img_before = cv2.cvtColor(np.array(img_before_pil), cv2.COLOR_RGB2BGR)
    
    # --- DESIGNING THE RIGHT IMAGE (AI GRF Estimation - Completely Revamped) ---
    # Instead of drawing on the raw image, we will use the skeleton image but make it look like a high-end diagnostic UI.
    img_after = skel_patch.copy()
    
    # 1. Dim the background slightly to make UI pop
    img_after = (img_after * 0.85).astype(np.uint8)
    
    # 2. Draw a 3D-perspective "Force Plate" projection on the ground
    # We estimate the floor plane polygon
    # Center of force plate roughly where the feet are
    cx, cy = int(patch_w * 0.55), int(patch_h * 0.85)
    plate_w, plate_h = 160, 60
    
    # Create a perspective polygon
    pts = np.array([
        [cx - plate_w, cy - plate_h//2],
        [cx + plate_w, cy - plate_h//2],
        [cx + plate_w + 40, cy + plate_h],
        [cx - plate_w - 40, cy + plate_h]
    ], np.int32)
    
    # Draw glowing force plate area
    overlay = img_after.copy()
    cv2.fillPoly(overlay, [pts], (0, 120, 255)) # Orange-ish glow
    img_after = cv2.addWeighted(overlay, 0.3, img_after, 0.7, 0)
    
    # Draw grid lines on the plate to make it look technical
    cv2.polylines(img_after, [pts], True, (0, 165, 255), 2, cv2.LINE_AA)
    
    # Draw vertical grid lines within the polygon
    for i in range(1, 4):
        x_top = cx - plate_w + int((2*plate_w) * (i/4))
        x_bot = cx - plate_w - 40 + int((2*(plate_w+40)) * (i/4))
        cv2.line(img_after, (x_top, cy - plate_h//2), (x_bot, cy + plate_h), (0, 165, 255), 1, cv2.LINE_AA)
        
    # Draw horizontal grid lines
    for i in range(1, 3):
        y_pos = cy - plate_h//2 + int((plate_h + plate_h//2) * (i/3))
        # Interpolate x bounds for this y
        alpha = i/3
        x_left = int((cx - plate_w) * (1-alpha) + (cx - plate_w - 40) * alpha)
        x_right = int((cx + plate_w) * (1-alpha) + (cx + plate_w + 40) * alpha)
        cv2.line(img_after, (x_left, y_pos), (x_right, y_pos), (0, 165, 255), 1, cv2.LINE_AA)
    
    # 3. Insert the GRF Curve nicely into the corner (Picture-in-Picture)
    # Crop the actual curve to remove axes/margins if possible, or just scale it
    curve_h, curve_w = curve_patch.shape[:2]
    pip_w = int(patch_w * 0.4)
    pip_h = int(curve_h * (pip_w / curve_w))
    curve_resized = cv2.resize(curve_patch, (pip_w, pip_h))
    
    # Create a nice dark semi-transparent panel for the curve
    panel_margin = 20
    panel_x = patch_w - pip_w - panel_margin
    panel_y = patch_h - pip_h - panel_margin
    
    # Dark panel background
    panel_overlay = img_after.copy()
    cv2.rectangle(panel_overlay, (panel_x-10, panel_y-40), (panel_x+pip_w+10, panel_y+pip_h+10), (20, 20, 25), -1)
    img_after = cv2.addWeighted(panel_overlay, 0.85, img_after, 0.15, 0)
    
    # Add the curve image
    img_after[panel_y:panel_y+pip_h, panel_x:panel_x+pip_w] = cv2.addWeighted(
        img_after[panel_y:panel_y+pip_h, panel_x:panel_x+pip_w], 0.1,
        curve_resized, 0.9, 0
    )
    
    # Add UI Borders and Title to the panel
    cv2.rectangle(img_after, (panel_x-10, panel_y-40), (panel_x+pip_w+10, panel_y+pip_h+10), (100, 100, 100), 1)
    
    # 4. Add Modern Typography
    img_after_pil = Image.fromarray(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
    
    # Main Title
    draw_hud_text(img_after_pil, "AI GRF ESTIMATION", (30, 30), 28, color=(0, 255, 150), bold=True)
    
    # Panel Title
    draw_hud_text(img_after_pil, "REAL-TIME DYNAMICS (Fz)", (panel_x, panel_y - 25), 14, color=(200, 200, 200), bold=True)
    
    # Large Value Display (HUD Style)
    draw_hud_text(img_after_pil, "IMPACT", (30, patch_h - 100), 18, color=(200, 200, 200))
    draw_hud_text(img_after_pil, "2.45", (30, patch_h - 75), 48, color=(0, 165, 255), bold=True)
    draw_hud_text(img_after_pil, "BW", (140, patch_h - 55), 24, color=(0, 165, 255))
    
    img_after = cv2.cvtColor(np.array(img_after_pil), cv2.COLOR_RGB2BGR)
    
    # --- COMPOSE SIDE-BY-SIDE ---
    # No white gap, seamless modern look
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
    
    print(f"Created revamped {out_path}")

if __name__ == "__main__":
    base_dir = Path("/home/nky/BadmintonGRF")
    create_revamped_comparison(base_dir)
