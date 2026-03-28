import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def draw_text(draw, text, pos, font, color=(255, 255, 255)):
    # Soft shadow
    draw.text((pos[0]+1, pos[1]+1), text, font=font, fill=(0, 0, 0, 150))
    draw.text(pos, text, font=font, fill=color)

def create_final_teaser(base_dir):
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
    
    raw_patch = img[0:row1_h, col_w:col_w*2]
    skel_patch = img[row1_h:row2_h, col_w:col_w*2]
    
    # Ensure exact same size
    min_h = min(raw_patch.shape[0], skel_patch.shape[0])
    raw_patch = raw_patch[:min_h, :]
    skel_patch = skel_patch[:min_h, :]
    
    ph, pw = raw_patch.shape[:2]
    
    # --- DESIGN ---
    # Left: Raw
    # Right: Skeleton + Force Indicator
    
    img_left = raw_patch.copy()
    img_right = skel_patch.copy()
    
    # Foot position for force indicator
    # Based on previous cam04 analysis
    fx, fy = int(pw * 0.55), int(ph * 0.85)
    
    # Draw a clean, thin force indicator that doesn't block the person
    # We will draw it as a thin line from the foot to a small HUD box on the side
    indicator_color = (0, 255, 255) # Cyan
    
    # Draw a small dot at foot
    cv2.circle(img_right, (fx, fy), 4, indicator_color, -1)
    
    # Draw a thin "lead line" to the side
    target_x = fx + 60
    target_y = fy - 40
    cv2.line(img_right, (fx, fy), (target_x, target_y), indicator_color, 1, cv2.LINE_AA)
    
    # Draw a small HUD box at the end of the line
    box_w, box_h = 110, 30
    cv2.rectangle(img_right, (target_x, target_y - box_h), (target_x + box_w, target_y), (20, 20, 25), -1)
    cv2.rectangle(img_right, (target_x, target_y - box_h), (target_x + box_w, target_y), indicator_color, 1)
    
    # Convert to PIL for nice text
    left_pil = Image.fromarray(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
    right_pil = Image.fromarray(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
    draw_l = ImageDraw.Draw(left_pil)
    draw_r = ImageDraw.Draw(right_pil)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        
    draw_text(draw_l, "Input Video", (20, 20), font_large)
    draw_text(draw_r, "Pose & GRF Detection", (20, 20), font_large, color=(0, 255, 150))
    
    # Draw force value in the HUD box
    draw_r.text((target_x + 8, target_y - 24), "Fz: 2.45 BW", font=font_small, fill=indicator_color)
    
    # Convert back to CV2
    img_left = cv2.cvtColor(np.array(left_pil), cv2.COLOR_RGB2BGR)
    img_right = cv2.cvtColor(np.array(right_pil), cv2.COLOR_RGB2BGR)
    
    # Compose
    canvas = np.zeros((ph, pw * 2 + 4, 3), dtype=np.uint8)
    canvas.fill(255) # white divider
    canvas[:, :pw] = img_left
    canvas[:, pw+4:] = img_right
    
    out_path = base_dir / "docs/demo_content/teaser.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"Created final teaser at {out_path}")

if __name__ == "__main__":
    create_final_teaser(Path("/home/nky/BadmintonGRF"))
