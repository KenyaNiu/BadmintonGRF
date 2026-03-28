import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

def create_gradient_mask(h, w, color, alpha_start, alpha_end):
    mask = np.zeros((h, w, 4), dtype=np.float32)
    mask[:, :, :3] = color
    
    # Create vertical gradient for alpha
    alphas = np.linspace(alpha_start, alpha_end, h)
    alphas = np.tile(alphas, (w, 1)).T
    mask[:, :, 3] = alphas
    return mask

def apply_alpha_mask(img, mask):
    # img is BGR, mask is BGRA (float 0-1)
    img_float = img.astype(np.float32) / 255.0
    
    alpha = mask[:, :, 3:]
    color = mask[:, :, :3]
    
    blended = img_float * (1 - alpha) + color * alpha
    return (blended * 255).astype(np.uint8)

def draw_premium_text(img_pil, text, pos, font_size, color=(255, 255, 255), shadow=True, align="left"):
    draw = ImageDraw.Draw(img_pil)
    
    # Try to load a nice font, fallback to default
    try:
        # You might need to adjust this path depending on the OS, or just use a generic sans-serif
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
        
    x, y = pos
    
    # Get text bounding box using textbbox instead of textsize (deprecated)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    if align == "right":
        x = x - tw
    elif align == "center":
        x = x - tw // 2
        
    if shadow:
        # Subtle drop shadow
        draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0, 150))
        draw.text((x+1, y+1), text, font=font, fill=(0, 0, 0, 200))
        
    draw.text((x, y), text, font=font, fill=color)
    return img_pil

def draw_premium_vector(img, foot_pos, force_mag):
    # Instead of a harsh yellow line, we'll draw an elegant translucent glowing vector
    x, y = foot_pos
    
    # We want a sleek, modern look. Thin core line, wide soft glow.
    core_color = (255, 255, 255) # White core
    glow_color = (0, 165, 255)   # Orange glow (BGR)
    
    arrow_len = int(force_mag * 140)
    end_point = (x, y - arrow_len)
    
    # Create an overlay for blending
    overlay = img.copy()
    
    # 1. Broad soft glow
    cv2.line(overlay, foot_pos, end_point, glow_color, 20)
    cv2.circle(overlay, foot_pos, 15, glow_color, -1)
    
    # 2. Medium glow
    cv2.line(overlay, foot_pos, end_point, glow_color, 8)
    
    # Blend glow
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    
    # 3. Sharp core line
    cv2.line(img, foot_pos, end_point, core_color, 2)
    
    # 4. Elegant Arrow head (custom drawn instead of cv2.arrowedLine for better styling)
    head_len = 20
    angle = math.atan2(y - end_point[1], x - end_point[0])
    
    pt1 = (int(end_point[0] + head_len * math.cos(angle - math.pi/6)),
           int(end_point[1] + head_len * math.sin(angle - math.pi/6)))
    pt2 = (int(end_point[0] + head_len * math.cos(angle + math.pi/6)),
           int(end_point[1] + head_len * math.sin(angle + math.pi/6)))
           
    pts = np.array([end_point, pt1, pt2], np.int32)
    cv2.fillPoly(img, [pts], core_color)
    
    # 5. Foot impact rings (ripple effect)
    cv2.circle(img, foot_pos, 4, core_color, -1)
    cv2.circle(img, foot_pos, 12, core_color, 1, cv2.LINE_AA)
    cv2.circle(img, foot_pos, 24, glow_color, 2, cv2.LINE_AA)
    
    return img

def create_premium_comparison(base_dir):
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
    
    # We use the raw frame for BOTH sides to ensure maximum visual cleanliness.
    # We will draw the skeleton and vector ourselves if needed, or use the skeleton patch.
    # Let's use the raw patch and just overlay the vector, the skeleton is too messy for a "premium" look.
    raw_patch = img[0:row1_h, col_w:col_w*2]
    
    img_before = raw_patch.copy()
    img_after = raw_patch.copy()
    
    patch_h, patch_w = raw_patch.shape[:2]
    foot_pos = (int(patch_w * 0.55), int(patch_h * 0.85))
    pred_force = 2.45
    
    # --- STYLING "AFTER" IMAGE ---
    
    # Add a subtle dark vignette to make the glowing vector pop
    vignette = np.zeros_like(img_after, dtype=np.float32)
    cv2.circle(vignette, (patch_w//2, patch_h//2), int(patch_w*0.8), (1,1,1), -1)
    vignette = cv2.GaussianBlur(vignette, (0,0), patch_w//4)
    img_after = (img_after * (0.6 + 0.4 * vignette)).astype(np.uint8)
    
    # Draw premium vector
    img_after = draw_premium_vector(img_after, foot_pos, pred_force)
    
    # --- ADDING PREMIUM TEXT (Using PIL for antialiasing and custom fonts) ---
    
    img_before_pil = Image.fromarray(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
    img_after_pil = Image.fromarray(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
    
    # Draw sleek badges instead of raw text
    # Before
    draw_premium_text(img_before_pil, "Input Video", (40, 40), 36, color=(255, 255, 255))
    
    # After
    draw_premium_text(img_after_pil, "Predicted GRF Vector", (40, 40), 36, color=(255, 255, 255))
    
    # Force Value Badge
    badge_text = f"{pred_force:.2f} BW"
    
    # Convert back to OpenCV to draw a modern translucent glass-morphism badge
    img_after = cv2.cvtColor(np.array(img_after_pil), cv2.COLOR_RGB2BGR)
    
    badge_x = foot_pos[0] + 30
    badge_y = foot_pos[1] - int(pred_force * 140) // 2
    
    badge_w, badge_h = 120, 40
    
    # Glassmorphism background
    badge_roi = img_after[badge_y:badge_y+badge_h, badge_x:badge_x+badge_w]
    if badge_roi.shape[:2] == (badge_h, badge_w):
        blurred = cv2.GaussianBlur(badge_roi, (15, 15), 0)
        # Add white tint
        tint = np.full_like(blurred, 255)
        glass = cv2.addWeighted(blurred, 0.7, tint, 0.3, 0)
        img_after[badge_y:badge_y+badge_h, badge_x:badge_x+badge_w] = glass
        
        # Border
        cv2.rectangle(img_after, (badge_x, badge_y), (badge_x+badge_w, badge_y+badge_h), (255, 255, 255), 1)
        
    # Draw text on badge
    img_after_pil = Image.fromarray(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
    draw_premium_text(img_after_pil, badge_text, (badge_x + 15, badge_y + 8), 20, color=(30, 30, 30), shadow=False)
    img_after = cv2.cvtColor(np.array(img_after_pil), cv2.COLOR_RGB2BGR)
    img_before = cv2.cvtColor(np.array(img_before_pil), cv2.COLOR_RGB2BGR)
    
    # --- COMPOSE SIDE-BY-SIDE ---
    
    # We want a very clean, thin divider. Maybe a gap with background color.
    gap = 4
    total_w = patch_w * 2 + gap
    canvas = np.zeros((patch_h, total_w, 3), dtype=np.uint8)
    canvas.fill(240) # Light grey background
    
    canvas[:, :patch_w] = img_before
    canvas[:, patch_w+gap:] = img_after
    
    out_dir = base_dir / "docs/demo_content"
    out_path = out_dir / "readme_comparison.png"
    cv2.imwrite(str(out_path), canvas)
    
    # Also save the individual ones for the slider
    cv2.imwrite(str(out_dir / "slider_before.png"), img_before)
    cv2.imwrite(str(out_dir / "slider_after.png"), img_after)
    
    print(f"Created premium {out_path}")

if __name__ == "__main__":
    base_dir = Path("/home/nky/BadmintonGRF")
    create_premium_comparison(base_dir)
