import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def draw_text(draw, text, pos, font, color=(255, 255, 255)):
    # Soft shadow
    draw.text((pos[0]+1, pos[1]+1), text, font=font, fill=(0, 0, 0, 150))
    draw.text(pos, text, font=font, fill=color)

def apply_face_mosaic(img, pixel_size=15):
    # Load face cascade
    cascade_path = "/home/nky/miniconda3/envs/badminton_grf/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # If Haar misses (common in sports), use a heuristic based on the person's location
    # In this specific cam04 shot, the person's head is roughly at (0.55*w, 0.25*h) relative to the person's bbox
    # But let's try to be more general. If faces is empty, we'll use a hardcoded ROI for this specific teaser
    if len(faces) == 0:
        # Hardcoded ROI for the athlete in rank_001_cam04 middle frame
        # Relative to the patch size
        h, w = img.shape[:2]
        # Athlete head is roughly here:
        faces = [(int(w * 0.52), int(h * 0.38), int(w * 0.06), int(h * 0.12))]

    for (x, y, w_f, h_f) in faces:
        # Expand ROI slightly
        x = max(0, x - int(w_f * 0.2))
        y = max(0, y - int(h_f * 0.2))
        w_f = min(img.shape[1] - x, int(w_f * 1.4))
        h_f = min(img.shape[0] - y, int(h_f * 1.4))
        
        face_roi = img[y:y+h_f, x:x+w_f]
        if face_roi.size == 0: continue
        
        # Pixelate
        temp = cv2.resize(face_roi, (w_f // pixel_size, h_f // pixel_size), interpolation=cv2.INTER_LINEAR)
        img[y:y+h_f, x:x+w_f] = cv2.resize(temp, (w_f, h_f), interpolation=cv2.INTER_NEAREST)
    return img

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
    
    # Apply face mosaic to BOTH patches
    raw_patch = apply_face_mosaic(raw_patch)
    skel_patch = apply_face_mosaic(skel_patch)
    
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
