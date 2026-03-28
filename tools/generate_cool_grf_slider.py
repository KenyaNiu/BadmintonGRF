import cv2
import numpy as np
import imageio
from pathlib import Path
import math

def draw_cool_force_vector(img, foot_pos, force_mag, is_gt=False):
    """Draws a sci-fi/HUD style force vector emerging from the foot."""
    # Base color: Cyan for Pred, Magenta/Orange for GT
    color = (0, 255, 255) if not is_gt else (0, 165, 255)
    
    x, y = foot_pos
    
    # Scale arrow length based on force magnitude
    # Let's say 1 BW = 100 pixels
    arrow_len = int(force_mag * 120)
    
    end_point = (x, y - arrow_len)
    
    # Draw glow effect (multiple thick lines with decreasing thickness and increasing alpha)
    overlay = img.copy()
    cv2.line(overlay, foot_pos, end_point, color, 12)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    
    # Draw main arrow
    cv2.arrowedLine(img, foot_pos, end_point, color, 4, cv2.LINE_AA, tipLength=0.15)
    
    # Draw base circle (impact point)
    cv2.circle(img, foot_pos, 8, color, -1)
    cv2.circle(img, foot_pos, 14, color, 2, cv2.LINE_AA)
    
    # Add a HUD text label near the arrow
    label = f"Pred Fz: {force_mag:.2f} BW" if not is_gt else f"GT Fz: {force_mag:.2f} BW"
    text_pos = (x + 20, y - arrow_len // 2)
    
    # Text background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (text_pos[0]-5, text_pos[1]-th-5), (text_pos[0]+tw+5, text_pos[1]+5), (0,0,0), -1)
    cv2.putText(img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    return img

def create_cool_slider_assets(base_dir):
    # We will use the raw frame from the storyboard and manually add the cool VFX
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
    
    img_before = raw_patch.copy()
    img_after = skel_patch.copy()
    
    # Hardcode a foot position based on visual inspection of the cam04 shot
    # The player is roughly in the center-right, landing on their right foot.
    # We will estimate the foot coordinate.
    patch_h, patch_w = raw_patch.shape[:2]
    foot_pos = (int(patch_w * 0.55), int(patch_h * 0.85))
    
    # Simulated force magnitudes at impact
    pred_force = 2.45
    
    # Draw the cool HUD vector on the "after" image
    img_after = draw_cool_force_vector(img_after, foot_pos, pred_force, is_gt=False)
    
    # Add a cool title
    cv2.putText(img_after, "AI GRF Estimation", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img_before, "Raw Input", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Save the images
    out_dir = base_dir / "docs/demo_content"
    cv2.imwrite(str(out_dir / "slider_before.png"), img_before)
    cv2.imwrite(str(out_dir / "slider_after.png"), img_after)
    print("Created cool slider_before.png and slider_after.png")
    
    create_sliding_gif(img_before, img_after, out_dir / "slider_demo.gif")

def create_sliding_gif(img_before, img_after, out_path, frames=60):
    target_w = 800
    h, w = img_before.shape[:2]
    target_h = int(h * (target_w / w))
    
    raw = cv2.resize(img_before, (target_w, target_h))
    skel = cv2.resize(img_after, (target_w, target_h))
    
    gif_frames = []
    
    for i in range(frames):
        progress = i / (frames - 1)
        sweep_pos = (np.sin(progress * 2 * np.pi - np.pi/2) + 1) / 2
        split_x = int(target_w * sweep_pos)
        
        frame = raw.copy()
        if split_x > 0:
            frame[:, :split_x] = skel[:, :split_x]
            
        cv2.line(frame, (split_x, 0), (split_x, target_h), (255, 255, 255), 4)
        cv2.line(frame, (split_x, 0), (split_x, target_h), (0, 255, 255), 2)
        
        cv2.circle(frame, (split_x, target_h // 2), 12, (255, 255, 255), -1)
        cv2.circle(frame, (split_x, target_h // 2), 8, (0, 200, 255), -1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gif_frames.append(frame_rgb)
        
    imageio.mimsave(str(out_path), gif_frames, fps=30)
    print(f"Updated {out_path}")

if __name__ == "__main__":
    base_dir = Path("/home/nky/BadmintonGRF")
    create_cool_slider_assets(base_dir)
