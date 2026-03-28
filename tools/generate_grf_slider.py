import cv2
import numpy as np
import imageio
from pathlib import Path

def create_grf_slider_assets(base_dir):
    # Load the storyboard which contains Raw Video (top), Skeleton (mid), Curve (bottom)
    rank_dir = base_dir / "runs/fig1_asset_pack_v6/per_rank/rank_001_cam04"
    storyboard_path = rank_dir / "storyboard_raw_skeleton_curve.png"
    
    if not storyboard_path.exists():
        print("Storyboard image not found!")
        return
        
    img = cv2.imread(str(storyboard_path))
    h, w = img.shape[:2]
    
    # The storyboard is 3 rows (raw, skeleton, curve)
    # We want to create two images for the slider:
    # 1. Left side (before slide): Raw Video
    # 2. Right side (after slide): Raw Video + GRF curve overlay at the bottom
    
    # The storyboard from export_fig1_asset_pack.py has 3 rows with height ratios 1:1:1.1
    # and 3 columns. Let's just use the middle column (index 1) for the best action shot.
    
    col_w = w // 3
    row1_h = int(h * (1 / 3.1))
    row2_h = int(h * (2 / 3.1))
    
    # Extract the middle column patches
    raw_patch = img[0:row1_h, col_w:col_w*2]
    skel_patch = img[row1_h:row2_h, col_w:col_w*2]
    curve_patch = img[row2_h:h, col_w:col_w*2]
    
    # Create the base "Raw" image for the slider
    # Let's make it a nice 16:9 ratio or similar. We will just use the raw patch.
    target_h, target_w = raw_patch.shape[:2]
    
    # The "Before" image is just the raw video
    img_before = raw_patch.copy()
    
    # The "After" image is the raw video with the GRF curve overlaid at the bottom,
    # or the skeleton video with the GRF curve overlaid.
    # Let's use the skeleton video to show both pose and GRF prediction.
    
    img_after = skel_patch.copy()
    
    # We need to overlay the curve onto the bottom of the image.
    # Resize curve to fit the bottom
    overlay_h = int(target_h * 0.35)
    curve_resized = cv2.resize(curve_patch, (target_w, overlay_h))
    
    # Create a semi-transparent background for the curve
    alpha = 0.85
    overlay_start_y = target_h - overlay_h
    
    # Blend the curve into the bottom of img_after
    img_after[overlay_start_y:target_h, :] = cv2.addWeighted(
        img_after[overlay_start_y:target_h, :], 1 - alpha,
        curve_resized, alpha, 0
    )
    
    # Add a cool label
    cv2.putText(img_after, "GRF Prediction vs GT", (10, target_h - overlay_h - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                
    cv2.putText(img_before, "Raw Input Video", (10, target_h - overlay_h - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Save the images
    out_dir = base_dir / "docs/demo_content"
    cv2.imwrite(str(out_dir / "slider_before.png"), img_before)
    cv2.imwrite(str(out_dir / "slider_after.png"), img_after)
    print("Created slider_before.png and slider_after.png")
    
    # Also update the GIF for the README
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
            
        # Draw the slider line
        cv2.line(frame, (split_x, 0), (split_x, target_h), (255, 255, 255), 4)
        cv2.line(frame, (split_x, 0), (split_x, target_h), (0, 165, 255), 2)
        
        # Add slider handle
        cv2.circle(frame, (split_x, target_h // 2), 12, (255, 255, 255), -1)
        cv2.circle(frame, (split_x, target_h // 2), 8, (0, 165, 255), -1)
        cv2.circle(frame, (split_x, target_h // 2), 4, (255, 255, 255), -1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gif_frames.append(frame_rgb)
        
    imageio.mimsave(str(out_path), gif_frames, fps=30)
    print(f"Updated {out_path}")

if __name__ == "__main__":
    base_dir = Path("/home/nky/BadmintonGRF")
    create_grf_slider_assets(base_dir)
