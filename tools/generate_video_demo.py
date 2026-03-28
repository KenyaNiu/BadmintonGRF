import cv2
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm

def create_simulated_video(base_dir, out_path, frames=120):
    rank_dir = base_dir / "runs/fig1_asset_pack_v6/per_rank/rank_001_cam04"
    curve_path = rank_dir / "curve.png"
    skel_paths = [
        rank_dir / "singles_9" / "skeleton_t1.png",
        rank_dir / "singles_9" / "skeleton_t2.png",
        rank_dir / "singles_9" / "skeleton_t3.png"
    ]
    
    if not curve_path.exists() or not all(p.exists() for p in skel_paths):
        print("Missing source images, creating a static placeholder.")
        return
        
    curve = cv2.imread(str(curve_path))
    skels = [cv2.imread(str(p)) for p in skel_paths]
    
    # Target dimensions
    target_w = 1200
    
    # Resize skeleton
    h, w = skels[0].shape[:2]
    skel_h = int(h * (target_w / w))
    skels = [cv2.resize(s, (target_w, skel_h)) for s in skels]
    
    # Resize curve
    h_c, w_c = curve.shape[:2]
    curve_h = int(h_c * (target_w / w_c))
    curve = cv2.resize(curve, (target_w, curve_h))
    
    video_frames = []
    
    for i in tqdm(range(frames), desc="Generating Video"):
        progress = i / (frames - 1)
        
        # Determine which skeleton frame to show based on progress
        if progress < 0.33:
            alpha = progress / 0.33
            skel_frame = cv2.addWeighted(skels[0], 1 - alpha, skels[1], alpha, 0)
        elif progress < 0.66:
            alpha = (progress - 0.33) / 0.33
            skel_frame = cv2.addWeighted(skels[1], 1 - alpha, skels[2], alpha, 0)
        else:
            alpha = (progress - 0.66) / 0.34
            skel_frame = cv2.addWeighted(skels[2], 1 - alpha, skels[2], alpha, 0)
            
        # Draw sweeping line on curve
        curve_frame = curve.copy()
        
        # Calculate line x-coordinate. We want it to map to the plot area.
        start_x = int(target_w * 0.12)
        end_x = int(target_w * 0.90)
        line_x = int(start_x + (end_x - start_x) * progress)
        
        cv2.line(curve_frame, (line_x, 0), (line_x, curve_h), (0, 0, 255), 4)
        
        # Combine vertically
        combined = np.vstack([skel_frame, curve_frame])
        
        # Convert BGR to RGB
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        video_frames.append(combined_rgb)
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), video_frames, fps=30, macro_block_size=None)
    print(f"Saved {out_path}")

def main():
    base_dir = Path("/home/nky/BadmintonGRF")
    out_vid = base_dir / "docs/demo_content/grf_sync_demo.mp4"
    create_simulated_video(base_dir, out_vid, frames=90)

if __name__ == "__main__":
    main()
