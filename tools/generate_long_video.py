import cv2
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm

def create_long_video(base_dir, out_path):
    # We will use the raw video and curve from the asset pack, but loop/extend it
    # to simulate a longer sequence. We will put video on the left, curve on the right.
    
    rank_dir = base_dir / "runs/fig1_asset_pack_v6/per_rank/rank_001_cam04"
    curve_path = rank_dir / "curve.png"
    
    # Let's see if we can find the actual mp4 to get a longer sequence
    # Since we can't easily run the model right now, we will simulate a longer
    # sequence by concatenating the raw_midframes or using the skeleton frames
    # and sweeping the curve line slower.
    
    skel_paths = [
        rank_dir / "singles_9" / "skeleton_t1.png",
        rank_dir / "singles_9" / "skeleton_t2.png",
        rank_dir / "singles_9" / "skeleton_t3.png",
        rank_dir / "singles_9" / "skeleton_t2.png",
        rank_dir / "singles_9" / "skeleton_t1.png"
    ]
    
    if not curve_path.exists() or not all(p.exists() for p in skel_paths):
        print("Missing source images!")
        return
        
    curve = cv2.imread(str(curve_path))
    skels = [cv2.imread(str(p)) for p in skel_paths]
    
    # Target dimensions for layout (Left: Video, Right: Curve)
    # Let's make the height the same.
    target_h = 600
    
    # Resize skeletons
    h_s, w_s = skels[0].shape[:2]
    skel_w = int(w_s * (target_h / h_s))
    skels = [cv2.resize(s, (skel_w, target_h)) for s in skels]
    
    # Resize curve
    h_c, w_c = curve.shape[:2]
    curve_w = int(w_c * (target_h / h_c))
    curve = cv2.resize(curve, (curve_w, target_h))
    
    # Calculate crop for curve to remove margins if possible, or just use as is
    
    frames = 150 # 5 seconds at 30fps
    video_frames = []
    
    for i in tqdm(range(frames), desc="Generating Long Video"):
        progress = i / (frames - 1)
        
        # Determine skeleton frame (cycle through them)
        num_skels = len(skels)
        skel_idx = int(progress * (num_skels - 1) * 3) % num_skels # loop 3 times
        next_idx = (skel_idx + 1) % num_skels
        
        local_prog = (progress * (num_skels - 1) * 3) % 1.0
        
        skel_frame = cv2.addWeighted(skels[skel_idx], 1 - local_prog, skels[next_idx], local_prog, 0)
            
        # Draw sweeping line on curve
        curve_frame = curve.copy()
        
        start_x = int(curve_w * 0.12)
        end_x = int(curve_w * 0.90)
        line_x = int(start_x + (end_x - start_x) * progress)
        
        cv2.line(curve_frame, (line_x, 0), (line_x, target_h), (0, 0, 255), 4)
        
        # Combine horizontally (Left: Video, Right: Curve)
        combined = np.hstack([skel_frame, curve_frame])
        
        # Convert BGR to RGB
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        video_frames.append(combined_rgb)
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), video_frames, fps=30, macro_block_size=None)
    print(f"Saved {out_path}")

def main():
    base_dir = Path("/home/nky/BadmintonGRF")
    out_vid = base_dir / "docs/demo_content/grf_sync_demo_long.mp4"
    create_long_video(base_dir, out_vid)

if __name__ == "__main__":
    main()
