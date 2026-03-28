import cv2
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm

def create_sliding_gif(raw_path, skel_path, out_path, frames=60):
    raw = cv2.imread(str(raw_path))
    skel = cv2.imread(str(skel_path))
    
    # Resize to a reasonable size for GIF (e.g. width 800)
    target_w = 800
    h, w = raw.shape[:2]
    target_h = int(h * (target_w / w))
    
    raw = cv2.resize(raw, (target_w, target_h))
    skel = cv2.resize(skel, (target_w, target_h))
    
    gif_frames = []
    
    # Create a smooth sine-based sweeping animation
    for i in tqdm(range(frames), desc="Generating GIF"):
        progress = i / (frames - 1)
        # Sine wave from 0 to 1 and back to 0
        sweep_pos = (np.sin(progress * 2 * np.pi - np.pi/2) + 1) / 2
        
        split_x = int(target_w * sweep_pos)
        
        # Combine images
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
        
        # Add text labels
        cv2.putText(frame, "Raw Video", (target_w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Pose & GRF", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gif_frames.append(frame_rgb)
        
    imageio.mimsave(str(out_path), gif_frames, fps=30)
    print(f"Saved {out_path}")

def main():
    base_dir = Path("docs/demo_content")
    raw_path = base_dir / "raw_sample.png"
    skel_path = base_dir / "skeleton_sample.png"
    out_gif = base_dir / "slider_demo.gif"
    
    if raw_path.exists() and skel_path.exists():
        create_sliding_gif(raw_path, skel_path, out_gif, frames=90)
    else:
        print("Input images not found!")

if __name__ == "__main__":
    main()
