"""
Extract frames from a video at level start times.
Useful when you only need the images (e.g. for backgrounds, analysis) without running heatmap generation.
"""
import os
import argparse
import json
import sys
import cv2
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video at level start times",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py extract_frames.py --video gameplay.mp4 --levels levels.json
  py extract_frames.py -v gameplay.mp4 -l levels.json -o game1/extracted_frames
  py extract_frames.py -v gameplay.mp4 -l levels.json --bw

Levels JSON format:
  [
    {"level": 1, "start": 0, "end": 90},
    {"level": 2, "start": 95.59, "end": 200}
  ]
        """
    )
    
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to video file"
    )
    
    parser.add_argument(
        "--levels", "-l",
        type=str,
        required=True,
        help="JSON file with level time ranges"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="lvls_imgs",
        help="Output directory for extracted frames (default: lvls_imgs)"
    )
    
    parser.add_argument(
        "--bw",
        action="store_true",
        help="Apply black and white (grayscale) filter to extracted frames"
    )
    
    args = parser.parse_args()
    
    # === Resolve video path ===
    video_path = Path(args.video)
    if not video_path.is_absolute():
        script_dir = Path(__file__).parent.absolute()
        candidate = script_dir / video_path
        if candidate.exists():
            video_path = candidate
        elif not video_path.exists():
            print(f"❌ Error: Video file not found: {args.video}")
            sys.exit(1)
    
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # === Load levels ===
    try:
        with open(args.levels, 'r') as f:
            levels = json.load(f)
        print(f"✅ Loaded {len(levels)} levels from {args.levels}")
    except FileNotFoundError:
        print(f"❌ Error: Levels file not found: {args.levels}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON file {args.levels}: {e}")
        sys.exit(1)
    
    for level in levels:
        if not all(key in level for key in ["level", "start", "end"]):
            print(f"❌ Error: Invalid level format. Each level must have 'level', 'start', and 'end' keys.")
            print(f"   Found: {level}")
            sys.exit(1)
    
    # === Open video ===
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        sys.exit(1)
    
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📐 Video dimensions: {video_w} x {video_h} pixels, FPS: {fps}")
    
    # === Ensure output directory exists ===
    os.makedirs(args.output_dir, exist_ok=True)
    
    # === Extract frames ===
    print(f"🎬 Extracting frames at level start times...")
    for level_info in levels:
        level_num = level_info["level"]
        start_time = level_info["start"]
        
        frame_number = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            if args.bw:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            output_path = os.path.join(args.output_dir, f"level{int(level_num)}.png")
            cv2.imwrite(output_path, frame)
            filter_status = " (B&W)" if args.bw else ""
            print(f"  ✅ Level {int(level_num)} at {start_time:.2f}s -> {output_path}{filter_status}")
        else:
            print(f"  ⚠️ Could not extract frame for Level {int(level_num)} at {start_time:.2f}s")
    
    cap.release()
    print(f"✅ Done! Frames saved to {args.output_dir}")


if __name__ == "__main__":
    main()
