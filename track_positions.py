import argparse
import sys
import pandas as pd
import cv2
from ultralytics import YOLO
from pathlib import Path

def main():
    # === Parse command-line arguments ===
    parser = argparse.ArgumentParser(
        description="Track character positions in a video using YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py track_positions.py --video gameplay.mp4
  py track_positions.py -v gameplay.mp4 -o results.csv
  py track_positions.py --character fireboy  # Track fireboy
  py track_positions.py -c watergirl  # Track watergirl (default)
  py track_positions.py  # Uses default values
        """
    )
    
    parser.add_argument(
        "--video", "-v",
        type=str,
        default="game1_right.mp4",
        help="Path to input video file (default: game1_right.mp4)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV filename (default: positions_<video_name>.csv)"
    )
    
    parser.add_argument(
        "--character", "-c",
        type=str,
        choices=["watergirl", "fireboy"],
        default="watergirl",
        help="Character to track (watergirl or fireboy) (default: watergirl)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.60,
        help="Minimum confidence threshold for detections (default: 0.60)"
    )

    args = parser.parse_args()
    
    # Assign character for easier use
    character = args.character

    # === Setup paths ===
    SCRIPT_DIR = Path(__file__).parent.absolute()
    MODEL_PATH = SCRIPT_DIR / "YOLOv8n" / "runs" / "detect" / "train2" / "weights" / "best.pt"

    video_path = Path(args.video)
    if not video_path.is_absolute():
        candidate = SCRIPT_DIR / video_path
        if candidate.exists():
            video_path = candidate
        elif video_path.exists():
            video_path = video_path.resolve()
        else:
            print(f"❌ Error: Video not found: {args.video}")
            sys.exit(1)
    else:
        video_path = video_path.resolve()

    # === Verify model file ===
    if not MODEL_PATH.exists():
        print(f"❌ Error: Trained model not found at: {MODEL_PATH}")
        sys.exit(1)

    # === Determine output path ===
    video_stem = video_path.stem
    output_csv = Path(args.output) if args.output else (SCRIPT_DIR / f"positions_{video_stem}_{character}.csv")

    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Using YOLO model: {MODEL_PATH}")
    print(f"Video file: {video_path}")
    print(f"Tracking character: {character}")
    print(f"Output CSV: {output_csv}")
    print(f"Confidence threshold: {args.confidence}")

    # === Load model ===
    model = YOLO(str(MODEL_PATH))

    # === Get FPS ===
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"🎞️ Detected FPS: {fps}")

    # === Run YOLO inference ===
    results = model(str(video_path), stream=True, save=True)

    detections = []
    for frame_idx, r in enumerate(results):
        for box in r.boxes.data.tolist():
            x_min, y_min, x_max, y_max, conf, cls = box
            if model.names[int(cls)] != character:
                continue
            if conf < args.confidence:
                continue

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            detections.append({
                "frame": frame_idx,
                "class": model.names[int(cls)],
                "x_center": x_center,
                "y_center": y_center,
                "confidence": conf,
                "time_seconds": frame_idx / fps
            })

    # === Save to CSV ===
    df = pd.DataFrame(detections)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(output_csv), index=False)

    print(f"Saved {len(df)} {character.capitalize()} detections with timestamps to {output_csv}")

if __name__ == "__main__":
    main()
