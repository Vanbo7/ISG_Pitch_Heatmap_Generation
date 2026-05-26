import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import argparse
import json
import sys
import cv2
from pathlib import Path


def generate_heatmap_for_level(
    df,
    level_id,
    bg_path,
    output_path,
    grid_size=30,
    pitch_min=0.0,
    pitch_max=200.0,
    video_w=None,
    video_h=None,
):
    """
    Generate a single pitch heatmap for one level.
    df: DataFrame with x_center, y_center, f0_hz (already filtered to this level)
    level_id: level number for title
    bg_path: path to background image (level{N}.png)
    output_path: where to save the heatmap
    """
    df = df[(df["f0_hz"] >= pitch_min) & (df["f0_hz"] <= pitch_max)]
    if df.empty:
        return False

    bg = mpimg.imread(bg_path)
    h, w = bg.shape[:2]

    if video_w is None or video_h is None:
        video_w = int(df["x_center"].max()) + 1
        video_h = int(df["y_center"].max()) + 1

    group = df.copy()
    if w == video_w and h == video_h:
        group["x_bin"] = (group["x_center"] // grid_size).astype(int)
        group["y_bin"] = (group["y_center"] // grid_size).astype(int)
    else:
        scale_x = w / video_w
        scale_y = h / video_h
        group["x_scaled"] = group["x_center"] * scale_x
        group["y_scaled"] = group["y_center"] * scale_y
        group["x_bin"] = (group["x_scaled"] // grid_size).astype(int)
        group["y_bin"] = (group["y_scaled"] // grid_size).astype(int)

    heatmap_data = group.groupby(["y_bin", "x_bin"])["f0_hz"].mean().reset_index()
    heatmap_matrix = np.full((h // grid_size, w // grid_size), np.nan)
    for _, row in heatmap_data.iterrows():
        yb, xb = int(row["y_bin"]), int(row["x_bin"])
        if yb < heatmap_matrix.shape[0] and xb < heatmap_matrix.shape[1]:
            heatmap_matrix[yb, xb] = row["f0_hz"]

    plt.figure(figsize=(12, 8))
    if bg.ndim == 2:
        plt.imshow(bg, extent=[0, w, h, 0], cmap="gray")
    else:
        plt.imshow(bg, extent=[0, w, h, 0])

    im = plt.imshow(
        heatmap_matrix,
        cmap="coolwarm",
        alpha=0.5,
        extent=[0, w, h, 0],
        origin="upper",
    )
    plt.colorbar(im, label="Average Pitch (Hz)")
    plt.title(
        f"Watergirl Absolute Pitch Heatmap - Level {int(level_id)} "
        f"(Filtered {pitch_min}–{pitch_max} Hz, aggregated)"
    )
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def main():
    # === Parse command-line arguments ===
    parser = argparse.ArgumentParser(
        description="Generate pitch heatmaps from merged YOLO and F0 data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py map.py --input merged_watergirl_f0.csv --levels levels.json
  py map.py -i merged.csv -l levels.json --output-dir game1/heatmaps
  py map.py -i merged.csv -l levels.json --bg-dir game1/lvls_imgs
  py map.py -i merged.csv -l levels.json --video gameplay.mp4 --extract-frames
  py map.py -i merged.csv -l levels.json --video gameplay.mp4 --extract-frames --bw
  py map.py -i merged.csv -l levels.json --video gameplay.mp4 --extract-frames --frames-dir game1/extracted_frames
    
Levels JSON format (timestamps.json style):
  [
    {"level_id": 1, "start": 0, "end": 90},
    {"level_id": 2, "start": 95.59, "end": 200}
  ]
  Also accepts legacy format with "level" instead of "level_id".
  
Note: Use --video with --extract-frames to automatically extract frames from the video
      at each level's start time. This ensures background images match video resolution.
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="game2/csv/merged_watergirl_f0.csv",
        help="Input CSV file with merged YOLO and F0 data (default: game2/csv/merged_watergirl_f0.csv)"
    )
    
    parser.add_argument(
        "--levels", "-l",
        type=str,
        required=True,
        help="JSON file with level time ranges (required)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="game1/heatmaps",
        help="Output directory for heatmap images (default: game1/heatmaps)"
    )
    
    parser.add_argument(
        "--bg-dir",
        type=str,
        default="game1/lvls_imgs",
        help="Directory containing background level images (default: game1/lvls_imgs)"
    )
    
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to video file. If provided, will extract frames at level start times and save to --bg-dir (optional)"
    )
    
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract frames from video at level start times (requires --video). Overwrites existing images in --bg-dir"
    )
    
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Separate directory to save extracted frames (optional). When set with --extract-frames, frames are also saved here for use elsewhere"
    )
    
    parser.add_argument(
        "--bw",
        action="store_true",
        help="Apply black and white (grayscale) filter to extracted frames"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        default=30,
        help="Grid size in pixels for heatmap binning (default: 20)"
    )
    
    parser.add_argument(
        "--pitch-min",
        type=float,
        default=0.0,
        help="Minimum pitch in Hz to include (default: 20.0)"
    )
    
    parser.add_argument(
        "--pitch-max",
        type=float,
        default=200.0,
        help="Maximum pitch in Hz to include (default: 200.0)"
    )
    
    args = parser.parse_args()
    
    # === Load merged dataset ===
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"❌ Error: Input CSV file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading CSV file {args.input}: {e}")
        sys.exit(1)
    
    # === Load level definitions ===
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
    
    # Normalize levels: accept either "level_id" (timestamps.json) or "level"
    normalized = []
    for level in levels:
        if not ("start" in level and "end" in level):
            print(f"❌ Error: Invalid level format. Each level must have 'start' and 'end', and either 'level' or 'level_id'.")
            print(f"   Found: {level}")
            sys.exit(1)
        level_num = level.get("level_id", level.get("level"))
        if level_num is None:
            print(f"❌ Error: Invalid level format. Each level must have 'level' or 'level_id'.")
            print(f"   Found: {level}")
            sys.exit(1)
        normalized.append({"level": level_num, "start": level["start"], "end": level["end"]})
    levels = normalized

    # === Assign level based on time_seconds ===
    def assign_level(time):
        for lvl in levels:
            if lvl["start"] <= time <= lvl["end"]:
                return lvl["level"]
        return None
    
    if "time_seconds" not in df.columns:
        print("❌ Error: CSV must contain 'time_seconds' column.")
        sys.exit(1)
    
    # Assign levels BEFORE filtering to see what we have
    df["level"] = df["time_seconds"].apply(assign_level)
    
    # Show data distribution before filtering
    print(f"📊 Data distribution before pitch filtering:")
    print(f"   Total data points: {len(df)}")
    level_counts_before = df["level"].value_counts().sort_index()
    for lvl, count in level_counts_before.items():
        if pd.notna(lvl):
            print(f"   Level {int(lvl)}: {count} data points")
    unassigned = df["level"].isna().sum()
    if unassigned > 0:
        print(f"   Unassigned (outside level ranges): {unassigned} data points")
    
    # Show time range in data
    if len(df) > 0:
        min_time = df["time_seconds"].min()
        max_time = df["time_seconds"].max()
        print(f"   Time range in data: {min_time:.2f}s to {max_time:.2f}s")
    
    # === Filter out pitch outliers ===
    df_before_filter = df.copy()
    df = df[(df["f0_hz"] >= args.pitch_min) & (df["f0_hz"] <= args.pitch_max)]
    
    # Show what was filtered out
    removed = len(df_before_filter) - len(df)
    if removed > 0:
        print(f"✅ Filtered pitch data: {args.pitch_min}–{args.pitch_max} Hz")
        print(f"   Removed {removed} data points outside pitch range ({len(df)} remaining)")
    else:
        print(f"✅ All data within pitch range: {args.pitch_min}–{args.pitch_max} Hz")
    
    # === Extract frames from video if requested ===
    video_w = None
    video_h = None
    if args.video and args.extract_frames:
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
        
        # Open video to get dimensions
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {video_path}")
            sys.exit(1)
        
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"📐 Video dimensions: {video_w} x {video_h} pixels, FPS: {fps}")
        
        # Ensure bg-dir exists
        os.makedirs(args.bg_dir, exist_ok=True)
        # Ensure frames-dir exists if specified
        if args.frames_dir:
            os.makedirs(args.frames_dir, exist_ok=True)
        
        # Extract frame at start time for each level
        print(f"🎬 Extracting frames from video at level start times...")
        for level_info in levels:
            level_num = level_info["level"]
            start_time = level_info["start"]
            
            # Calculate frame number from timestamp
            frame_number = int(start_time * fps)
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Apply black and white filter if requested
                if args.bw:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Save to bg-dir (used by heatmaps)
                output_path = os.path.join(args.bg_dir, f"level{int(level_num)}.png")
                cv2.imwrite(output_path, frame)
                filter_status = " (B&W)" if args.bw else ""
                print(f"  ✅ Extracted Level {int(level_num)} frame at {start_time:.2f}s -> {output_path}{filter_status}")
                
                # Also save to frames-dir if specified (for use elsewhere)
                if args.frames_dir:
                    frames_path = os.path.join(args.frames_dir, f"level{int(level_num)}.png")
                    cv2.imwrite(frames_path, frame)
                    print(f"     → also saved to {frames_path}")
            else:
                print(f"  ⚠️ Could not extract frame for Level {int(level_num)} at {start_time:.2f}s")
        
        cap.release()
        print(f"✅ Frame extraction complete!")
    
    # === Determine video frame dimensions ===
    if video_w is None or video_h is None:
        # Estimate from data (fallback)
        video_w = int(df["x_center"].max()) + 1  # Add 1 to account for 0-indexing
        video_h = int(df["y_center"].max()) + 1
        print(f"📐 Estimated video frame dimensions from data: {video_w} x {video_h} pixels")
    else:
        print(f"📐 Using video frame dimensions: {video_w} x {video_h} pixels")
    
    grid_size = args.grid_size
    
    # === Ensure output directory exists ===
    os.makedirs(args.output_dir, exist_ok=True)
    
    # === Get all levels from JSON and check which have data ===
    levels_with_data = set(df["level"].dropna().unique())
    all_levels = [lvl["level"] for lvl in levels]
    missing_data_levels = [lvl for lvl in all_levels if lvl not in levels_with_data]
    
    if missing_data_levels:
        print(f"⚠️ Warning: No data found for levels: {missing_data_levels}")
        print(f"   (These levels will be skipped)")
    
    # === Create per-level heatmaps of absolute pitch values ===
    processed = 0
    skipped_no_data = 0
    skipped_no_image = 0
    
    for lvl, group in df.groupby("level"):
        if group.empty:
            skipped_no_data += 1
            print(f"⏭️ Skipping Level {int(lvl)}: no data")
            continue

        # Load background image first to get its dimensions
        bg_path = os.path.join(args.bg_dir, f"level{int(lvl)}.png")
        try:
            bg = mpimg.imread(bg_path)
        except FileNotFoundError:
            skipped_no_image += 1
            print(f"⚠️ Skipping Level {int(lvl)}: No background image found at {bg_path}")
            continue
    
        h, w = bg.shape[:2]
        print(f"📐 Level {int(lvl)} background image dimensions: {w} x {h} pixels")
        
        # If background image matches video dimensions, no scaling needed
        if w == video_w and h == video_h:
            # Direct mapping - no scaling required
            group_scaled = group.copy()
            group_scaled["x_bin"] = (group_scaled["x_center"] // grid_size).astype(int)
            group_scaled["y_bin"] = (group_scaled["y_center"] // grid_size).astype(int)
        else:
            # Scale coordinates from video frame space to background image space
            scale_x = w / video_w
            scale_y = h / video_h
            group_scaled = group.copy()
            group_scaled["x_scaled"] = group_scaled["x_center"] * scale_x
            group_scaled["y_scaled"] = group_scaled["y_center"] * scale_y
            
            # Calculate bins based on scaled coordinates aligned to background image
            group_scaled["x_bin"] = (group_scaled["x_scaled"] // grid_size).astype(int)
            group_scaled["y_bin"] = (group_scaled["y_scaled"] // grid_size).astype(int)
    
        # Average pitch (f0_hz) per grid cell
        heatmap_data = group_scaled.groupby(["y_bin", "x_bin"])["f0_hz"].mean().reset_index()
    
        # Build heatmap grid aligned to image dimensions
        heatmap_matrix = np.full((h // grid_size, w // grid_size), np.nan)
        for _, row in heatmap_data.iterrows():
            yb, xb = int(row["y_bin"]), int(row["x_bin"])
            if yb < heatmap_matrix.shape[0] and xb < heatmap_matrix.shape[1]:
                heatmap_matrix[yb, xb] = row["f0_hz"]
    
        # Plot
        plt.figure(figsize=(12, 8))
        if bg.ndim == 2:
            plt.imshow(bg, extent=[0, w, h, 0], cmap='gray')
        else:
            plt.imshow(bg, extent=[0, w, h, 0])
    
        # Overlay absolute pitch values
        im = plt.imshow(
            heatmap_matrix,
            cmap="coolwarm",
            alpha=0.5,
            extent=[0, w, h, 0],
            origin="upper"
        )
    
        plt.colorbar(im, label="Average Pitch (Hz)")
        plt.title(f"Watergirl Absolute Pitch Heatmap - Level {int(lvl)} (Filtered {args.pitch_min}–{args.pitch_max} Hz)")
        plt.axis("off")
        
        output_path = os.path.join(args.output_dir, f"heatmap_level_{int(lvl)}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved filtered absolute pitch heatmap for Level {int(lvl)} to {output_path}")
        processed += 1
    
    # === Summary ===
    print(f"\n📊 Summary:")
    print(f"   ✅ Processed: {processed} levels")
    if skipped_no_data > 0:
        print(f"   ⏭️ Skipped (no data): {skipped_no_data} levels")
    if skipped_no_image > 0:
        print(f"   ⚠️ Skipped (no image): {skipped_no_image} levels")
        print(f"   💡 Tip: Use --video with --extract-frames to automatically generate background images")

if __name__ == "__main__":
    main()