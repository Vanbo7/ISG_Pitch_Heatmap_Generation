import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import argparse
import json
import sys
from pathlib import Path
import cv2


def compute_kde_pitch(df, bins_x, bins_y, video_w, video_h, bandwidth=0.05, min_density=1e-12):
    """
    Compute KDE-smoothed local average pitch in true Hz units.

    For each point on a (bins_x * bins_y) grid, the pitch estimate is a
    Gaussian-kernel-weighted average of all observed f0 values.  Unlike a
    density-weighted KDE, this returns actual Hz values rather than
    visit-frequency, so regions the player passes through quickly but speaks
    at a consistent pitch still show that pitch correctly.

    bandwidth is in normalised coordinate space [0, 1].
    Returns a 2-D numpy array of shape (bins_y, bins_x) with NaN where the
    kernel weight sum is below min_density.
    """
    # Normalise positions to [0, 1]
    x  = df["x_center"].to_numpy(dtype=float) / video_w
    y  = df["y_center"].to_numpy(dtype=float) / video_h
    f0 = df["f0_hz"].to_numpy(dtype=float)

    pts = np.column_stack([x, y])          # (N, 2)

    # Regular grid in normalised space
    grid_x = np.linspace(0, 1, bins_x)
    grid_y = np.linspace(0, 1, bins_y)
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="xy")
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])   # (M, 2)

    # Squared distances from every grid point to every sample  →  (M, N)
    dx    = grid_pts[:, 0][:, None] - pts[:, 0][None, :]
    dy    = grid_pts[:, 1][:, None] - pts[:, 1][None, :]
    dist2 = dx * dx + dy * dy

    # Gaussian kernel weights
    weights = np.exp(-0.5 * dist2 / (bandwidth ** 2))

    denom = weights.sum(axis=1)
    numer = weights @ f0

    pitch_flat = np.where(denom > min_density, numer / denom, np.nan)

    # Reshape to (bins_y, bins_x) — origin="upper" layout
    return pitch_flat.reshape(bins_y, bins_x)


def generate_kde_heatmap_for_level(
    df,
    level_id,
    bg_path,
    output_path,
    video_w,
    video_h,
    bins_x=60,
    bandwidth=0.05,
    pitch_min=40.0,
    pitch_max=200.0,
    alpha=0.5,
    cmap="coolwarm",
):
    """
    Generate one KDE pitch heatmap for a single level and save it.
    """
    df = df[(df["f0_hz"] >= pitch_min) & (df["f0_hz"] <= pitch_max)].copy()
    if df.empty:
        return False

    bg   = mpimg.imread(bg_path)
    h, w = bg.shape[:2]

    # Derive bins_y so the grid preserves the image aspect ratio
    bins_y = max(1, int(round(bins_x * h / w)))

    pitch_grid = compute_kde_pitch(df, bins_x, bins_y, video_w=video_w, video_h=video_h, bandwidth=bandwidth)

    plt.figure(figsize=(12, 8))
    if bg.ndim == 2:
        plt.imshow(bg, extent=[0, w, h, 0], cmap="gray")
    else:
        plt.imshow(bg, extent=[0, w, h, 0])

    im = plt.imshow(
        pitch_grid,
        origin="upper",
        aspect="auto",
        cmap=cmap,
        alpha=alpha,
        extent=[0, w, h, 0],
        vmin=pitch_min,
        vmax=pitch_max,
    )

    plt.colorbar(im, label="KDE Average Pitch (Hz)")
    plt.title(
        f"Watergirl KDE Pitch Heatmap – Level {int(level_id)} "
        f"(bw={bandwidth}, {pitch_min}–{pitch_max} Hz)"
    )
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate smooth KDE pitch heatmaps (Gaussian-weighted average Hz).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python map_gaussian.py --input merged_watergirl_f0.csv --levels levels.json
  python map_gaussian.py -i merged.csv -l levels.json --bandwidth 0.03 --bins-x 80
  python map_gaussian.py -i merged.csv -l levels.json --output-dir results/kde

Levels JSON format (same as map.py):
  [
    {"level_id": 1, "start": 0,     "end": 90},
    {"level_id": 2, "start": 95.59, "end": 200}
  ]
  Also accepts legacy format with "level" instead of "level_id".
        """
    )

    parser.add_argument("--input",  "-i", type=str,
                        default="game2/csv/merged_watergirl_f0.csv",
                        help="Input CSV file with merged YOLO and F0 data")
    parser.add_argument("--levels", "-l", type=str, required=True,
                        help="JSON file with level time ranges (required)")
    parser.add_argument("--output-dir", type=str, default="game1/heatmaps_kde",
                        help="Output directory for heatmap images (default: game1/heatmaps_kde)")
    parser.add_argument("--bg-dir", type=str, default="game1/lvls_imgs",
                        help="Directory containing background level images (default: game1/lvls_imgs)")
    parser.add_argument("--bins-x", type=int, default=60,
                        help="Number of grid columns (default: 60); rows are set automatically to preserve aspect ratio")
    parser.add_argument("--bandwidth", type=float, default=0.05,
                        help="KDE bandwidth in normalised [0,1] coordinate space (default: 0.05). "
                             "Smaller = more detail, larger = smoother")
    parser.add_argument("--pitch-min", type=float, default=40.0,
                        help="Minimum pitch in Hz to include (default: 40.0)")
    parser.add_argument("--pitch-max", type=float, default=200.0,
                        help="Maximum pitch in Hz to include (default: 200.0)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Overlay transparency 0–1 (default: 0.5)")
    parser.add_argument("--cmap", type=str, default="coolwarm",
                        help="Matplotlib colormap (default: coolwarm)")
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="Path to video file. Used with --extract-frames to pull background images automatically")
    parser.add_argument("--extract-frames", action="store_true",
                        help="Extract frames from video at level start times and save to --bg-dir (requires --video)")
    parser.add_argument("--frames-dir", type=str, default=None,
                        help="Optional extra directory to also save extracted frames")
    parser.add_argument("--bw", action="store_true",
                        help="Apply grayscale filter to extracted frames")

    args = parser.parse_args()

    # ── Load CSV ──────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"❌ Error: Input CSV file not found: {args.input}")
        sys.exit(1)

    required_cols = ["x_center", "y_center", "f0_hz"]
    # time column can be named "time_seconds" or "frame"
    time_col = "time_seconds" if "time_seconds" in df.columns else "frame"
    if time_col not in df.columns:
        print("❌ Error: CSV must contain 'time_seconds' or 'frame' column.")
        sys.exit(1)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Error: Missing required CSV columns: {missing}")
        sys.exit(1)

    # ── Load level definitions ────────────────────────────────────────────────
    try:
        with open(args.levels) as f:
            levels_raw = json.load(f)
        print(f"✅ Loaded {len(levels_raw)} levels from {args.levels}")
    except FileNotFoundError:
        print(f"❌ Error: Levels file not found: {args.levels}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON {args.levels}: {e}")
        sys.exit(1)

    # Normalise key names (support both "level_id" and legacy "level")
    levels = []
    for entry in levels_raw:
        lvl_id = entry.get("level_id", entry.get("level"))
        if lvl_id is None:
            print(f"⚠️ Skipping malformed level entry (no level_id/level key): {entry}")
            continue
        levels.append({"level": float(lvl_id), "start": entry["start"], "end": entry["end"]})

    # ── Assign each row to a level ────────────────────────────────────────────
    def assign_level(t):
        for lvl in levels:
            if lvl["start"] <= t <= lvl["end"]:
                return lvl["level"]
        return None

    df["level"] = df[time_col].apply(assign_level)

    print(f"📊 Data distribution before pitch filtering:")
    print(f"   Total rows: {len(df)}")
    for lvl_id, cnt in df["level"].value_counts().sort_index().items():
        if pd.notna(lvl_id):
            print(f"   Level {int(lvl_id)}: {cnt} rows")
    unassigned = df["level"].isna().sum()
    if unassigned:
        print(f"   Unassigned (outside level ranges): {unassigned} rows")

    # ── Filter pitch ──────────────────────────────────────────────────────────
    before = len(df)
    df = df[(df["f0_hz"] >= args.pitch_min) & (df["f0_hz"] <= args.pitch_max)].copy()
    removed = before - len(df)
    if removed:
        print(f"✅ Pitch filter {args.pitch_min}–{args.pitch_max} Hz: removed {removed} rows ({len(df)} remaining)")
    if df.empty:
        print("❌ No data remains after pitch filtering.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Extract frames from video if requested ────────────────────────────────
    video_w = video_h = None
    if args.video and args.extract_frames:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            candidate = Path(__file__).parent.absolute() / video_path
            if candidate.exists():
                video_path = candidate
        if not video_path.exists():
            print(f"❌ Error: Video file not found: {args.video}")
            sys.exit(1)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Error: Could not open video: {video_path}")
            sys.exit(1)

        fps      = cap.get(cv2.CAP_PROP_FPS)
        video_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📐 Video dimensions: {video_w} x {video_h}, FPS: {fps}")

        os.makedirs(args.bg_dir, exist_ok=True)
        if args.frames_dir:
            os.makedirs(args.frames_dir, exist_ok=True)

        print("🎬 Extracting frames at level start times...")
        for lvl in levels:
            frame_num = int(lvl["start"] * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                if args.bw:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out_bg = os.path.join(args.bg_dir, f"level{int(lvl['level'])}.png")
                cv2.imwrite(out_bg, frame)
                bw_tag = " (B&W)" if args.bw else ""
                print(f"  ✅ Level {int(lvl['level'])} @ {lvl['start']:.2f}s → {out_bg}{bw_tag}")
                if args.frames_dir:
                    also = os.path.join(args.frames_dir, f"level{int(lvl['level'])}.png")
                    cv2.imwrite(also, frame)
                    print(f"     → also saved to {also}")
            else:
                print(f"  ⚠️ Could not extract frame for Level {int(lvl['level'])} at {lvl['start']:.2f}s")
        cap.release()
        print("✅ Frame extraction complete!")

    # Fall back to estimating dimensions from data if video wasn't provided
    if video_w is None or video_h is None:
        if "video_w" in df.columns and "video_h" in df.columns:
            video_w = int(df["video_w"].iloc[0])
            video_h = int(df["video_h"].iloc[0])
            print(f"📐 Using video dimensions from CSV: {video_w} x {video_h}")
        else:
            video_w = int(df["x_center"].max()) + 1
            video_h = int(df["y_center"].max()) + 1
            print(f"📐 Estimated video dimensions from data: {video_w} x {video_h}")

    # ── Generate one heatmap per level ───────────────────────────────────────
    processed = skipped_no_data = skipped_no_image = 0

    for lvl_id, group in df.groupby("level"):
        if group.empty:
            skipped_no_data += 1
            print(f"⏭️ Skipping Level {int(lvl_id)}: no data after filtering")
            continue

        bg_path = os.path.join(args.bg_dir, f"level{int(lvl_id)}.png")
        if not os.path.exists(bg_path):
            skipped_no_image += 1
            print(f"⚠️ Skipping Level {int(lvl_id)}: background image not found at {bg_path}")
            continue

        out_path = os.path.join(args.output_dir, f"heatmap_kde_level_{int(lvl_id)}.png")

        ok = generate_kde_heatmap_for_level(
            df=group,
            level_id=lvl_id,
            bg_path=bg_path,
            output_path=out_path,
            video_w=video_w,
            video_h=video_h,
            bins_x=args.bins_x,
            bandwidth=args.bandwidth,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max,
            alpha=args.alpha,
            cmap=args.cmap,
        )

        if ok:
            print(f"✅ Saved KDE heatmap for Level {int(lvl_id)} → {out_path}")
            processed += 1
        else:
            skipped_no_data += 1
            print(f"⏭️ Skipping Level {int(lvl_id)}: no data in pitch range")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n📊 Summary:")
    print(f"   ✅ Processed : {processed} levels")
    if skipped_no_data:
        print(f"   ⏭️ Skipped (no data)  : {skipped_no_data} levels")
    if skipped_no_image:
        print(f"   ⚠️ Skipped (no image) : {skipped_no_image} levels")
        print(f"   💡 Tip: add --video <path> --extract-frames to generate background images automatically")


if __name__ == "__main__":
    main()