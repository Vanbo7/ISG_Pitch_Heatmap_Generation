import argparse
import json
import os
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Optional: OpenCV for video metadata (recommended)
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def load_timestamps(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    levels = []
    for entry in data:
        levels.append(
            {
                "level_id": str(entry["level_id"]),
                "start": float(entry["start"]),
                "end": float(entry["end"]),
            }
        )
    return levels


def assign_level(time_sec: float, levels):
    for lvl in levels:
        if lvl["start"] <= time_sec <= lvl["end"]:
            return lvl["level_id"]
    return None


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def find_level_bg(levels_img_dir: str, level_id: str):
    candidates = [
        Path(levels_img_dir) / f"Level {level_id}.png",
        Path(levels_img_dir) / f"Level {int(level_id)}.png",
        Path(levels_img_dir) / f"level{level_id}.png",
        Path(levels_img_dir) / f"level{int(level_id)}.png",
        Path(levels_img_dir) / f"Level_{level_id}.png",
        Path(levels_img_dir) / f"Level_{int(level_id)}.png",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    hits = sorted(glob.glob(str(Path(levels_img_dir) / f"*{level_id}*.png")))
    if hits:
        return hits[0]
    return None


def find_video_for_game(root: str, game_dir: str, game_name: str):
    """
    Supports:
      A) games/game1/game1.mp4  (video inside folder)
      B) games/game1/video.mp4  (standardized name)
      C) games/game1.mp4        (video at root)
      D) any mp4/mkv/avi/mov found in folder
    """
    # 1) inside game folder, named like game1.mp4
    candidates = [
        os.path.join(game_dir, f"{game_name}.mp4"),
        os.path.join(game_dir, f"{game_name}.mkv"),
        os.path.join(game_dir, f"{game_name}.avi"),
        os.path.join(game_dir, f"{game_name}.mov"),
        os.path.join(game_dir, "video.mp4"),
        os.path.join(game_dir, "video.mkv"),
        os.path.join(game_dir, "video.avi"),
        os.path.join(game_dir, "video.mov"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # 2) any video file inside game folder
    for ext in ("*.mp4", "*.mkv", "*.avi", "*.mov"):
        vids = sorted(glob.glob(os.path.join(game_dir, ext)))
        if vids:
            return vids[0]

    # 3) at root, named like games/game1.mp4
    for ext in (".mp4", ".mkv", ".avi", ".mov"):
        p = os.path.join(root, f"{game_name}{ext}")
        if os.path.exists(p):
            return p

    return None


def get_video_dims(video_path: str):
    if not HAS_CV2:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if w > 0 and h > 0:
        return w, h
    return None


def generate_aggregated_heatmap(
    df_level: pd.DataFrame,
    level_id: str,
    bg_path: str,
    output_path: str,
    bins_x: int,
    pitch_min: float,
    pitch_max: float,
    character_label: str,
):
    # Use ALL data (only pitch filter, no bin-count filtering)
    df_level = df_level[(df_level["f0_hz"] >= pitch_min) & (df_level["f0_hz"] <= pitch_max)].copy()
    if df_level.empty:
        return False, "no_data_after_pitch_filter"

    bg = mpimg.imread(bg_path)
    bg_h, bg_w = bg.shape[:2]

    # Consistent visual resolution across levels
    grid_size = max(1, bg_w // bins_x)

    # Scale from per-game coordinate space into bg image space
    df_level["x_scaled"] = df_level["x_center"] * (bg_w / df_level["video_w"])
    df_level["y_scaled"] = df_level["y_center"] * (bg_h / df_level["video_h"])

    # Clip to bounds (does not delete data; keeps it inside drawable area)
    df_level["x_scaled"] = df_level["x_scaled"].clip(0, bg_w - 1)
    df_level["y_scaled"] = df_level["y_scaled"].clip(0, bg_h - 1)

    df_level["x_bin"] = (df_level["x_scaled"] // grid_size).astype(int)
    df_level["y_bin"] = (df_level["y_scaled"] // grid_size).astype(int)

    # Mean pitch per bin (no dropping of bins)
    agg = df_level.groupby(["y_bin", "x_bin"])["f0_hz"].mean().reset_index(name="mean_f0")

    heat_h = max(1, bg_h // grid_size)
    heat_w = max(1, bg_w // grid_size)
    heatmap_matrix = np.full((heat_h, heat_w), np.nan)

    for _, row in agg.iterrows():
        yb, xb = int(row["y_bin"]), int(row["x_bin"])
        if 0 <= yb < heat_h and 0 <= xb < heat_w:
            heatmap_matrix[yb, xb] = row["mean_f0"]

    plt.figure(figsize=(12, 8))
    if bg.ndim == 2:
        plt.imshow(bg, extent=[0, bg_w, bg_h, 0], cmap="gray")
    else:
        plt.imshow(bg, extent=[0, bg_w, bg_h, 0])

    im = plt.imshow(
        heatmap_matrix,
        cmap="coolwarm",
        alpha=0.5,
        extent=[0, bg_w, bg_h, 0],
        origin="upper",
    )

    plt.colorbar(im, label="Average Pitch (Hz)")
    plt.title(f"Aggregated {character_label.title()} Pitch Heatmap - Level {int(level_id)} ({pitch_min}–{pitch_max} Hz)")
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return True, "ok"


def main():
    ap = argparse.ArgumentParser(description="Aggregate identical levels across games and generate per-level heatmaps.")
    ap.add_argument("--root", required=True, help="Root directory containing game folders (e.g., ./games)")
    ap.add_argument("--games-glob", default="game*", help="Glob for game folders inside --root (default: game*)")

    # If your merged filenames differ, script will auto-find merged*.csv in each game folder.
    ap.add_argument("--merged-pattern", default="merged*.csv", help="Glob pattern inside each game folder (default: merged*.csv)")
    ap.add_argument("--timestamps", default="timestamps.json", help="Timestamps JSON relative to each game folder")

    ap.add_argument("--levels-img-dir", required=True, help='Directory with images like "Level #.png"')
    ap.add_argument("--out-dir", default="aggregated", help="Output folder name inside --root")

    ap.add_argument(
        "--character",
        choices=["watergirl", "fireboy"],
        default="watergirl",
        help="Which character to aggregate (default: watergirl)",
    )
    ap.add_argument(
        "--fireboy",
        action="store_true",
        help="Shortcut for --character fireboy",
    )

    ap.add_argument("--min-f0", type=float, default=40.0)
    ap.add_argument("--max-f0", type=float, default=200.0)
    ap.add_argument("--min-samples-per-bin", type=int, default=10)
    ap.add_argument("--bins-x", type=int, default=120, help="How many bins across the level image width.")

    args = ap.parse_args()
    if args.fireboy:
        args.character = "fireboy"

    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, args.out_dir)
    levels_csv_dir = os.path.join(out_dir, "levels_csv")
    levels_hm_dir = os.path.join(out_dir, "levels_heatmaps")

    ensure_dir(out_dir)
    ensure_dir(levels_csv_dir)
    ensure_dir(levels_hm_dir)

    buckets = {}
    debug_rows = []
    heatmap_debug = []

    game_dirs = [d for d in glob.glob(os.path.join(root, args.games_glob)) if os.path.isdir(d)]
    game_dirs.sort()

    if not game_dirs:
        print(f"No game folders found in: {root} matching {args.games_glob}")
        return

    for gdir in game_dirs:
        game_name = os.path.basename(gdir)

        # Find merged csv (auto)
        merged_candidates = sorted(glob.glob(os.path.join(gdir, args.merged_pattern)))
        # Handle uppercase extension too
        merged_candidates += sorted(glob.glob(os.path.join(gdir, args.merged_pattern.replace(".csv", ".CSV"))))

        if not merged_candidates:
            debug_rows.append({
                "game": game_name,
                "status": "skipped_missing_merged",
                "path": os.path.join(gdir, args.merged_pattern),
            })
            continue

        preferred = [p for p in merged_candidates if args.character in os.path.basename(p).lower()]
        merged_path = preferred[0] if preferred else merged_candidates[0]

        timestamps_path = os.path.join(gdir, args.timestamps)
        if not os.path.exists(timestamps_path):
            debug_rows.append({"game": game_name, "status": "skipped_missing_timestamps", "path": timestamps_path})
            continue

        try:
            df = pd.read_csv(merged_path)
        except Exception as e:
            debug_rows.append({"game": game_name, "status": f"skipped_bad_csv:{e}", "path": merged_path})
            continue

        levels = load_timestamps(timestamps_path)

        # Filter to selected character if class column exists
        if "class" in df.columns:
            df = df[df["class"].astype(str).str.lower() == args.character].copy()

        required_cols = {"f0_hz", "time_seconds", "x_center", "y_center"}
        missing = required_cols - set(df.columns)
        if missing:
            debug_rows.append({"game": game_name, "status": f"skipped_missing_columns:{sorted(missing)}", "path": merged_path})
            continue

        # Pitch filter early
        df = df[df["f0_hz"].between(args.min_f0, args.max_f0)].copy()

        # Real video dims per game (Fix 1)
        video_path = find_video_for_game(root=root, game_dir=gdir, game_name=game_name)
        dims = get_video_dims(video_path) if video_path else None

        if dims is None:
            # Robust fallback (much safer than max)
            video_w = float(df["x_center"].quantile(0.995)) + 1.0
            video_h = float(df["y_center"].quantile(0.995)) + 1.0
            dims_source = "fallback_quantile"
        else:
            video_w, video_h = dims
            dims_source = "video_meta"

        df["level_id"] = df["time_seconds"].apply(lambda t: assign_level(float(t), levels))
        df = df.dropna(subset=["level_id"]).copy()

        df.insert(0, "game", game_name)
        df["video_w"] = float(video_w)
        df["video_h"] = float(video_h)

        for lvl, gdf in df.groupby("level_id"):
            buckets.setdefault(str(lvl), []).append(gdf)

        found_lvls = sorted(df["level_id"].unique(), key=lambda x: int(x))
        debug_rows.append({
            "game": game_name,
            "status": "ok",
            "merged_csv": merged_path,
            "video_path": video_path if video_path else "",
            "dims_source": dims_source,
            "video_w": int(video_w),
            "video_h": int(video_h),
            "rows_used": int(len(df)),
            "levels_found": ",".join(found_lvls),
        })

    for level_id, parts in sorted(buckets.items(), key=lambda kv: int(kv[0])):
        out_df = pd.concat(parts, ignore_index=True)
        out_csv = os.path.join(levels_csv_dir, f"{level_id}.csv")
        out_df.to_csv(out_csv, index=False)

        bg_path = find_level_bg(args.levels_img_dir, level_id)
        if not bg_path:
            heatmap_debug.append({"level_id": level_id, "status": "skipped_missing_bg"})
            continue

        out_png = os.path.join(levels_hm_dir, f"heatmap_level_{int(level_id)}.png")
        ok, reason = generate_aggregated_heatmap(
            df_level=out_df,
            level_id=level_id,
            bg_path=bg_path,
            output_path=out_png,
            bins_x=args.bins_x,
            pitch_min=args.min_f0,
            pitch_max=args.max_f0,
            character_label=args.character,
        )
        heatmap_debug.append({"level_id": level_id, "status": "ok" if ok else f"skipped_{reason}", "bg": bg_path})

    pd.DataFrame(debug_rows).to_csv(os.path.join(out_dir, "debug_summary.csv"), index=False)
    pd.DataFrame(heatmap_debug).to_csv(os.path.join(out_dir, "debug_heatmaps.csv"), index=False)

    print("Aggregation + heatmaps complete.")
    print(f"CSVs:     {levels_csv_dir}")
    print(f"Heatmaps: {levels_hm_dir}")
    print(f"Debug:    {os.path.join(out_dir, 'debug_summary.csv')}")


if __name__ == "__main__":
    main()