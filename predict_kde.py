import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def compute_bins(df, bins_x):
    """
    Add bin_x and bin_y using normalized coordinates.
    bins_y is chosen to preserve aspect ratio.
    """
    width = float(df["video_w"].iloc[0])
    height = float(df["video_h"].iloc[0])

    bins_y = max(1, int(round(bins_x * (height / width))))

    out = df.copy()
    out["bin_x"] = np.floor(out["x_center"] / width * bins_x).astype(int)
    out["bin_y"] = np.floor(out["y_center"] / height * bins_y).astype(int)

    out["bin_x"] = out["bin_x"].clip(0, bins_x - 1)
    out["bin_y"] = out["bin_y"].clip(0, bins_y - 1)

    return out, bins_y


def compute_kde_pitch(df, bins_x, bins_y, bandwidth=0.05, min_density=1e-12):
    """
    Compute KDE-smoothed local average pitch in true Hz units.

    bandwidth is in normalized coordinate space [0,1].
    Returns dataframe with:
        bin_x, bin_y, pitch
    """
    # Normalize positions to [0,1]
    x = (df["x_center"].to_numpy(dtype=float) / df["video_w"].to_numpy(dtype=float))
    y = (df["y_center"].to_numpy(dtype=float) / df["video_h"].to_numpy(dtype=float))
    f0 = df["f0_hz"].to_numpy(dtype=float)

    pts = np.column_stack([x, y])   # shape: (N, 2)

    # Grid in normalized space
    grid_x = np.linspace(0, 1, bins_x)
    grid_y = np.linspace(0, 1, bins_y)
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="xy")
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])   # shape: (M, 2)

    # Squared Euclidean distances from each grid point to each sample
    # result shape: (M, N)
    dx = grid_pts[:, 0][:, None] - pts[:, 0][None, :]
    dy = grid_pts[:, 1][:, None] - pts[:, 1][None, :]
    dist2 = dx * dx + dy * dy

    # Gaussian kernel weights
    # bandwidth is like sigma in normalized coordinates
    weights = np.exp(-0.5 * dist2 / (bandwidth ** 2))

    # Weighted average pitch at each grid point
    denom = weights.sum(axis=1)
    numer = weights @ f0

    pitch = np.divide(
        numer,
        denom,
        out=np.full_like(numer, np.nan, dtype=float),
        where=denom > min_density
    )

    out = pd.DataFrame({
        "bin_x": np.tile(np.arange(bins_x), bins_y),
        "bin_y": np.repeat(np.arange(bins_y), bins_x),
        "pitch": pitch
    })

    return out


def compute_metrics(merged):
    merged = merged.dropna(subset=["predicted_pitch", "actual_pitch"])

    pred = merged["predicted_pitch"].to_numpy(dtype=float)
    actual = merged["actual_pitch"].to_numpy(dtype=float)

    if len(merged) < 2:
        corr = np.nan
    else:
        corr = np.corrcoef(pred, actual)[0, 1]

    if len(merged) == 0:
        rmse = np.nan
    else:
        rmse = np.sqrt(np.mean((pred - actual) ** 2))

    return corr, rmse

def save_pitch_heatmap(grid_df, value_col, bins_x, bins_y, out_path, title):
    """
    Save a heatmap from a grid dataframe with columns:
    bin_x, bin_y, and value_col
    """
    import numpy as np
    import matplotlib.pyplot as plt

    heatmap = np.full((bins_y, bins_x), np.nan)

    for _, row in grid_df.iterrows():
        bx = int(row["bin_x"])
        by = int(row["bin_y"])
        if 0 <= bx < bins_x and 0 <= by < bins_y:
            heatmap[by, bx] = row[value_col]

    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        heatmap,
        origin="upper",
        aspect="auto",
        cmap="coolwarm",
        vmin=40,
        vmax=200,
    )
    plt.colorbar(im, label="Pitch (Hz)")
    plt.title(title)
    plt.xlabel("bin_x")
    plt.ylabel("bin_y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_kde_overlay_on_image(grid_df, value_col, bins_x, bins_y, bg_path, out_path, title,
                              vmin=40, vmax=200, cmap="coolwarm", alpha=0.5):
    bg = mpimg.imread(bg_path)
    h, w = bg.shape[:2]

    heatmap = np.full((bins_y, bins_x), np.nan)

    for _, row in grid_df.iterrows():
        bx = int(row["bin_x"])
        by = int(row["bin_y"])
        if 0 <= bx < bins_x and 0 <= by < bins_y:
            heatmap[by, bx] = row[value_col]

    plt.figure(figsize=(12, 8))

    if bg.ndim == 2:
        plt.imshow(bg, extent=[0, w, h, 0], cmap="gray")
    else:
        plt.imshow(bg, extent=[0, w, h, 0])

    im = plt.imshow(
        heatmap,
        origin="upper",
        aspect="auto",
        cmap=cmap,
        alpha=alpha,
        extent=[0, w, h, 0],
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, label=value_col)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
def compute_binned_pitch(df, bins_x, bins_y):
    x = df["x_center"].to_numpy(dtype=float) / df["video_w"].to_numpy(dtype=float)
    y = df["y_center"].to_numpy(dtype=float) / df["video_h"].to_numpy(dtype=float)
    f0 = df["f0_hz"].to_numpy(dtype=float)

    bin_x = np.floor(x * bins_x).astype(int)
    bin_y = np.floor(y * bins_y).astype(int)

    bin_x = np.clip(bin_x, 0, bins_x - 1)
    bin_y = np.clip(bin_y, 0, bins_y - 1)

    temp = pd.DataFrame({
        "bin_x": bin_x,
        "bin_y": bin_y,
        "f0_hz": f0
    })

    out = (
        temp.groupby(["bin_x", "bin_y"], as_index=False)["f0_hz"]
        .mean()
        .rename(columns={"f0_hz": "pitch"})
    )

    return out
    
def make_scatterplot(merged, out_path, title, axis_min=0, axis_max=200):
    merged = merged.dropna(subset=["predicted_pitch", "actual_pitch"])
    if merged.empty:
        return

    x = merged["predicted_pitch"]
    y = merged["actual_pitch"]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, alpha=0.75)

    plt.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", linewidth=1)
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)

    plt.xlabel("Predicted Pitch (Hz)")
    plt.ylabel("Actual Pitch (Hz)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_holdout(df, holdout_game, bins_x, bins_y, output_dir, base_name, axis_min=0, axis_max=200, bandwidth=0.05):
    """
    Run leave-one-out for a single holdout game.
    """
    df_holdout = df[df["game"] == holdout_game].copy()
    df_others = df[df["game"] != holdout_game].copy()

    if df_holdout.empty:
        print(f"Skipping {holdout_game}: no holdout data found.")
        return None

    if df_others.empty:
        print(f"Skipping {holdout_game}: no non-holdout data available.")
        return None

    actual = compute_binned_pitch(df_holdout, bins_x, bins_y).rename(columns={"pitch": "actual_pitch"})
    predicted = compute_kde_pitch(df_others, bins_x, bins_y, bandwidth=bandwidth).rename(columns={"pitch": "predicted_pitch"})
    merged = pd.merge(
        predicted,
        actual,
        on=["bin_x", "bin_y"],
        how="inner",
    )

    corr, rmse = compute_metrics(merged)

    out_csv = os.path.join(output_dir, f"{base_name}_{holdout_game}_prediction.csv")
    out_plot = os.path.join(output_dir, f"{base_name}_{holdout_game}_scatter.png")
    out_metrics = os.path.join(output_dir, f"{base_name}_{holdout_game}_metrics.txt")

    merged.to_csv(out_csv, index=False)
    out_pred_overlay = os.path.join(output_dir, f"{base_name}_{holdout_game}_predicted_overlay.png")
    out_actual_overlay = os.path.join(output_dir, f"{base_name}_{holdout_game}_actual_overlay.png")

    bg_path = os.path.join("lvls_imgs", f"level{base_name}.png")  # adjust path if needed

    save_kde_overlay_on_image(
        predicted,
        value_col="predicted_pitch",
        bins_x=bins_x,
        bins_y=bins_y,
        bg_path=bg_path,
        out_path=out_pred_overlay,
        title=f"Predicted KDE Pitch Overlay\nLevel {base_name}, Holdout {holdout_game}",
        vmin=40,
        vmax=200,
    )

    save_kde_overlay_on_image(
        actual,
        value_col="actual_pitch",
        bins_x=bins_x,
        bins_y=bins_y,
        bg_path=bg_path,
        out_path=out_actual_overlay,
        title=f"Actual Binned Pitch Overlay\nLevel {base_name}, Holdout {holdout_game}",
        vmin=40,
        vmax=200,
    )
    
    merged["pitch_diff"] = merged["predicted_pitch"] - merged["actual_pitch"]

    out_diff_heatmap = os.path.join(output_dir, f"{base_name}_{holdout_game}_difference_heatmap.png")

    plt.figure(figsize=(10, 6))
    diff_map = np.full((bins_y, bins_x), np.nan)

    for _, row in merged.iterrows():
        bx = int(row["bin_x"])
        by = int(row["bin_y"])
        if 0 <= bx < bins_x and 0 <= by < bins_y:
            diff_map[by, bx] = row["pitch_diff"]

    im = plt.imshow(
        diff_map,
        origin="upper",
        aspect="auto",
        cmap="bwr",
        vmin=-40,
        vmax=40,
    )
    plt.colorbar(im, label="Predicted - Actual Pitch (Hz)")
    plt.title(f"Prediction Error Heatmap\nLevel {base_name}, Holdout {holdout_game}")
    plt.xlabel("bin_x")
    plt.ylabel("bin_y")
    plt.tight_layout()
    plt.savefig(out_diff_heatmap, dpi=300, bbox_inches="tight")
    plt.close()

    plot_title = f"Predicted vs Actual Pitch\nLevel {base_name}, Holdout {holdout_game}"
    make_scatterplot(
        merged,
        out_plot,
        plot_title,
        axis_min=axis_min,
        axis_max=axis_max,
    )

    with open(out_metrics, "w", encoding="utf-8") as f:
        f.write(f"input_file: {base_name}\n")
        f.write(f"holdout_game: {holdout_game}\n")
        f.write(f"bins_x: {bins_x}\n")
        f.write(f"bins_y: {bins_y}\n")
        f.write(f"raw_holdout_rows: {len(df_holdout)}\n")
        f.write(f"raw_pred_rows: {len(df_others)}\n")
        f.write(f"holdout_bins: {len(actual)}\n")
        f.write(f"pred_bins: {len(predicted)}\n")
        f.write(f"overlap_bins: {len(merged)}\n")
        f.write(f"correlation: {corr:.6f}\n" if not np.isnan(corr) else "correlation: NaN\n")
        f.write(f"rmse: {rmse:.6f}\n" if not np.isnan(rmse) else "rmse: NaN\n")

    print(f"Done holdout {holdout_game}")
    print(f"  Overlap bins: {len(merged)}")
    print(f"  Correlation: {corr:.4f}" if not np.isnan(corr) else "  Correlation: NaN")
    print(f"  RMSE: {rmse:.4f}" if not np.isnan(rmse) else "  RMSE: NaN")

    return {
        "level": base_name,
        "holdout_game": holdout_game,
        "bins_x": bins_x,
        "bins_y": bins_y,
        "raw_holdout_rows": len(df_holdout),
        "raw_pred_rows": len(df_others),
        "holdout_bins": len(actual),
        "pred_bins": len(predicted),
        "overlap_bins": len(merged),
        "correlation": corr,
        "rmse": rmse,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Leave-one-game-out binned pitch prediction using pooled non-holdout data."
    )
    parser.add_argument("--input", required=True, help="Path to level CSV, e.g. 1.csv")
    parser.add_argument(
        "--holdout",
        default=None,
        help="Optional single game ID to hold out, e.g. game2. If omitted and --all-holdouts is used, runs all.",
    )
    parser.add_argument(
        "--all-holdouts",
        action="store_true",
        help="Run leave-one-out for every game present in the level CSV.",
    )
    parser.add_argument("--bins-x", type=int, default=60, help="Number of bins across x-axis")
    parser.add_argument("--output-dir", default="results", help="Directory to save outputs")
    parser.add_argument("--min-f0", type=float, default=40.0, help="Minimum pitch to keep")
    parser.add_argument("--max-f0", type=float, default=200.0, help="Maximum pitch to keep")
    parser.add_argument("--axis-min", type=float, default=0.0, help="Scatterplot axis minimum")
    parser.add_argument("--axis-max", type=float, default=200.0, help="Scatterplot axis maximum")
    parser.add_argument("--bandwidth", type=float, default=0.05, help="KDE bandwidth in normalized coordinate space")

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required_cols = ["game", "x_center", "y_center", "f0_hz", "video_w", "video_h"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["f0_hz"].between(args.min_f0, args.max_f0)].copy()
    df["game"] = df["game"].astype(str)

    if df.empty:
        raise ValueError("No rows remain after pitch filtering.")

    df, bins_y = compute_bins(df, args.bins_x)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    # decide which holdouts to run
    if args.all_holdouts:
        holdout_games = sorted(df["game"].unique())
    elif args.holdout is not None:
        holdout_games = [args.holdout]
    else:
        raise ValueError("Use either --holdout gameX or --all-holdouts")

    all_results = []

    for holdout_game in holdout_games:
        result = run_holdout(
            df=df,
            holdout_game=holdout_game,
            bins_x=args.bins_x,
            bins_y=bins_y,
            output_dir=args.output_dir,
            base_name=base_name,
            axis_min=args.axis_min,
            axis_max=args.axis_max,
            bandwidth=args.bandwidth,
        )
        if result is not None:
            all_results.append(result)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_csv = os.path.join(args.output_dir, f"{base_name}_all_holdouts_summary.csv")
        summary_df.to_csv(summary_csv, index=False)

        print()
        print("Finished all requested holdouts.")
        print(f"Saved summary: {summary_csv}")

        if len(summary_df) > 0:
            mean_corr = summary_df["correlation"].mean(skipna=True)
            mean_rmse = summary_df["rmse"].mean(skipna=True)
            print(f"Average correlation across holdouts: {mean_corr:.4f}" if not np.isnan(mean_corr) else "Average correlation across holdouts: NaN")
            print(f"Average RMSE across holdouts: {mean_rmse:.4f}" if not np.isnan(mean_rmse) else "Average RMSE across holdouts: NaN")
    else:
        print("No valid holdout runs were completed.")


if __name__ == "__main__":
    main()