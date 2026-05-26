import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt


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


def compute_binned_pitch(df, pitch_col_name, count_col_name):
    """
    Compute average pitch per bin and number of raw points per bin.
    """
    grouped = (
        df.groupby(["bin_x", "bin_y"])
        .agg(
            **{
                pitch_col_name: ("f0_hz", "mean"),
                count_col_name: ("f0_hz", "size"),
            }
        )
        .reset_index()
    )
    return grouped


def compute_metrics(merged):
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


def make_scatterplot(merged, out_path, title, axis_min=0, axis_max=200):
    if merged.empty:
        return

    x = merged["predicted_pitch"]
    y = merged["actual_pitch"]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, alpha=0.75)

    # fixed reference line and fixed axes
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


def run_holdout(df, holdout_game, bins_x, bins_y, output_dir, base_name, axis_min=0, axis_max=200):
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

    actual = compute_binned_pitch(
        df_holdout,
        pitch_col_name="actual_pitch",
        count_col_name="n_actual_points",
    )

    predicted = compute_binned_pitch(
        df_others,
        pitch_col_name="predicted_pitch",
        count_col_name="n_pred_points",
    )

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