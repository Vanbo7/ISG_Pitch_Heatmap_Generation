import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def find_summary(level_dir):
    files = list(Path(level_dir).glob("*_all_holdouts_summary.csv"))
    if not files:
        raise FileNotFoundError(f"No summary CSV found in {level_dir}")
    return files[0]


def load_summary(path, method_name):
    df = pd.read_csv(path)
    df["method"] = method_name
    return df


def plot_average_by_level(binned_root, kde_root, character, max_level, output):
    rows = []

    for level in range(1, max_level + 1):
        binned_dir = Path(binned_root) / character / f"level{level}"
        kde_dir = Path(kde_root) / character / f"level{level}"

        if not binned_dir.exists() or not kde_dir.exists():
            print(f"Skipping level {level}: missing directory")
            continue

        binned_df = load_summary(find_summary(binned_dir), "Binned")
        kde_df = load_summary(find_summary(kde_dir), "KDE")

        binned_corr_mean = binned_df["correlation"].mean()
        kde_corr_mean = kde_df["correlation"].mean()

        binned_rmse_mean = binned_df["rmse"].mean() if "rmse" in binned_df.columns else None
        kde_rmse_mean = kde_df["rmse"].mean() if "rmse" in kde_df.columns else None

        print(f"\nLevel {level}")
        print("Binned dir:", binned_dir)
        print("KDE dir:", kde_dir)
        print("Binned mean correlation:", binned_corr_mean)
        print("KDE mean correlation:", kde_corr_mean)

        if binned_rmse_mean is not None:
            print("Binned mean RMSE:", binned_rmse_mean)
        if kde_rmse_mean is not None:
            print("KDE mean RMSE:", kde_rmse_mean)

        rows.append({
            "level": level,
            "Binned": binned_corr_mean,
            "KDE": kde_corr_mean,
            "Binned_RMSE": binned_rmse_mean,
            "KDE_RMSE": kde_rmse_mean
        })

    result = pd.DataFrame(rows)

    if result.empty:
        print("No levels were found. Nothing to plot.")
        return

    print("\nAverage results per level")
    print("-" * 78)
    print(f"{'Level':<8}{'Binned Corr':>15}{'KDE Corr':>15}{'Binned RMSE':>15}{'KDE RMSE':>15}")
    print("-" * 78)

    for _, row in result.iterrows():
        def fmt(value):
            return "N/A" if pd.isna(value) else f"{value:.4f}"

        print(
            f"{int(row['level']):<8}"
            f"{fmt(row['Binned']):>15}"
            f"{fmt(row['KDE']):>15}"
            f"{fmt(row['Binned_RMSE']):>15}"
            f"{fmt(row['KDE_RMSE']):>15}"
        )

    print("-" * 78)

    print("\nOverall average across levels")
    print("-" * 40)
    print(f"Binned average correlation: {result['Binned'].mean():.4f}")
    print(f"KDE average correlation:    {result['KDE'].mean():.4f}")

    if "Binned_RMSE" in result.columns and result["Binned_RMSE"].notna().any():
        print(f"Binned average RMSE:        {result['Binned_RMSE'].mean():.4f}")
    else:
        print("Binned average RMSE:        N/A")

    if "KDE_RMSE" in result.columns and result["KDE_RMSE"].notna().any():
        print(f"KDE average RMSE:           {result['KDE_RMSE'].mean():.4f}")
    else:
        print("KDE average RMSE:           N/A")

    ax = result.plot(
        x="level",
        y=["Binned", "KDE"],
        kind="bar",
        figsize=(12, 6)
    )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Level")
    plt.ylabel("Average Correlation (r)")
    plt.title(f"Average Correlation by Level: Binned vs KDE ({character})")
    plt.ylim(-1, 1)
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved average chart to {output}")


def plot_holdouts_for_level(binned_root, kde_root, character, level, output):
    binned_dir = Path(binned_root) / character / f"level{level}"
    kde_dir = Path(kde_root) / character / f"level{level}"

    binned_df = load_summary(find_summary(binned_dir), "Binned")
    kde_df = load_summary(find_summary(kde_dir), "KDE")

    df = pd.concat([binned_df, kde_df], ignore_index=True)

    pivot = df.pivot_table(
        index="holdout_game",
        columns="method",
        values="correlation"
    ).reset_index()

    pivot = pivot.sort_values("holdout_game")

    ax = pivot.plot(
        x="holdout_game",
        y=["Binned", "KDE"],
        kind="bar",
        figsize=(14, 6)
    )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Holdout Game")
    plt.ylabel("Correlation (r)")
    plt.title(f"Holdout Correlations for Level {level}: Binned vs KDE ({character})")
    plt.ylim(-1, 1)
    plt.xticks(rotation=45)
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved holdout chart to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare binned vs KDE correlation results."
    )

    parser.add_argument("--binned-root", default="prediction_bins")
    parser.add_argument("--kde-root", default="prediction_kde")
    parser.add_argument("--character", choices=["fireboy", "watergirl"], default="fireboy")
    parser.add_argument("--max-level", type=int, default=21)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--mode", choices=["average", "holdouts"], required=True)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.output is None:
        if args.mode == "average":
            args.output = f"{args.character}_average_correlation_binned_vs_kde.png"
        else:
            args.output = f"{args.character}_level{args.level}_holdout_correlation_binned_vs_kde.png"

    if args.mode == "average":
        plot_average_by_level(
            args.binned_root,
            args.kde_root,
            args.character,
            args.max_level,
            args.output
        )

    elif args.mode == "holdouts":
        if args.level is None:
            raise ValueError("--level is required when mode is holdouts")

        plot_holdouts_for_level(
            args.binned_root,
            args.kde_root,
            args.character,
            args.level,
            args.output
        )


if __name__ == "__main__":
    main()