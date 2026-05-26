import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os
import sys

def main():
    # === Parse command-line arguments ===
    parser = argparse.ArgumentParser(
        description="Generate pitch vs time charts for outlier analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py outlier_charts.py --input merged_watergirl_f0.csv --f0 game1_right.f0 --levels levels.json
  py outlier_charts.py -i merged.csv -f game1_right.f0 -l levels.json --output-dir game1/pitch_charts
  
Levels JSON format:
  [
    {"level": 1, "start": 0, "end": 90},
    {"level": 2, "start": 95.59, "end": 200}
  ]
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="game1/csv/merged_watergirl_f0.csv",
        help="Input CSV file with merged YOLO and F0 data (default: game1/csv/merged_watergirl_f0.csv)"
    )
    
    parser.add_argument(
        "--f0", "-f",
        type=str,
        default="game1/game1_right.f0",
        help="REAPER F0 output file (default: game1/game1_right.f0)"
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
        default="game1/pitch_charts",
        help="Output directory for pitch charts (default: game1/pitch_charts)"
    )
    
    args = parser.parse_args()
    
    # === Load merged dataset ===
    try:
        merged = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"❌ Error: Input CSV file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading CSV file {args.input}: {e}")
        sys.exit(1)
    
    # === Load REAPER F0 file ===
    # Skip header until "EST_Header_End" (same as merge.py)
    try:
        with open(args.f0) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Error: F0 file not found: {args.f0}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading F0 file {args.f0}: {e}")
        sys.exit(1)
    
    # Check if file has header (REAPER format)
    if "EST_Header_End" in "".join(lines):
        start_idx = [i for i, line in enumerate(lines) if "EST_Header_End" in line][0] + 1
        f0 = pd.read_csv(
            args.f0,
            sep=r"\s+",
            names=["time", "voiced", "f0_hz"],
            skiprows=start_idx,
            engine="python"
        )
    else:
        # No header, read directly
        f0 = pd.read_csv(
            args.f0,
            sep=r"\s+",
            names=["time", "voiced", "f0_hz"],
            engine="python"
        )
    
    # Ensure numeric dtypes
    f0["time"] = pd.to_numeric(f0["time"], errors='coerce')
    f0["f0_hz"] = pd.to_numeric(f0["f0_hz"], errors='coerce')
    
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
    
    # Validate levels format
    for level in levels:
        if not all(key in level for key in ["level", "start", "end"]):
            print(f"❌ Error: Invalid level format. Each level must have 'level', 'start', and 'end' keys.")
            print(f"   Found: {level}")
            sys.exit(1)
    
    # === Ensure output directory exists ===
    os.makedirs(args.output_dir, exist_ok=True)
    
    # === Plot pitch vs time for each level ===
    for lvl in levels:
        start, end = lvl["start"], lvl["end"]
        
        # Slice data for this level
        m_slice = merged[(merged["time_seconds"] >= start) & (merged["time_seconds"] <= end)]
        f_slice = f0[(f0["time"] >= start) & (f0["time"] <= end)]
        
        # Plot
        plt.figure(figsize=(14, 5))
        plt.scatter(f_slice["time"], f_slice["f0_hz"], s=3, c="gray", alpha=0.5, label="Original REAPER")
        plt.scatter(m_slice["time_seconds"], m_slice["f0_hz"], s=5, c="red", alpha=0.8, label="Merged CSV")
        
        plt.title(f"Pitch vs Time - Level {lvl['level']}")
        plt.xlabel("Time (s)")
        plt.ylabel("Pitch (Hz)")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Add more ticks every 10 seconds
        xticks = np.arange(start, end + 1, 10)  
        plt.xticks(xticks, [f"{int(x//60)}:{int(x%60):02d}" for x in xticks])  # format as M:SS
        
        output_path = os.path.join(args.output_dir, f"pitch_level{lvl['level']}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved plot for Level {lvl['level']} to {output_path}")

if __name__ == "__main__":
    main()

