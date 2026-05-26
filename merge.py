import pandas as pd
import argparse
import sys

def main():
    # === Parse command-line arguments ===
    parser = argparse.ArgumentParser(
        description="Merge YOLO position detections with REAPER F0 pitch data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py merge.py --yolo positions.csv --f0 game1_right.f0 --output merged.csv
  py merge.py -y positions.csv -f game1_right.f0 -o merged.csv --character fireboy
  py merge.py --character fireboy  # Merge fireboy data
  py merge.py -c watergirl  # Merge watergirl data (default)
  py merge.py  # Uses default values (watergirl)
        """
    )
    
    parser.add_argument(
        "--yolo", "-y",
        type=str,
        default="positions.csv",
        help="YOLO detections CSV file (default: positions.csv)"
    )
    
    parser.add_argument(
        "--f0", "-f",
        type=str,
        default="game1_right.f0",
        help="REAPER F0 output file (default: game1_right.f0)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="merged_watergirl_f0.csv",
        help="Output CSV file (default: merged_watergirl_f0.csv)"
    )
    
    parser.add_argument(
        "--character", "-c",
        type=str,
        choices=["watergirl", "fireboy"],
        default="watergirl",
        help="Character to merge (watergirl or fireboy) (default: watergirl)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.60,
        help="Minimum confidence threshold for detections (default: 0.60)"
    )
    
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.025,
        help="Time tolerance for merging in seconds (default: 0.025)"
    )
    
    args = parser.parse_args()
    
    # Assign to variables for easier use
    yolo_csv = args.yolo
    f0_file = args.f0
    output_csv = args.output
    character = args.character
    confidence_threshold = args.confidence
    tolerance = args.tolerance
    
    # Update default output filename if not provided
    if args.output == "merged_watergirl_f0.csv" and character != "watergirl":
        output_csv = f"merged_{character}_f0.csv"
    
    # === Load REAPER F0 file ===
    # Skip header until "EST_Header_End"
    try:
        with open(f0_file) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Error: F0 file not found: {f0_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading F0 file {f0_file}: {e}")
        sys.exit(1)
    
    start_idx = [i for i, line in enumerate(lines) if "EST_Header_End" in line][0] + 1
    
    # Now load the numeric F0 data
    f0 = pd.read_csv(
        f0_file,
        sep=r"\s+",
        names=["time", "voiced", "f0_hz"],
        skiprows=start_idx,
        engine="python"
    )
    
    # Keep only valid voiced frames (pitch > 0)
    f0 = f0[(f0["voiced"] == 1) & (f0["f0_hz"] > 0)].copy()
    
    # Ensure numeric dtypes
    f0["time"] = f0["time"].astype(float)
    f0["f0_hz"] = f0["f0_hz"].astype(float)
    
    # === Load YOLO positions CSV ===
    # Format: frame,class,x_center,y_center,confidence
    try:
        pos = pd.read_csv(yolo_csv)
    except FileNotFoundError:
        print(f"❌ Error: YOLO CSV file not found: {yolo_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading YOLO CSV file {yolo_csv}: {e}")
        sys.exit(1)
    
    # Keep only selected character with confidence >= threshold
    pos = pos[(pos["class"] == character) & (pos["confidence"] >= confidence_threshold)]
    
    # Require time_seconds column (should be present from track_positions.py)
    if "time_seconds" not in pos.columns:
        print("❌ Error: CSV must contain 'time_seconds' column.")
        print("   Please use track_positions.py to generate the CSV file with timestamps.")
        sys.exit(1)
    
    # Ensure time_seconds is float
    pos["time_seconds"] = pos["time_seconds"].astype(float)
    
    # Save counts before merging
    total_yolo = len(pos)
    total_f0 = len(f0)
    
    # === Merge on nearest timestamp ===
    merged = pd.merge_asof(
        pos.sort_values("time_seconds"),
        f0.sort_values("time"),
        left_on="time_seconds",
        right_on="time",
        direction="nearest",
        tolerance=tolerance  # allow configurable time difference
    )
    
    # Drop rows where no F0 match was found
    merged_clean = merged.dropna(subset=["f0_hz"])
    matched = len(merged_clean)
    
    # === Save output ===
    merged_clean.to_csv(output_csv, index=False)
    
    # === Debug summary ===
    print(f"✅ Merged dataset saved to {output_csv}")
    print(f"Character: {character}")
    print(f"Total {character.capitalize()} YOLO detections (conf≥{confidence_threshold}): {total_yolo}")
    print(f"Total valid F0 samples: {total_f0}")
    print(f"Total matched detections with pitch: {matched}")
    print("Preview:")
    print(merged_clean.head(15))

if __name__ == "__main__":
    main()
