# ISG Pitch Heatmap Generation

A pipeline for extracting, aligning, and visualizing pitch (F0) data with player position in the cooperative video game **Fireboy and Watergirl**. Uses YOLOv8 for character tracking and REAPER for pitch extraction, combining both into spatial heatmaps that map vocal behavior onto game levels.

---

## Pipeline Overview

```
track_positions.py   →   merge.py   →   map.py / map_gaussian.py
                                     →   aggregate_levels.py
                                     →   predict.py / predict_kde.py
```

---

## Scripts

### `track_positions.py`
Runs YOLOv8 object detection on a gameplay video to extract the x and y coordinates of Watergirl and Fireboy across every frame. Outputs a CSV with bounding box centers, confidence scores, frame numbers, and timestamps.

**Requires:** a trained YOLOv8 model (`.pt` file) and a gameplay video.

---

### `merge.py`
Merges the YOLO position data with pitch (F0) data extracted by REAPER. Aligns both sources by timestamp and outputs a single CSV containing position, frame number, and pitch (Hz) for each time point. This merged CSV is the input for all downstream scripts.

---

### `extract_frames.py`
Extracts background images from a video at specified timestamps. These images are used as the base layer for heatmap overlays. Can also be called directly from `map.py` and `map_gaussian.py` using the `--extract-frames` flag.

---

### `map.py`
The main heatmap generation script. Divides each game level into a spatial grid and computes the average pitch within each cell, then overlays the result onto the level background image. Produces one heatmap per level per gameplay session.

```bash
python map.py --input games\game2\merged_watergirl.csv \
              --levels games\game2\timestamps.json \
              --video games\game2\game_2.mp4 \
              --extract-frames \
              --output-dir games\game2\heatmaps
```

Key arguments:
- `--input` — merged CSV file
- `--levels` — JSON file defining level time ranges
- `--video` — gameplay video (used with `--extract-frames`)
- `--extract-frames` — pull background images from the video automatically
- `--grid-size` — pixel size of each spatial bin (default: 30)
- `--bw` — save background images in grayscale

---

### `map_gaussian.py`
An alternative heatmap script that uses Gaussian KDE instead of grid binning. Rather than averaging pitch within fixed cells, it computes a kernel-weighted average pitch at every point on a continuous grid, producing smoother heatmaps. Takes the same inputs as `map.py`.

```bash
python map_gaussian.py --input games\game2\merged_watergirl.csv \
                       --levels games\game2\timestamps.json \
                       --video games\game2\game_2.mp4 \
                       --extract-frames \
                       --output-dir games\game2\heatmaps\kde
```

Additional arguments:
- `--bandwidth` — controls smoothing in normalized [0,1] coordinate space (default: 0.05; try 0.01 for more detail)
- `--bins-x` — number of grid columns (default: 60)

---

### `aggregate_levels.py`
Combines merged CSVs from multiple gameplay sessions into a single aggregated dataset per level. Since level layouts are consistent across sessions, this allows data from different games to be mapped onto the same background, producing denser and more reliable heatmaps.

---

### `predict.py`
Leave-one-out evaluation using grid-based binning. For each gameplay session, holds it out as ground truth and uses the remaining sessions to build a predicted pitch map. Computes Pearson correlation and RMSE between predicted and actual spatial pitch distributions.

---

### `predict_kde.py`
Same leave-one-out evaluation as `predict.py`, but the prediction is generated using Gaussian KDE instead of strict binning. The KDE smooths the aggregated pitch surface so that small positional differences between sessions affect the comparison less. Saves predicted and actual overlay images, a difference heatmap, a scatter plot, and a metrics file for each holdout.

```bash
python predict_kde.py --input 1.csv --all-holdouts --bins-x 60 --bandwidth 0.05
```

---

### `plot_correlation_comparison.py`
Takes the summary CSVs output by `predict.py` and `predict_kde.py` and generates bar charts comparing average Pearson correlation per level for both the binned and KDE methods. Used to produce the results figures in the paper.

---

### `outlier_charts.py`
Generates diagnostic charts showing the distribution of pitch values across the dataset. Useful for identifying and visualizing outliers (e.g., breathing artifacts, creaky voice) before deciding on pitch filtering thresholds.

---

### `test_video_read.py`
A small utility script for verifying that a video file can be opened and read correctly by OpenCV. Useful for debugging video path or codec issues before running the main pipeline.

---

## Input Format

### Merged CSV
The main input file expected by `map.py`, `map_gaussian.py`, `predict.py`, and `predict_kde.py`. Must contain:

| Column | Description |
|--------|-------------|
| `x_center` | X coordinate of character bounding box center |
| `y_center` | Y coordinate of character bounding box center |
| `f0_hz` | Pitch value in Hz |
| `time_seconds` or `frame` | Timestamp or frame number |
| `video_w` | Video width in pixels (optional if `--video` is provided) |
| `video_h` | Video height in pixels (optional if `--video` is provided) |

### Levels JSON
Defines the time ranges for each level. Example:
```json
[
  {"level_id": 1, "start": 0, "end": 90},
  {"level_id": 2, "start": 95.5, "end": 200}
]
```

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
opencv-python
ultralytics
scipy
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn opencv-python ultralytics scipy
```

Pitch extraction requires [REAPER](https://github.com/google/REAPER) (separate tool, not a Python package).

---

## Project Structure

```
games/
  game1/
    merged_watergirl.csv
    timestamps.json
    game_1.mp4
    lvls_imgs/
    heatmaps/
  game2/
    ...
```

---

## Related Work

Based on the research described in:

- Nigel G. Ward. *Action Prosody*. University of Texas at El Paso. https://www.cs.utep.edu/nigel/abstracts/action-prosody.html
- Nigel G. Ward, Fernando Alvarado, Harm Lameris. *How Much Value can Appropriate Prosody have for a Collaborative Agent?* Under submission to SIGDIAL.
