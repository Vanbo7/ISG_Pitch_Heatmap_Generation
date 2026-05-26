# ISG_Pitch_Heatmap_Generation
A pipeline for extracting, aligning, and visualizing pitch (F0) data with player position in the cooperative video game Fireboy and Watergirl. Uses YOLOv8 for character tracking and REAPER for pitch extraction, combining both into spatial heatmaps that map vocal behavior onto game levels.
Includes grid-binned and Gaussian KDE-based heatmap generation, aggregation across gameplay sessions, and a leave-one-out evaluation framework to assess whether spatial position is a consistent predictor of prosodic behavior.
Features

YOLO-based character detection and coordinate extraction
Pitch extraction and alignment via REAPER
Grid-binned and KDE pitch heatmaps overlaid on game level backgrounds
Cross-session aggregation
Leave-one-out prediction evaluation with correlation and RMSE metrics

