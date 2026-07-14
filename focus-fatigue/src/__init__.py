"""Focus Fatigue — Cognitive Fatigue Detection in Football Defence.

This package contains:
- src/loaders.py       — Data loading (tracking, team mappings)
- src/smoothing.py     — Savitzky-Golay trajectory smoothing
- src/segments.py      — Match segmentation into 5-minute blocks
- src/pipeline.py      — Single entry point: Model 1 → Signals → Merge
- src/merge_outputs.py — Merge all outputs into unified parquet
- src/pressure/        — Model 1: Pressure Exposure indicators
- src/signals/         — Model 2: Defensive quality signals
"""
