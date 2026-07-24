#!/usr/bin/env python3
"""Profile the full pipeline step-by-step on a real match on the Pi.

Times each of the 7 steps:
  1. load_tracking_statsperform
  2. smooth_trajectory
  3. compute_velocity_features
  4. split_into_blocks
  5. polarisation signal compute
  6. centroid_distance signal compute
  7. save CSVs

Usage: python3 scripts/profile_pipeline.py [--match 2215790]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders import load_tracking_statsperform
from src.smoothing import smooth_trajectory, compute_velocity_features
from src.segments import split_into_blocks
import src.signals.polarisation
import src.signals.team_centroid_distance

DATA_ROOT = Path("/mnt/usb/conor_downloads/team_mappings")
TRACKING_DIR = DATA_ROOT / "tracking"
SAMPLE_DIR = DATA_ROOT / "sample"
OUTPUT_DIR = Path("./outputs/profile_pipeline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


def profile_match(match_id: str):
    tracking_path = TRACKING_DIR / match_id / "tracking.parquet"
    if not tracking_path.exists():
        tracking_path = SAMPLE_DIR / match_id / "tracking.parquet"
    if not tracking_path.exists():
        print(f"❌ Match {match_id}: tracking.parquet not found")
        return

    print(f"\n{'='*70}")
    print(f"PROFILING FULL PIPELINE — match {match_id}")
    print(f"{'='*70}")

    timings = {}

    # Step 1: LOAD
    print(f"\n{'─'*60}\nSTEP 1: load_tracking_statsperform\n{'─'*60}")
    t0 = time.time()
    df = load_tracking_statsperform(str(tracking_path), match_id=match_id,
                                     normalise_dop=True, include_ball=True)
    df["frame"] = df["frame_count"]
    load_time = time.time() - t0
    timings["1_load"] = load_time
    n_rows, n_frames, n_players = len(df), df["frame_count"].nunique(), df["player_id"].nunique()
    print(f"  Loaded: {n_rows:,} rows, {n_frames:,} frames, {n_players} players")
    print(f"  Time:   {format_time(load_time)}")

    # Step 2: SMOOTH
    print(f"\n{'─'*60}\nSTEP 2: smooth_trajectory\n{'─'*60}")
    t0 = time.time()
    df = smooth_trajectory(df, inplace=False)
    smooth_time = time.time() - t0
    timings["2_smooth"] = smooth_time
    print(f"  Time:   {format_time(smooth_time)}")

    # Step 3: VELOCITY FEATURES
    print(f"\n{'─'*60}\nSTEP 3: compute_velocity_features\n{'─'*60}")
    t0 = time.time()
    df = compute_velocity_features(df)
    vel_time = time.time() - t0
    timings["3_velocity_features"] = vel_time
    print(f"  Time:   {format_time(vel_time)}")

    # Step 4: SPLIT BLOCKS
    print(f"\n{'─'*60}\nSTEP 4: split_into_blocks\n{'─'*60}")
    t0 = time.time()
    blocks_dfs = split_into_blocks(df, window_minutes=5, min_frames=100)
    block_time = time.time() - t0
    timings["4_split_blocks"] = block_time
    print(f"  Blocks: {len(blocks_dfs)}")
    print(f"  Time:   {format_time(block_time)}")

    # Step 5: POLARISATION
    print(f"\n{'─'*60}\nSTEP 5: polarisation signal\n{'─'*60}")
    t0 = time.time()
    pol_signal = src.signals.polarisation.PolarisationSignal()
    pol_df = pol_signal.compute(match_df=df, blocks=blocks_dfs, game_id=match_id)
    pol_time = time.time() - t0
    timings["5_polarisation"] = pol_time
    print(f"  Output: {len(pol_df)} rows")
    print(f"  Time:   {format_time(pol_time)}")

    # Step 6: CENTROID DISTANCE
    print(f"\n{'─'*60}\nSTEP 6: centroid_distance signal\n{'─'*60}")
    t0 = time.time()
    cd_signal = src.signals.team_centroid_distance.TeamCentroidDistanceSignal()
    cd_df = cd_signal.compute(match_df=df, blocks=blocks_dfs, game_id=match_id)
    cd_time = time.time() - t0
    timings["6_centroid_distance"] = cd_time
    print(f"  Output: {len(cd_df)} rows")
    print(f"  Time:   {format_time(cd_time)}")

    # Step 7: SAVE CSVs
    print(f"\n{'─'*60}\nSTEP 7: Save CSVs\n{'─'*60}")
    t0 = time.time()
    pol_out = OUTPUT_DIR / f"team_polarisation_{match_id}.csv"
    cd_out = OUTPUT_DIR / f"team_centroid_distance_{match_id}.csv"
    pol_df.to_csv(pol_out, index=False)
    cd_df.to_csv(cd_out, index=False)
    save_time = time.time() - t0
    timings["7_save_csv"] = save_time
    print(f"  Polarisation CSV:       {pol_out} ({pol_out.stat().st_size/1024:.0f} KB)")
    print(f"  Centroid distance CSV:  {cd_out} ({cd_out.stat().st_size/1024:.0f} KB)")
    print(f"  Time:   {format_time(save_time)}")

    # SUMMARY
    total_time = sum(timings.values())
    print(f"\n{'='*70}")
    print("TIMING BREAKDOWN")
    print(f"{'='*70}")
    step_labels = {
        "1_load": "1. Load",
        "2_smooth": "2. Smooth trajectory",
        "3_velocity_features": "3. Velocity features",
        "4_split_blocks": "4. Split blocks",
        "5_polarisation": "5. Polarisation signal",
        "6_centroid_distance": "6. Centroid distance signal",
        "7_save_csv": "7. Save CSVs",
    }
    
    for key in ["1_load", "2_smooth", "3_velocity_features", "4_split_blocks",
                 "5_polarisation", "6_centroid_distance", "7_save_csv"]:
        t = timings.get(key, 0)
        pct = (t / total_time * 100) if total_time > 0 else 0
        label = step_labels.get(key, key)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {label:<25s} {format_time(t):>10s}  {bar} {pct:5.1f}%")

    print(f"  {'─'*56}")
    print(f"  {'TOTAL':<25s} {format_time(total_time):>10s}  {'█' * 50} 100.0%")
    print()

    sorted_steps = sorted(timings.items(), key=lambda x: x[1], reverse=True)
    bottleneck_step = step_labels.get(sorted_steps[0][0], sorted_steps[0][0])
    print(f"  Bottleneck: {bottleneck_step} ({format_time(sorted_steps[0][1])})")
    print(f"  Data: {(n_rows/1e6):.1f}M rows, {n_frames:,} frames, {n_players} players, {len(blocks_dfs)} blocks")

    return timings


def main():
    parser = argparse.ArgumentParser(description="Profile full pipeline on Pi")
    parser.add_argument("--match", type=str, default="2215790")
    args = parser.parse_args()
    profile_match(args.match)


if __name__ == "__main__":
    main()
