#!/usr/bin/env python3
"""Entry point for Model 1 — Pressure Exposure computation.

Runs all four load indicators on sample matches, computes pressure
composite, and saves results.

Usage:
    python3 src/run_pressure.py                    # Run on 3 sample matches
    python3 src/run_pressure.py --all              # Run on all 100 matches
    python3 src/run_pressure.py --match 2215790    # Run on one match
    python3 src/run_pressure.py --match 2215790,2215791  # Multiple matches
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loaders.load_tracking import load_tracking_statsperform
from src.loaders.load_shapes import get_match_info
from src.segments import split_into_blocks, block_summary
from src.smoothing import smooth_trajectory, compute_velocity_features
from src.pressure.config import PressureConfig, DEFAULT_CONFIG
from src.pressure.opponent_proximity import (
    compute_opponent_proximity,
    aggregate_opponent_proximity_to_blocks,
)
from src.pressure.defensive_depth import (
    compute_defensive_depth,
    aggregate_defensive_depth_to_blocks,
)
from src.pressure.reorientations import (
    detect_reorientations,
    aggregate_reorientations_to_blocks,
)
from src.pressure.transitions import (
    detect_transition_frames,
    count_zone_transitions,
    aggregate_transitions_to_blocks,
)
from src.pressure.composite import (
    compute_block_baselines,
    compute_pressure_composite,
    classify_pressure_blocks,
    build_pressure_dataset,
)


def process_one_match(
    match_id: str,
    tracking_path: Path,
    config: PressureConfig = DEFAULT_CONFIG,
) -> dict:
    """Process a single match through the entire Model 1 pipeline.

    Parameters
    ----------
    match_id : str
    tracking_path : Path
        Path to tracking.parquet
    config : PressureConfig

    Returns
    -------
    dict with keys:
        - match_id
        - n_frames: total tracking frames
        - n_blocks: number of 5-minute blocks
        - n_players: players processed
        - indicators_path: path to saved per-block indicators
        - pressure_path: path to saved pressure composites
        - high_pressure_blocks: count
        - low_pressure_blocks: count
        - elapsed_s: processing time in seconds
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Processing match {match_id}...")
    print(f"{'='*60}")

    # Step 1: Load tracking data
    print("  [1/8] Loading tracking data...")
    df = load_tracking_statsperform(
        str(tracking_path),
        match_id=match_id,
        normalise_dop=True,
        include_ball=True,
    )
    print(f"         {len(df):,} rows, {df['frame_count'].nunique():,} frames")

    # Step 2: Smooth trajectories (needed for velocity-based indicators)
    print("  [2/8] Smoothing trajectories...")
    df = smooth_trajectory(df, inplace=False)
    df = compute_velocity_features(df)

    # Step 3: Segment into blocks
    print("  [3/8] Segmenting into blocks...")
    blocks = split_into_blocks(
        df,
        window_minutes=config.block_window_minutes,
        min_frames=config.block_min_frames,
    )
    summary = block_summary(blocks)
    print(f"         {len(blocks)} blocks, {len(summary)} valid")

    # Step 4: Compute opponent proximity
    print("  [4/8] Computing opponent proximity...")
    df = compute_opponent_proximity(df, radius=config.opponent_radius, config=config)
    proximity_agg = aggregate_opponent_proximity_to_blocks(blocks, config=config)

    # Step 5: Compute defensive depth
    print("  [5/8] Computing defensive depth...")
    df = compute_defensive_depth(df, goal_line_x=config.goal_line_x, config=config)
    depth_agg = aggregate_defensive_depth_to_blocks(blocks, config=config)

    # Step 6: Detect reorientations
    print("  [6/8] Detecting reorientations...")
    df = detect_reorientations(df, config=config)
    reo_agg = aggregate_reorientations_to_blocks(blocks, config=config)

    # Step 7: Detect transitions
    print("  [7/8] Detecting possession transitions...")
    trans_frames = detect_transition_frames(df)
    print(f"         {len(trans_frames)} transitions detected")
    trans_records = count_zone_transitions(
        df, trans_frames, radius=config.transition_zone_radius, config=config
    )
    trans_agg = aggregate_transitions_to_blocks(blocks, trans_records)

    # Step 8: Build pressure composite
    print("  [8/8] Building pressure composite...")
    pressure_dataset = build_pressure_dataset(
        blocks, proximity_agg, depth_agg, reo_agg, trans_agg,
        config=config,
    )

    if len(pressure_dataset) == 0:
        print("  ⚠️  No pressure data produced!")
        return {
            "match_id": match_id,
            "n_frames": len(df),
            "n_blocks": len(blocks),
            "n_players": 0,
            "elapsed_s": round(time.time() - t0, 1),
            "error": "No pressure data",
        }

    # Compute baselines
    baselines = compute_block_baselines(pressure_dataset, config=config)

    # Compute pressure composite
    pressure_df = compute_pressure_composite(pressure_dataset, baselines, config=config)

    # Classify blocks
    result_df = classify_pressure_blocks(pressure_df, config=config)

    # Save outputs
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indicators_path = output_dir / f"pressure_indicators_{match_id}.csv"
    pressure_path = output_dir / f"pressure_composite_{match_id}.csv"

    pressure_dataset.to_csv(indicators_path, index=False)
    result_df.to_csv(pressure_path, index=False)

    # Stats
    n_players = result_df["player_id"].nunique()
    high_count = (result_df["pressure_category"] == "high").sum()
    low_count = (result_df["pressure_category"] == "low").sum()

    elapsed = round(time.time() - t0, 1)
    print(f"\n  ✅ Match {match_id} complete in {elapsed}s")
    print(f"     Players: {n_players} | Blocks: {len(result_df)}")
    print(f"     High pressure: {high_count} | Low pressure: {low_count}")
    print(f"     Indicators: {indicators_path}")
    print(f"     Composite:  {pressure_path}")

    return {
        "match_id": match_id,
        "n_frames": len(df),
        "n_blocks": len(blocks),
        "n_players": n_players,
        "high_pressure_blocks": int(high_count),
        "low_pressure_blocks": int(low_count),
        "indicators_path": str(indicators_path),
        "pressure_path": str(pressure_path),
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Model 1 — Pressure Exposure computation"
    )
    parser.add_argument(
        "--match", type=str, default=None,
        help="Comma-separated match IDs (default: 3 sample matches)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all matches in tracking directory"
    )
    parser.add_argument(
        "--sample-dir", type=str,
        default="/home/conormalone/conor_downloads/team_mappings/sample",
        help="Sample data directory"
    )
    parser.add_argument(
        "--tracking-dir", type=str,
        default="/home/conormalone/conor_downloads/team_mappings/tracking",
        help="Full tracking data directory"
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG

    # Determine matches to process
    if args.all:
        tracking_dir = Path(args.tracking_dir)
        match_ids = sorted([
            d.name for d in tracking_dir.iterdir()
            if d.is_dir() and (d / "tracking.parquet").exists()
        ])
        source_dir = tracking_dir
        print(f"Processing ALL matches ({len(match_ids)}) from {tracking_dir}")
    elif args.match:
        match_ids = args.match.split(",")
        tracking_dir = Path(args.tracking_dir)
        source_dir = tracking_dir
        print(f"Processing specified matches: {match_ids}")
    else:
        # Default: process 3 sample matches
        sample_dir = Path(args.sample_dir)
        match_ids = sorted([
            d.name for d in sample_dir.iterdir()
            if d.is_dir() and (d / "tracking.parquet").exists()
        ])
        source_dir = sample_dir
        print(f"Processing sample matches ({len(match_ids)}): {match_ids}")

    # Process each match sequentially
    results = []
    total_start = time.time()

    for match_id in match_ids:
        tracking_path = source_dir / match_id / "tracking.parquet"
        if not tracking_path.exists():
            print(f"  ⚠️  {match_id}: tracking.parquet not found at {tracking_path}")
            continue

        result = process_one_match(match_id, tracking_path, config=config)
        results.append(result)

    # Summary
    total_elapsed = round(time.time() - total_start, 1)
    success = [r for r in results if "error" not in r]

    print(f"\n{'='*60}")
    print(f"MODEL 1 — PRESSURE EXPOSURE: COMPLETE")
    print(f"{'='*60}")
    print(f"  Matches processed: {len(success)}/{len(results)}")
    print(f"  Total time: {total_elapsed}s")
    if success:
        avg_time = np.mean([r["elapsed_s"] for r in success])
        print(f"  Avg per match: {avg_time:.1f}s")
        total_players = sum(r["n_players"] for r in success)
        total_high = sum(r.get("high_pressure_blocks", 0) for r in success)
        total_low = sum(r.get("low_pressure_blocks", 0) for r in success)
        print(f"  Total players: {total_players}")
        print(f"  Total high-pressure blocks: {total_high}")
        print(f"  Total low-pressure blocks: {total_low}")
    print(f"  Output: {config.output_dir}/")


if __name__ == "__main__":
    main()
