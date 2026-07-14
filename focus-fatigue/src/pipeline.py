#!/usr/bin/env python3
"""Single entry point: Model 1 (Pressure Exposure) → All Signals → Merge.

Usage:
    python3 src/pipeline.py --all
    python3 src/pipeline.py --match 2215790
    python3 src/pipeline.py --match 2215790 --nrows 5000
    python3 src/pipeline.py --list-signals
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.loaders import load_tracking_statsperform
from src.smoothing import smooth_trajectory, compute_velocity_features
from src.segments import split_into_blocks, block_summary
from src.pressure.config import DEFAULT_CONFIG
from src.pressure.opponent_proximity import compute_opponent_proximity, aggregate_opponent_proximity_to_blocks
from src.pressure.defensive_depth import compute_defensive_depth, aggregate_defensive_depth_to_blocks
from src.pressure.reorientations import detect_reorientations, aggregate_reorientations_to_blocks
from src.pressure.transitions import detect_transition_frames, count_zone_transitions, aggregate_transitions_to_blocks
from src.pressure.composite import build_pressure_dataset, compute_block_baselines, compute_pressure_composite, classify_pressure_blocks

# Import all signal modules to trigger @register_signal decorators
import src.signals.drift          # noqa: F401 — registers positional_drift
import src.signals.shift          # noqa: F401 — registers shift_latency
import src.signals.pressing       # noqa: F401 — registers pressing_accuracy
import src.signals.transition     # noqa: F401 — registers transition_latency

from src.signals.registry import list_signals, SIGNAL_REGISTRY

OUTPUT_DIR = Path(DEFAULT_CONFIG.output_dir).resolve()


SIGNAL_DESCRIPTIONS = {
    "positional_drift": "Mean Euclidean distance (m) from expected role centroid during out-of-possession.",
    "shift_latency": "Mean reaction time (s) to ball speed spikes and aggressive opponent runs.",
    "pressing_accuracy": "Fraction of pressing actions classified as 'correct' (intercept probability > threshold).",
    "transition_latency": "Mean reaction time (s) to possession transitions (turnovers).",
}


def list_signal_descriptions():
    print("\nRegistered Signals")
    print("=" * 72)
    for name in list_signals():
        desc = SIGNAL_DESCRIPTIONS.get(name, "")
        print(f"  {name:<26} {desc}")
    print()


def find_available_matches(tracking_dir, sample_dir):
    td = Path(tracking_dir)
    matches = sorted(d.name for d in td.iterdir() if d.is_dir() and (d / "tracking.parquet").exists())
    if not matches:
        sd = Path(sample_dir)
        matches = sorted(d.name for d in sd.iterdir() if d.is_dir() and (d / "tracking.parquet").exists())
    return matches


def find_shape_file(match_id, search_dirs):
    for sd in search_dirs:
        base = Path(sd)
        candidates = [base / f"{match_id}.json", base / match_id / f"{match_id}.json"]
        for c in candidates:
            if c.exists(): return c
    return None


def convert_blocks_to_dicts(blocks_dfs):
    """Convert list[pd.DataFrame] blocks to list[dict] (needed by shift_latency)."""
    result = []
    for blk in blocks_dfs:
        bid = str(blk["block_id"].iloc[0])
        phase = int(bid.split("_")[0])
        result.append({"block_id": bid, "phase": phase,
                       "start_frame": int(blk["frame_count"].min()),
                       "end_frame": int(blk["frame_count"].max())})
    return result


def run_model1_on_match(match_id, tracking_path, config, nrows=None):
    """Run Model 1 (Pressure Exposure) on a single match."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Model 1: match {match_id}")
    print(f"{'='*60}")

    df = load_tracking_statsperform(str(tracking_path), match_id=match_id, normalise_dop=True, include_ball=True)
    if nrows: df = df.head(nrows)
    df["frame"] = df["frame_count"]
    print(f"  Loaded {len(df):,} rows")

    df = smooth_trajectory(df, inplace=False)
    df = compute_velocity_features(df)
    print(f"  Smoothed trajectories")

    df = compute_opponent_proximity(df, config=config)
    df = compute_defensive_depth(df, config=config)
    df = detect_reorientations(df, config=config)
    print(f"  Computed per-frame indicators")

    trans_frames = detect_transition_frames(df)
    trans_records = count_zone_transitions(df, trans_frames, config=config)
    print(f"  Detected {len(trans_frames)} transitions")

    blocks = split_into_blocks(df, window_minutes=config.block_window_minutes, min_frames=config.block_min_frames)
    print(f"  Split into {len(blocks)} blocks")

    prox_agg = aggregate_opponent_proximity_to_blocks(blocks, config=config)
    depth_agg = aggregate_defensive_depth_to_blocks(blocks, config=config)
    reo_agg = aggregate_reorientations_to_blocks(blocks, config=config)
    trans_agg = aggregate_transitions_to_blocks(blocks, trans_records)
    print(f"  Aggregated indicators to blocks")

    pressure_dataset = build_pressure_dataset(blocks, prox_agg, depth_agg, reo_agg, trans_agg,
                                              config=config, game_id=match_id)
    if len(pressure_dataset) == 0:
        print("  ⚠️  No pressure data produced!")
        return {"match_id": match_id, "n_players": 0, "blocks": 0, "error": "No pressure data"}

    baselines = compute_block_baselines(pressure_dataset, config=config)
    pressure_df = compute_pressure_composite(pressure_dataset, baselines, config=config)
    result_df = classify_pressure_blocks(pressure_df, config=config)
    result_df["game_id"] = match_id

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_dir / f"pressure_composite_{match_id}.csv", index=False)
    pressure_dataset.to_csv(out_dir / f"pressure_indicators_{match_id}.csv", index=False)

    n_players = result_df["player_id"].nunique()
    high = (result_df["pressure_category"] == "high").sum()
    low = (result_df["pressure_category"] == "low").sum()
    elapsed = round(time.time() - t0, 1)
    print(f"\n  ✅ {match_id}: {n_players} players, {len(result_df)} blocks ({high} high, {low} low) in {elapsed}s")

    return {"match_id": match_id, "n_players": n_players, "n_blocks": len(result_df),
            "high": int(high), "low": int(low), "elapsed_s": elapsed}


def run_signals_on_match(match_id, tracking_path, config, signals_config, source_dir, nrows=None):
    """Run all registered signals on a single match."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Signals: match {match_id}")
    print(f"{'='*60}")

    df = load_tracking_statsperform(str(tracking_path), match_id=match_id, normalise_dop=True, include_ball=True)
    if nrows: df = df.head(nrows)
    df["frame"] = df["frame_count"]
    print(f"  Loaded {len(df):,} rows, {df['frame_count'].nunique():,} frames")

    df = smooth_trajectory(df, inplace=False)
    df = compute_velocity_features(df)
    print(f"  Smoothed trajectories")

    blocks_dfs = split_into_blocks(df, window_minutes=config.block_window_minutes,
                                    min_frames=config.block_min_frames)
    print(f"  {len(blocks_dfs)} blocks")

    if len(blocks_dfs) == 0:
        print("  ⚠️  No valid blocks!")
        return {"match_id": match_id, "signals": {}}

    block_dicts = convert_blocks_to_dicts(blocks_dfs)
    outfield_teams = df[df["player_id"] != -1]["team_id_opta"].unique()
    team_a = int(outfield_teams[0]) if len(outfield_teams) > 0 else 0
    team_b = int(outfield_teams[1]) if len(outfield_teams) > 1 else 0

    results = {}
    for signal_name in list_signals():
        sig_cls = SIGNAL_REGISTRY[signal_name]
        ts = time.time()
        print(f"  Computing {signal_name}...", end=" ", flush=True)

        try:
            signal = sig_cls()
            kwargs = {"match_df": df, "game_id": match_id}

            if signal_name == "shift_latency":
                kwargs["blocks"] = block_dicts
            else:
                kwargs["blocks"] = blocks_dfs

            if signal_name == "positional_drift":
                shape_dirs = [
                    str(Path(source_dir).parent / "shapes"),
                    str(Path(source_dir).parent / "shape_outputs"),
                    str(Path(source_dir).parent / "sample"),
                ]
                shape_path = find_shape_file(match_id, shape_dirs)
                if shape_path:
                    kwargs["shape_path"] = str(shape_path)
                else:
                    print("⚠️ No shape file")
                    results[signal_name] = {"rows": 0, "error": "No shape file"}
                    continue

            if signal_name == "pressing_accuracy":
                kwargs["own_team_id"] = team_a
                kwargs["opponent_team_id"] = team_b

            output_df = signal.compute(**kwargs)
            signal.validate(output_df)
            signal.save(output_df, match_id=match_id)
            n_rows = len(output_df)
            et = time.time() - ts
            print(f"✅ {n_rows} rows in {et:.1f}s")
            results[signal_name] = {"rows": n_rows, "elapsed_s": round(et, 2)}

        except Exception as e:
            et = time.time() - ts
            print(f"❌ Error: {e}")
            results[signal_name] = {"rows": 0, "elapsed_s": round(et, 2), "error": str(e)}

    total_elapsed = round(time.time() - t0, 1)
    print(f"\n  ✅ Match {match_id} signals complete in {total_elapsed}s")
    return {"match_id": match_id, "elapsed_s": total_elapsed, "signals": results}


def run_pipeline(match_ids, tracking_dir, sample_dir, nrows=None, skip_merge=False):
    """Run full pipeline: Model 1 → Signals → Merge for given match IDs."""
    total_start = time.time()
    config = DEFAULT_CONFIG

    model1_results = []
    signals_results = []

    # ── Phase 1: Model 1 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 1: MODEL 1 — PRESSURE EXPOSURE")
    print(f"{'='*60}")

    for match_id in match_ids:
        tracking_path = Path(tracking_dir) / match_id / "tracking.parquet"
        if not tracking_path.exists():
            tracking_path = Path(sample_dir) / match_id / "tracking.parquet"
        if not tracking_path.exists():
            print(f"  ⚠️  {match_id}: tracking.parquet not found")
            continue
        result = run_model1_on_match(match_id, tracking_path, config, nrows=nrows)
        model1_results.append(result)

    # ── Phase 2: Signals ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 2: ALL SIGNALS ({len(list_signals())} total)")
    print(f"{'='*60}")

    for match_id in match_ids:
        tracking_path = Path(tracking_dir) / match_id / "tracking.parquet"
        if not tracking_path.exists():
            tracking_path = Path(sample_dir) / match_id / "tracking.parquet"
        if not tracking_path.exists():
            continue
        result = run_signals_on_match(match_id, tracking_path, config, None, tracking_dir, nrows=nrows)
        signals_results.append(result)

    # ── Phase 3: Merge ──────────────────────────────────────────────
    if not skip_merge:
        print(f"\n{'='*60}")
        print("PHASE 3: MERGING OUTPUTS")
        print(f"{'='*60}")
        try:
            from src.merge_outputs import merge_all
            output_path = "./outputs/unified_fatigue_dataset.parquet"
            merge_all(output_path=output_path)
            print(f"  ✅ Unified dataset saved to: {output_path}")
        except Exception as e:
            print(f"  ⚠️  Merge failed: {e}")

    # ── Summary ─────────────────────────────────────────────────────
    total_elapsed = round(time.time() - total_start, 1)
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {total_elapsed}s")
    print(f"  Matches: {len(match_ids)}")

    p_success = [r for r in model1_results if "error" not in r]
    if p_success:
        print(f"\n  Model 1: {len(p_success)}/{len(model1_results)}")
        print(f"    Players: {sum(r.get('n_players', 0) for r in p_success)}")
        print(f"    High-pressure blocks: {sum(r.get('high', 0) for r in p_success)}")
        print(f"    Low-pressure blocks: {sum(r.get('low', 0) for r in p_success)}")

    if signals_results:
        print(f"\n  Signals:")
        for sn in list_signals():
            total_rows = sum(r["signals"].get(sn, {}).get("rows", 0) for r in signals_results)
            errors = [r["match_id"] for r in signals_results if r["signals"].get(sn, {}).get("error")]
            status = "❌" if errors else "✅"
            err_msg = f" (errors: {errors})" if errors else ""
            print(f"    {status} {sn:<22} {total_rows:>6} rows{err_msg}")

    print(f"\n  Outputs:")
    print(f"    Pressure:  {OUTPUT_DIR}/")
    print(f"    Signals:   outputs/signals/{{signal}}/{{match}}.csv")
    print(f"    Unified:   outputs/unified_fatigue_dataset.parquet")
    print()


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: Model 1 → Signals → Merge")
    parser.add_argument("--match", type=str, default=None, help="Comma-separated match IDs")
    parser.add_argument("--all", action="store_true", help="Process all matches")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows per match (testing)")
    parser.add_argument("--skip-merge", action="store_true", help="Skip final merge step")
    parser.add_argument("--list-signals", action="store_true", help="List signals and exit")
    parser.add_argument("--tracking-dir", type=str, default=None, help="Override tracking directory")
    parser.add_argument("--sample-dir", type=str, default=None, help="Override sample directory")
    args = parser.parse_args()

    if args.list_signals:
        list_signal_descriptions()
        return

    config = DEFAULT_CONFIG
    tracking_dir = args.tracking_dir or config.tracking_dir
    sample_dir = args.sample_dir or config.sample_dir

    if args.all:
        match_ids = find_available_matches(tracking_dir, sample_dir)
        source_dir = tracking_dir
        print(f"Processing ALL matches ({len(match_ids)}) from {tracking_dir}")
    elif args.match:
        match_ids = args.match.split(",")
        source_dir = tracking_dir
        print(f"Processing matches: {match_ids}")
    else:
        match_ids = find_available_matches(sample_dir, "")
        source_dir = sample_dir
        print(f"Processing sample matches ({len(match_ids)}): {match_ids}")

    if not match_ids:
        print("No matches found. Check --tracking-dir or --sample-dir.")
        sys.exit(1)

    run_pipeline(match_ids, source_dir, sample_dir, nrows=args.nrows, skip_merge=args.skip_merge)


if __name__ == "__main__":
    main()
