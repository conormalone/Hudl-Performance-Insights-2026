#!/usr/bin/env python3
"""Run all registered signals on match tracking data.

Usage:
    python3 src/run_signals.py --all
    python3 src/run_signals.py --match 2215790
    python3 src/run_signals.py --signal transition_latency --all
    python3 src/run_signals.py --list
    python3 src/run_signals.py --match 2215790 --nrows 5000
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.signals.registry import SIGNAL_REGISTRY, list_signals
from src.loaders.load_tracking import load_tracking_statsperform
from src.smoothing import smooth_trajectory, compute_velocity_features
from src.segments import split_into_blocks
from src.pressure.config import DEFAULT_CONFIG

# Import all signal modules to trigger @register_signal decorators
import src.signals.positional_drift  # noqa: F401 — registers positional_drift
import src.signals.shift_latency    # noqa: F401 — registers shift_latency
import src.signals.pressing          # noqa: F401 — registers pressing_accuracy
import src.signals.transition        # noqa: F401 — registers transition_latency

# Signal-specific imports for special args
from src.signals.config import DEFAULT_SIGNAL_CONFIG

logger = logging.getLogger("run_signals")


# ── Helpers ──────────────────────────────────────────────────────────────────


SIGNAL_DESCRIPTIONS: dict[str, str] = {
    "positional_drift": (
        "Mean Euclidean distance (m) from a defender's actual position "
        "to their shape-model expected role centroid during out-of-possession phases."
    ),
    "shift_latency": (
        "Mean reaction time (s) for defenders to respond to sudden triggers "
        "(ball speed spikes, aggressive opponent runs)."
    ),
    "pressing_accuracy": (
        "Fraction of pressing actions classified as 'correct' (intercept "
        "probability > threshold) per block per defender."
    ),
    "transition_latency": (
        "Mean reaction time (s) for defenders to recognise and react to "
        "possession transitions (turnovers)."
    ),
}


def print_signal_list():
    """Print available signals with descriptions."""
    print("\nRegistered Signals")
    print("=" * 72)
    print(f"{'Signal Name':<28} {'Description'}")
    print("-" * 72)
    for name in list_signals():
        desc = SIGNAL_DESCRIPTIONS.get(name, "No description available.")
        print(f"  {name:<26} {desc}")
    print("=" * 72)


def _blocks_to_dicts(blocks_df: list[pd.DataFrame]) -> list[dict[str, Any]]:
    """Convert DataFrame-based blocks (from ``split_into_blocks``) to dicts.

    The ``shift_latency`` signal's ``aggregate_shift_latency_by_block``
    expects ``list[dict]`` with keys: block_id, phase, start_frame, end_frame.
    Other signals accept ``list[pd.DataFrame]`` natively.
    """
    result: list[dict[str, Any]] = []
    for blk in blocks_df:
        bid = str(blk["block_id"].iloc[0])
        phase = int(bid.split("_")[0])
        result.append({
            "block_id": bid,
            "phase": phase,
            "start_frame": int(blk["frame_count"].min()),
            "end_frame": int(blk["frame_count"].max()),
        })
    return result


def get_available_match_ids(tracking_dir: str) -> list[str]:
    """Get sorted list of match IDs with tracking.parquet."""
    td = Path(tracking_dir)
    if not td.exists():
        return []
    return sorted(
        d.name for d in td.iterdir()
        if d.is_dir() and (d / "tracking.parquet").exists()
    )


# ── Match Processing ─────────────────────────────────────────────────────────


def process_one_match(
    match_id: str,
    tracking_path: Path,
    nrows: int | None = None,
) -> dict:
    """Load, smooth, segment, and compute all registered signals for one match.

    Parameters
    ----------
    match_id : str
    tracking_path : Path
        Path to ``tracking.parquet``.
    nrows : int or None
        If set, only load this many rows (for testing).

    Returns
    -------
    dict
        Summary of results per signal.
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Signals: Processing match {match_id}...")
    print(f"{'='*60}")

    # Step 1: Load tracking data
    print("  [1] Loading tracking data...")
    df = load_tracking_statsperform(
        str(tracking_path),
        match_id=match_id,
        normalise_dop=True,
        include_ball=True,
    )

    if nrows is not None:
        df = df.head(nrows)
        print(f"      (Trimmed to {nrows} rows for testing)")

    print(f"      {len(df):,} rows, {df['frame_count'].nunique():,} frames")

    # Step 2: Smooth trajectories
    print("  [2] Smoothing trajectories...")
    df = smooth_trajectory(df, inplace=False)
    df = compute_velocity_features(df)

    # Step 3: Segment into blocks
    print("  [3] Segmenting into blocks...")
    blocks_dfs = split_into_blocks(
        df,
        window_minutes=DEFAULT_CONFIG.block_window_minutes,
        min_frames=DEFAULT_CONFIG.block_min_frames,
    )
    print(f"      {len(blocks_dfs)} blocks")

    if len(blocks_dfs) == 0:
        print("  ⚠️  No valid blocks found!")
        return {"match_id": match_id, "elapsed_s": round(time.time() - t0, 1), "signals": {}}

    # Pre-convert block dicts once (needed by shift_latency)
    block_dicts = _blocks_to_dicts(blocks_dfs)

    # Step 4: Run each registered signal
    results = {}
    for signal_name in list_signals():
        signal_cls = SIGNAL_REGISTRY[signal_name]
        t_sig = time.time()

        print(f"  [4] Computing signal: {signal_name}...")

        try:
            # Instantiate the signal class
            signal = signal_cls()

            # Prepare compute kwargs
            # run_pipeline() calls compute(match_df, blocks) without passing
            # game_id, so we call compute() and save() separately.
            compute_kwargs: dict[str, Any] = {
                "match_df": df,
                "game_id": match_id,
            }

            # Some signals need blocks as dicts, others as DataFrames
            if signal_name == "shift_latency":
                compute_kwargs["blocks"] = block_dicts
            else:
                compute_kwargs["blocks"] = blocks_dfs

            # Compute signal
            output_df = signal.compute(**compute_kwargs)

            # Validate
            signal.validate(output_df)

            # Save to outputs/signals/{signal_name}/{match_id}.csv
            signal.save(output_df, match_id=match_id)

            n_rows = len(output_df)
            elapsed = round(time.time() - t_sig, 2)
            results[signal_name] = {"rows": n_rows, "elapsed_s": elapsed}
            print(f"      ✅ {n_rows} rows in {elapsed}s")

        except Exception as exc:
            elapsed = round(time.time() - t_sig, 2)
            results[signal_name] = {"rows": 0, "elapsed_s": elapsed, "error": str(exc)}
            print(f"      ❌ Error in {elapsed}s: {exc}")

    total_elapsed = round(time.time() - t0, 1)
    print(f"\n  ✅ Match {match_id} signals complete in {total_elapsed}s")
    return {
        "match_id": match_id,
        "elapsed_s": total_elapsed,
        "signals": results,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run all registered signals on match tracking data."
    )
    parser.add_argument(
        "--match", type=str, default=None,
        help="Comma-separated match IDs to process."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all matches in tracking directory."
    )
    parser.add_argument(
        "--signal", type=str, default=None,
        help="Run only this signal (e.g. 'transition_latency')."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available signals with descriptions and exit."
    )
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="Only load this many rows per match (for testing)."
    )
    parser.add_argument(
        "--tracking-dir", type=str, default=None,
        help="Override tracking data directory."
    )
    parser.add_argument(
        "--sample-dir", type=str, default=None,
        help="Override sample data directory (used when no --match or --all)."
    )
    args = parser.parse_args()

    # Handle --list
    if args.list:
        print_signal_list()
        return

    # Validate --signal argument
    if args.signal:
        if args.signal not in SIGNAL_REGISTRY:
            print(f"Error: Unknown signal '{args.signal}'.")
            print(f"Available signals: {', '.join(list_signals())}")
            sys.exit(1)
        enabled_signals = [args.signal]
        print(f"Running only signal: {args.signal}")
    else:
        enabled_signals = list_signals()
        print(f"Running all {len(enabled_signals)} registered signals")

    # Resolve tracking directory
    tracking_dir = args.tracking_dir or DEFAULT_CONFIG.tracking_dir

    # Determine which matches to process
    if args.all:
        match_ids = get_available_match_ids(tracking_dir)
        source_dir = tracking_dir
        print(f"Processing ALL matches ({len(match_ids)}) from {tracking_dir}")
    elif args.match:
        match_ids = args.match.split(",")
        source_dir = tracking_dir
        print(f"Processing specified matches: {match_ids}")
    else:
        sample_dir = args.sample_dir or DEFAULT_CONFIG.sample_dir
        match_ids = get_available_match_ids(sample_dir)
        source_dir = sample_dir
        print(f"Processing sample matches ({len(match_ids)}): {match_ids}")

    if not match_ids:
        print("No matches found. Check --tracking-dir or --sample-dir paths.")
        sys.exit(1)

    # Process each match sequentially (Pi-friendly)
    all_results = []
    total_start = time.time()

    for match_id in match_ids:
        tracking_path = Path(source_dir) / match_id / "tracking.parquet"
        if not tracking_path.exists():
            print(f"  ⚠️  {match_id}: tracking.parquet not found at {tracking_path}")
            continue

        result = process_one_match(match_id, tracking_path, nrows=args.nrows)
        all_results.append(result)

    # Summary
    total_elapsed = round(time.time() - total_start, 1)
    print(f"\n{'='*60}")
    print(f"SIGNALS: ALL MATCHES COMPLETE")
    print(f"{'='*60}")
    print(f"  Matches processed: {len(all_results)}")
    print(f"  Total time: {total_elapsed}s")
    print(f"  Output root: {DEFAULT_SIGNAL_CONFIG.output_root}/")

    # Per-signal summary
    print(f"\n  Per-signal totals:")
    for signal_name in enabled_signals:
        total_rows = sum(
            r["signals"].get(signal_name, {}).get("rows", 0)
            for r in all_results
        )
        total_sig_time = sum(
            r["signals"].get(signal_name, {}).get("elapsed_s", 0)
            for r in all_results
        )
        errors = [
            r["match_id"]
            for r in all_results
            if r["signals"].get(signal_name, {}).get("error")
        ]
        status = "❌" if errors else "✅"
        err_msg = f" (errors: {errors})" if errors else ""
        print(f"    {status} {signal_name:<22} {total_rows:>6} rows  {total_sig_time:>7.1f}s{err_msg}")

    print()


if __name__ == "__main__":
    main()
