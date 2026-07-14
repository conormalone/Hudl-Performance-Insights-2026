#!/usr/bin/env python3
"""Complete pipeline: Model 1 (pressure exposure) → all signals → merge.

Runs Model 1 first on the same blocks, then all registered signals,
then merges everything into a single unified dataset.

Usage:
    python3 src/run_pipeline.py --all
    python3 src/run_pipeline.py --match 2215790
    python3 src/run_pipeline.py --all --nrows 5000
    python3 src/run_pipeline.py --match 2215790 --nrows 5000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.run_pressure import process_one_match as process_pressure
# Import signal modules to trigger @register_signal decorators
import src.signals.positional_drift  # noqa: F401
import src.signals.shift_latency    # noqa: F401
import src.signals.pressing          # noqa: F401
import src.signals.transition        # noqa: F401

from src.run_signals import (
    process_one_match as process_signals,
    list_signals,
    print_signal_list,
    get_available_match_ids,
)
from src.pressure.config import DEFAULT_CONFIG

OUTPUT_DIR = Path(DEFAULT_CONFIG.output_dir).resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline: Model 1 → signals → merge."
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
    parser.add_argument(
        "--list", action="store_true",
        help="List available signals with descriptions and exit."
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip the final merge step."
    )
    args = parser.parse_args()

    if args.list:
        print_signal_list()
        return

    # Resolve tracking directory
    tracking_dir = args.tracking_dir or DEFAULT_CONFIG.tracking_dir

    # Determine which matches to process
    if args.all:
        match_ids = get_available_match_ids(tracking_dir)
        source_dir = tracking_dir
        print(f"Pipeline: processing ALL matches ({len(match_ids)}) from {tracking_dir}")
    elif args.match:
        match_ids = args.match.split(",")
        source_dir = tracking_dir
        print(f"Pipeline: processing specified matches: {match_ids}")
    else:
        sample_dir = args.sample_dir or DEFAULT_CONFIG.sample_dir
        match_ids = get_available_match_ids(sample_dir)
        source_dir = sample_dir
        print(f"Pipeline: processing sample matches ({len(match_ids)}): {match_ids}")

    if not match_ids:
        print("No matches found. Check --tracking-dir or --sample-dir paths.")
        sys.exit(1)

    total_start = time.time()
    pressure_results = []
    signals_results = []

    # ── Phase 1: Model 1 (Pressure Exposure) ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 1: MODEL 1 — PRESSURE EXPOSURE")
    print(f"{'='*60}")

    for match_id in match_ids:
        tracking_path = Path(source_dir) / match_id / "tracking.parquet"
        if not tracking_path.exists():
            print(f"  ⚠️  {match_id}: tracking.parquet not found at {tracking_path}")
            continue

        result = process_pressure(
            match_id,
            tracking_path,
            config=DEFAULT_CONFIG,
        )
        pressure_results.append(result)

    # ── Phase 2: All Signals ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 2: ALL SIGNALS ({len(list_signals())} total)")
    print(f"{'='*60}")

    for match_id in match_ids:
        tracking_path = Path(source_dir) / match_id / "tracking.parquet"
        if not tracking_path.exists():
            continue

        result = process_signals(match_id, tracking_path, nrows=args.nrows, source_dir=source_dir)
        signals_results.append(result)

    # ── Phase 3: Merge ──────────────────────────────────────────────────
    if not args.skip_merge:
        print(f"\n{'='*60}")
        print(f"PHASE 3: MERGING OUTPUTS")
        print(f"{'='*60}")

        try:
            from src.merge_outputs import merge_all

            output_path = Path("./outputs/unified_fatigue_dataset.parquet")
            merge_all(output_path=str(output_path))
            print(f"  ✅ Unified dataset saved to: {output_path}")
        except Exception as exc:
            print(f"  ⚠️  Merge failed: {exc}")
            print("     You can re-run merge later with: python3 src/merge_outputs.py")

    # ── Summary ─────────────────────────────────────────────────────────
    total_elapsed = round(time.time() - total_start, 1)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {total_elapsed}s")
    print(f"  Matches: {len(match_ids)}")

    # Pressure summary
    p_success = [r for r in pressure_results if "error" not in r]
    print(f"\n  Model 1 (Pressure):")
    print(f"    Matches: {len(p_success)}/{len(pressure_results)}")
    total_players = sum(r.get("n_players", 0) for r in p_success)
    total_high = sum(r.get("high_pressure_blocks", 0) for r in p_success)
    total_low = sum(r.get("low_pressure_blocks", 0) for r in p_success)
    print(f"    Players: {total_players}")
    print(f"    High-pressure blocks: {total_high}")
    print(f"    Low-pressure blocks: {total_low}")

    # Signals summary
    print(f"\n  Signals:")
    for signal_name in list_signals():
        total_rows = sum(
            r["signals"].get(signal_name, {}).get("rows", 0)
            for r in signals_results
        )
        errors = [
            r["match_id"]
            for r in signals_results
            if r["signals"].get(signal_name, {}).get("error")
        ]
        status = "❌" if errors else "✅"
        err_msg = f" (errors: {errors})" if errors else ""
        print(f"    {status} {signal_name:<22} {total_rows:>6} rows{err_msg}")

    print(f"\n  Outputs:")
    print(f"    Pressure:  {OUTPUT_DIR}/")
    print(f"    Signals:   outputs/signals/{{signal_name}}/{{match_id}}.csv")
    print(f"    Unified:   outputs/unified_fatigue_dataset.parquet")
    print()


if __name__ == "__main__":
    main()
