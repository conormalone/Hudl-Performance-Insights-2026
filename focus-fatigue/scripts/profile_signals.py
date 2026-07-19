#!/usr/bin/env python3
"""Profile each signal's compute time on a single match (sample match 2215790).

Runs each signal independently, reports timing, and identifies bottlenecks.
Usage: python3 scripts/profile_signals.py [--match 2215790] [--nrows N]
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders import load_tracking_statsperform
from src.smoothing import smooth_trajectory, compute_velocity_features
from src.segments import split_into_blocks

# Import all signal modules to register them
import src.signals.drift
import src.signals.shift
import src.signals.pressing
import src.signals.transition
from src.signals.registry import list_signals, SIGNAL_REGISTRY


# ── Config ──────────────────────────────────────────────────────────────────

DATA_ROOT = Path("/mnt/usb/conor_downloads/team_mappings")
TRACKING_DIR = DATA_ROOT / "tracking"
SAMPLE_DIR = DATA_ROOT / "sample"
SHAPE_DIR = DATA_ROOT / "shape_outputs"
TEAM_MAPPINGS_PATH = DATA_ROOT / "team_mappings.csv"


def find_shape_file(match_id):
    for d in [SHAPE_DIR, SAMPLE_DIR, DATA_ROOT, DATA_ROOT / "shape_outputs"]:
        for c in [d / f"{match_id}.json", d / match_id / f"{match_id}.json"]:
            if c.exists():
                return c
    return None


def profile_signal_match(match_id, nrows=None):
    """Load match, smooth, split blocks, then time each signal."""
    tracking_path = SAMPLE_DIR / match_id / "tracking.parquet"
    if not tracking_path.exists():
        tracking_path = TRACKING_DIR / match_id / "tracking.parquet"
    if not tracking_path.exists():
        print(f"❌ Match {match_id}: tracking.parquet not found")
        return

    # Set team mappings path for drift signal
    src.signals.drift._TEAM_MAPPINGS_PATH = TEAM_MAPPINGS_PATH

    print(f"\n{'='*70}")
    print(f"PROFILING: match {match_id}")
    print(f"{'='*70}")

    # ── Data loading + smoothing (common to all signals) ──────────────
    t0 = time.time()
    df = load_tracking_statsperform(str(tracking_path), match_id=match_id,
                                     normalise_dop=True, include_ball=True)
    if nrows:
        df = df.head(nrows)
    df["frame"] = df["frame_count"]
    load_time = time.time() - t0
    print(f"\nLoad:   {load_time:.1f}s ({len(df):,} rows)")

    t0 = time.time()
    df = smooth_trajectory(df, inplace=False)
    df = compute_velocity_features(df)
    smooth_time = time.time() - t0
    print(f"Smooth: {smooth_time:.1f}s")

    t0 = time.time()
    blocks_dfs = split_into_blocks(df, window_minutes=5, min_frames=100)
    block_time = time.time() - t0
    print(f"Blocks: {block_time:.1f}s ({len(blocks_dfs)} blocks)")
    print(f"\n{'─'*70}")

    # Determine teams (for pressing_accuracy)
    outfield_teams = df[df["player_id"] != -1]["team_id_opta"].unique()
    team_a = int(outfield_teams[0]) if len(outfield_teams) > 0 else 0
    team_b = int(outfield_teams[1]) if len(outfield_teams) > 1 else 0

    # Shape path (for positional_drift)
    shape_path = find_shape_file(match_id)
    shape_available = shape_path is not None

    # ── Profile each signal ─────────────────────────────────────────────
    results = {}
    registered = list_signals()

    for signal_name in registered:
        if signal_name not in SIGNAL_REGISTRY:
            continue

        sig_cls = SIGNAL_REGISTRY[signal_name]
        ts = time.time()
        print(f"\n  ▶ {signal_name}...", flush=True)

        try:
            signal = sig_cls()
            kwargs = {"match_df": df, "game_id": match_id}

            if signal_name == "shift_latency":
                # shift_latency needs dict-format blocks
                block_dicts = []
                for blk in blocks_dfs:
                    bid = str(blk["block_id"].iloc[0])
                    phase = int(bid.split("_")[0])
                    block_dicts.append({
                        "block_id": bid, "phase": phase,
                        "start_frame": int(blk["frame_count"].min()),
                        "end_frame": int(blk["frame_count"].max()),
                    })
                kwargs["blocks"] = block_dicts
            else:
                kwargs["blocks"] = blocks_dfs

            if signal_name == "positional_drift":
                if shape_available:
                    kwargs["shape_path"] = str(shape_path)
                else:
                    print(f"  ⚠️  No shape file, skipping")
                    continue

            if signal_name == "pressing_accuracy":
                kwargs["own_team_id"] = team_a
                kwargs["opponent_team_id"] = team_b

            import psutil
            mem_before = psutil.Process().memory_info().rss / 1e6

            output_df = signal.compute(**kwargs)
            n_rows = len(output_df)

            mem_after = psutil.Process().memory_info().rss / 1e6

            et = time.time() - ts
            mem_delta = mem_after - mem_before

            print(f"  ✅ {n_rows:>6} rows in {et:>7.1f}s  ({n_rows/max(et,0.1):>8.1f} rows/s)"
                  f"  mem: +{mem_delta:.0f}MB")
            results[signal_name] = {"rows": n_rows, "elapsed_s": round(et, 2),
                                    "rows_per_sec": round(n_rows / max(et, 0.1), 1)}

        except Exception as e:
            import traceback
            et = time.time() - ts
            print(f"  ❌ Error at {et:.1f}s: {e}")
            traceback.print_exc()
            results[signal_name] = {"rows": 0, "elapsed_s": round(et, 2),
                                    "error": str(e)}

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PROFILE SUMMARY — match {match_id}")
    print(f"{'='*70}")
    print(f"{'Signal':<30} {'Rows':>8} {'Time':>8} {'Rows/s':>10}")
    print(f"{'─'*60}")
    for name in registered:
        if name in results:
            r = results[name]
            if "error" in r:
                print(f"  {name:<28} {'❌':>8} {r['elapsed_s']:>7.1f}s  {r['error']}")
            else:
                print(f"  {name:<28} {r['rows']:>8,} {r['elapsed_s']:>7.1f}s  {r.get('rows_per_sec', 0):>9.1f}")
        else:
            print(f"  {name:<28} {'skipped':>20}")

    overhead = load_time + smooth_time + block_time
    print(f"\n  Data prep overhead: {overhead:.1f}s")
    print(f"  Total signal time: {sum(r.get('elapsed_s', 0) for r in results.values()):.1f}s")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile signal computation")
    parser.add_argument("--match", type=str, default="2215790")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows (testing)")
    args = parser.parse_args()

    profile_signal_match(args.match, nrows=args.nrows)


if __name__ == "__main__":
    main()
