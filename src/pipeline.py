"""Main pipeline — orchestrates data loading, signal computation, and output.

Imports all signal modules to ensure they are registered. Provides a
convenient interface for running the two new signals (team_polarisation
and team_centroid_distance) across matches, either from the CLI or
from a notebook.

Usage:
    python3 src/pipeline.py --all                    # All matches
    python3 src/pipeline.py --match 1 --match 2      # Specific matches
    python3 src/pipeline.py --all --nrows 500        # Quick test
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Import signal modules to trigger @register_signal ──────────────
import src.signals.polarisation  # noqa: F401
import src.signals.team_centroid_distance  # noqa: F401
# ────────────────────────────────────────────────────────────────────

from src.config import (
    SIGNAL_OUTPUT_DIR,
    DEFAULT_MATCH_LIMIT,
    LOG_LEVEL,
    LOG_FORMAT,
    FPS,
)
from src.model1.base import get_available_match_ids, load_match_tracking, load_match_events
from src.model1.block_builder import build_blocks
from src.signals.registry import discover_signals, get_signal_names
from src.smoothing import compute_velocity_features

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline for new cognitive-load signals (polarisation, centroid distance).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Process all available matches")
    group.add_argument("--match", type=int, action="append", dest="match_ids",
                       help="Specific match ID(s) to process")

    parser.add_argument("--signal", type=str, default=None,
                        help="Run only this signal (default: all registered)")
    parser.add_argument("--list", action="store_true",
                        help="List registered signals and exit")
    parser.add_argument("--limit", type=int, default=DEFAULT_MATCH_LIMIT,
                        help=f"Max matches to process (default: {DEFAULT_MATCH_LIMIT})")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Limit tracking rows per match (for testing)")
    parser.add_argument("--output-dir", type=str, default=str(SIGNAL_OUTPUT_DIR),
                        help="Output directory for signal CSVs")
    return parser


def run_signals_for_match(
    match_id: int,
    signal_names: list[str],
    nrows: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """Run selected signals on a single match.

    Returns dict of signal_name -> output DataFrame.
    """
    from src.model1.block_builder import build_blocks

    signals = discover_signals()
    results = {}

    # Load and process tracking data
    df = load_match_tracking(match_id, nrows=nrows)
    if df is None:
        logger.warning("Match %d: no tracking data, skipping", match_id)
        return results

    # Compute velocity features (adds vx, vy, speed, heading)
    if all(c in df.columns for c in ["x", "y", "frame", "player_id"]):
        df = compute_velocity_features(df)

    # Build blocks (5-minute windows)
    blocks = build_blocks(df, match_id)
    if not blocks:
        logger.warning("Match %d: no blocks produced, skipping", match_id)
        return results

    # Run each signal
    for name in signal_names:
        signal_cls = signals[name]
        sig_instance = signal_cls()
        logger.info("Match %d: running signal '%s'", match_id, name)

        try:
            output = sig_instance.compute(blocks, match_id=match_id)
            if output is not None and not output.empty:
                results[name] = output
                logger.info("Match %d: signal '%s' → %d rows",
                            match_id, name, len(output))
        except Exception as e:
            logger.error("Match %d: signal '%s' crashed: %s", match_id, name, e)
            import traceback
            traceback.print_exc()

    return results


def save_signal_output(
    results: dict[str, pd.DataFrame],
    match_id: int,
    output_dir: Path,
) -> None:
    """Save each signal's output to a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        path = output_dir / f"signal_{name}_match_{match_id}.csv"
        df.to_csv(path, index=False)
        logger.info("Saved %s (%d rows)", path.name, len(df))


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    signal_names = [args.signal] if args.signal else get_signal_names()

    if args.list:
        print("Registered signals in src/signals/:")
        for name in get_signal_names():
            print(f"  - {name}")
        return

    # Determine match IDs
    if args.match_ids:
        match_ids = args.match_ids
    else:
        match_ids = get_available_match_ids(limit=args.limit)

    if not match_ids:
        logger.error("No matches to process.")
        sys.exit(1)

    logger.info("Processing %d matches with signals: %s", len(match_ids), signal_names)
    logger.info("Output directory: %s", output_dir)

    for match_id in match_ids:
        logger.info("=== Match %d ===", match_id)
        match_results = run_signals_for_match(
            match_id, signal_names,
            nrows=args.nrows,
        )
        if match_results:
            save_signal_output(match_results, match_id, output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
