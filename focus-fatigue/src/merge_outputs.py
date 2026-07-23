#!/usr/bin/env python3
"""Merge all signal outputs + Model 1 into a single unified dataset.

Auto-discovers signal CSVs in ``outputs/signals/*/`` and Model 1 outputs
in ``outputs/pressure_exposure/``, then merges on
``(game_id, block_id, player_id, team_id_opta)``.

Usage:
    python3 src/merge_outputs.py
    python3 src/merge_outputs.py --output outputs/unified_fatigue_dataset.parquet
    python3 src/merge_outputs.py --signals-dir outputs/signals
    python3 src/merge_outputs.py --pressure-dir outputs/pressure_exposure
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Column Constants ─────────────────────────────────────────────────────────

MERGE_KEYS = ["game_id", "block_id", "player_id", "team_id_opta"]
"""Columns used to join Model 1 and signal DataFrames."""

SIGNAL_VALUE_COL = "signal_value"
"""Column name containing the primary signal measurement."""


# ── Discovery Helpers ────────────────────────────────────────────────────────


def discover_signal_csvs(signals_dir: str) -> dict[str, list[Path]]:
    """Discover all signal output CSV files by signal name.

    Parameters
    ----------
    signals_dir : str
        Root directory containing ``{signal_name}/{match_id}.csv`` files.

    Returns
    -------
    dict[str, list[Path]]
        ``{signal_name: [path1, path2, ...]}``
    """
    sd = Path(signals_dir)
    if not sd.exists():
        print(f"  ⚠️  Signals output directory not found: {sd}")
        return {}

    signal_files: dict[str, list[Path]] = {}
    for signal_dir in sorted(sd.iterdir()):
        if not signal_dir.is_dir():
            continue
        signal_name = signal_dir.name
        csvs = sorted(signal_dir.glob("*.csv"))
        if csvs:
            signal_files[signal_name] = csvs

    return signal_files


def discover_pressure_csvs(pressure_dir: str, prefix: str = "pressure_composite") -> list[Path]:
    """Discover Model 1 pressure output CSV files.

    Looks for files like ``pressure_composite_{match_id}.csv``.

    Parameters
    ----------
    pressure_dir : str
        Directory containing pressure exposure outputs.
    prefix : str
        Filename prefix (default: ``'pressure_composite'``).

    Returns
    -------
    list[Path]
        Sorted list of matching CSV paths.
    """
    pd_ = Path(pressure_dir)
    if not pd_.exists():
        print(f"  ⚠️  Pressure output directory not found: {pd_}")
        return []

    return sorted(pd_.glob(f"{prefix}_*.csv"))


# ── Loading ─────────────────────────────────────────────────────────────────


def load_pressure_data(
    pressure_dir: str,
    prefix: str = "pressure_composite",
) -> pd.DataFrame:
    """Load and concatenate all Model 1 pressure composite CSVs.

    Parameters
    ----------
    pressure_dir : str
        Directory containing pressure exposure outputs.
    prefix : str
        Filename prefix for pressure composite files.

    Returns
    -------
    pd.DataFrame
        Concatenated pressure data with ``game_id`` column.
    """
    files = discover_pressure_csvs(pressure_dir, prefix=prefix)

    if not files:
        print("  ⚠️  No pressure composite files found.")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)

            # Ensure merge keys exist
            for key in MERGE_KEYS:
                if key not in df.columns:
                    # Try to infer game_id from filename
                    if key == "game_id":
                        match_id = f.stem.replace(f"{prefix}_", "")
                        df["game_id"] = match_id

            frames.append(df)
        except Exception as exc:
            print(f"  ⚠️  Error loading {f}: {exc}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # Standardise types for merge (handle NaN safely)
    for col in ["player_id", "team_id_opta"]:
        if col in combined.columns:
            combined[col] = combined[col].fillna(-1).astype(int)

    return combined


def load_signal_data(signals_dir: str) -> dict[str, pd.DataFrame]:
    """Load all signal CSVs and return a dict of DataFrames keyed by signal name.

    Parameters
    ----------
    signals_dir : str
        Root directory of signal outputs.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{signal_name: concatenated_df}``
    """
    signal_files = discover_signal_csvs(signals_dir)

    if not signal_files:
        print("  ⚠️  No signal output files found.")
        return {}

    signal_dfs: dict[str, pd.DataFrame] = {}
    for signal_name, csv_paths in sorted(signal_files.items()):
        frames: list[pd.DataFrame] = []
        for path in csv_paths:
            try:
                df = pd.read_csv(path, low_memory=False)
                frames.append(df)
            except Exception as exc:
                print(f"  ⚠️  Error loading {path}: {exc}")

        if frames:
            signal_dfs[signal_name] = pd.concat(frames, ignore_index=True)
            print(f"  Loaded {signal_name}: {len(signal_dfs[signal_name])} rows "
                  f"from {len(csv_paths)} files")

    return signal_dfs


# ── Signal-Level Detection ───────────────────────────────────────────────────


def _is_team_level_signal(df: pd.DataFrame) -> bool:
    """Check whether *df* is a team-level (not player-level) signal.

    Team-level signals either lack a ``player_id`` column entirely, or have
    *all* identical ``player_id`` values (e.g. the ``-1`` placeholder used by
    ``team_polarisation``).

    Parameters
    ----------
    df : pd.DataFrame
        Signal DataFrame to inspect.

    Returns
    -------
    bool
        ``True`` if the signal is team-level.
    """
    if "player_id" not in df.columns:
        return True
    # If player_id exists with only one distinct value (handles NaN safely), it
    # is a team-level signal.
    return df["player_id"].nunique(dropna=True) == 1


# ── Merging ──────────────────────────────────────────────────────────────────


def merge_all(
    signals_dir: str = "outputs/signals",
    pressure_dir: str = "outputs/pressure_exposure",
    output_path: str = "outputs/unified_fatigue_dataset.parquet",
) -> pd.DataFrame:
    """Merge Model 1 pressure data with all signal outputs.

    Strategy:
    1. Start with pressure data as the base (one row per player-block)
    2. For each signal, pivot the signal_value into a column named
       ``{signal_name}`` so each row stays wide
    3. Merge on ``(game_id, block_id, player_id, team_id_opta)``

    Parameters
    ----------
    signals_dir : str
        Path to signal outputs root directory.
    pressure_dir : str
        Path to Model 1 pressure output directory.
    output_path : str
        Path to save the merged parquet file.

    Returns
    -------
    pd.DataFrame
        Merged unified dataset, or empty DataFrame if no data found.
    """
    print("Loading pressure data...")
    pressure_df = load_pressure_data(pressure_dir)
    if len(pressure_df) == 0:
        print("  ⚠️  No pressure data loaded. Checking signals only mode.")

    print("Loading signal data...")
    signal_dfs = load_signal_data(signals_dir)

    if len(pressure_df) == 0 and not signal_dfs:
        print("  ❌ No data found at all. Nothing to merge.")
        return pd.DataFrame()

    # ── Strategy: start with the larger base and merge signals in ──────────
    if len(pressure_df) > 0:
        base = pressure_df.copy()
        base_source = "pressure"
    else:
        # No pressure data — use the first signal as base
        first_signal = list(signal_dfs.keys())[0]
        base = signal_dfs[first_signal].copy()
        base_source = f"signal ({first_signal})"

    print(f"  Base table: {base_source} ({len(base)} rows)")

    # ── Split signals into team-level and player-level ─────────────────
    team_signal_names = {
        name for name, df in signal_dfs.items()
        if _is_team_level_signal(df)
    }
    player_signal_names = set(signal_dfs.keys()) - team_signal_names

    if team_signal_names:
        print(f"  Team-level signals: {', '.join(sorted(team_signal_names))}")
    if player_signal_names:
        print(f"  Player-level signals: {', '.join(sorted(player_signal_names))}")

    # ── Phase 1: Merge team-level signals (no player_id) ────────────────
    #    Team-level signals have one value per (game_id, block_id,
    #    team_id_opta).  Merging without player_id broadcasts that value
    #    to every player row in the base, avoiding NaN matches from
    #    the -1 placeholder.
    team_merge_keys = [k for k in MERGE_KEYS if k != "player_id"]

    for signal_name in sorted(team_signal_names):
        signal_df = signal_dfs[signal_name]

        merge_cols = [c for c in team_merge_keys if c in signal_df.columns]
        if not merge_cols:
            print(f"  ⚠️  {signal_name}: no merge keys found, skipping.")
            continue

        # Pivot signal values into a column named after the signal
        signal_slim = signal_df[merge_cols + [SIGNAL_VALUE_COL]].copy()
        signal_slim = signal_slim.rename(
            columns={SIGNAL_VALUE_COL: signal_name}
        )

        # Drop duplicate rows (multiple entries per team-block from
        # different sub-tables)
        signal_slim = signal_slim.drop_duplicates(subset=merge_cols)

        # Merge (many-to-one: base has many players, signal has one team
        # value per block → broadcasts to all players)
        base = base.merge(signal_slim, on=merge_cols, how="left")

        n_matched = signal_slim[merge_cols].drop_duplicates().shape[0]
        print(f"  + {signal_name} [team-level]: {n_matched} matches → "
              f"{len(base)} rows in base")

    # ── Phase 2: Merge player-level signals (with full MERGE_KEYS) ──────
    for signal_name in sorted(player_signal_names):
        signal_df = signal_dfs[signal_name]

        merge_cols = [c for c in MERGE_KEYS if c in signal_df.columns]
        if not merge_cols:
            print(f"  ⚠️  {signal_name}: no merge keys found, skipping.")
            continue

        # Pivot signal values into a column named after the signal
        signal_slim = signal_df[merge_cols + [SIGNAL_VALUE_COL]].copy()
        signal_slim = signal_slim.rename(
            columns={SIGNAL_VALUE_COL: signal_name}
        )

        # Drop duplicate rows (multiple entries per player-block from
        # different sub-tables in the same signal)
        signal_slim = signal_slim.drop_duplicates(subset=merge_cols)

        # Merge
        base = base.merge(signal_slim, on=merge_cols, how="left")

        n_matched = signal_slim[merge_cols].drop_duplicates().shape[0]
        print(f"  + {signal_name} [player-level]: {n_matched} matches → "
              f"{len(base)} rows in base")

    # ── Save ─────────────────────────────────────────────────────────────
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(output, index=False)
    print(f"\n  ✅ Unified dataset saved to: {output}")
    print(f"     Shape: {base.shape[0]} rows × {base.shape[1]} columns")

    return base


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Merge all signal outputs + Model 1 into a unified dataset."
    )
    parser.add_argument(
        "--output", type=str, default="outputs/unified_fatigue_dataset.parquet",
        help="Output path for the merged parquet file "
             "(default: outputs/unified_fatigue_dataset.parquet)"
    )
    parser.add_argument(
        "--signals-dir", type=str, default="outputs/signals",
        help="Path to signal outputs root directory (default: outputs/signals)"
    )
    parser.add_argument(
        "--pressure-dir", type=str, default="outputs/pressure_exposure",
        help="Path to Model 1 pressure output directory "
             "(default: outputs/pressure_exposure)"
    )
    args = parser.parse_args()

    merge_all(
        signals_dir=args.signals_dir,
        pressure_dir=args.pressure_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
