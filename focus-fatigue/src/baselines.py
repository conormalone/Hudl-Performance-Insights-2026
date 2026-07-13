"""Baseline computation for fatigue signal normalisation.

Per-player baselines are computed from the first N minutes of each match,
under the assumption that fatigue is minimal early in the game.

Rotation players (<180 min across dataset) fall back to position-level averages.
"""

from typing import Optional

import numpy as np
import pandas as pd


def compute_player_baselines(
    df: pd.DataFrame,
    first_n_minutes: int = 15,
    time_col: str = "time",
    phase_col: str = "phase",
    player_col: str = "player_id",
    position_col: Optional[str] = None,
    metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute per-player baseline statistics from early match data.

    Parameters
    ----------
    df : pd.DataFrame
        Tracking data for a single match (all phases).
    first_n_minutes : int, default 15
        Number of minutes from match start to use for baseline.
    time_col : str, default 'time'
        Column with time in seconds.
    phase_col : str, default 'phase'
        Column with phase (1 or 2).
    player_col : str, default 'player_id'
        Column identifying players.
    position_col : str, optional
        Column with position/role labels (e.g. 'CB', 'WB').
    metrics : list of str, optional
        Metric columns to compute baselines for.
        Default: speed, speed_x, speed_y, v_mag if available.

    Returns
    -------
    pd.DataFrame
        One row per player with baseline mean and std for each metric.
        Columns: player_id, metric_mean, metric_std, n_frames, position (if available)
    """
    # Default metrics
    if metrics is None:
        metrics = ["speed"]
        for col in ["speed_x", "speed_y", "v_mag", "turn_rate"]:
            if col in df.columns:
                metrics.append(col)

    # Filter to first N minutes of phase 1 (first half only)
    baseline_end = first_n_minutes * 60  # convert to seconds
    baseline_df = df[
        (df[phase_col] == 1) & (df[time_col] <= baseline_end)
    ].copy()

    if len(baseline_df) == 0:
        # Fallback: if no phase 1 data, use first N minutes overall
        all_times = sorted(df[time_col].unique())
        cutoff = all_times[min(len(all_times) - 1, int(first_n_minutes * 60 / 0.04))]
        baseline_df = df[df[time_col] <= cutoff].copy()

    # Compute per-player stats
    records = []
    for pid, group in baseline_df.groupby(player_col):
        row = {player_col: pid, "n_frames": len(group)}
        for metric in metrics:
            if metric in group.columns:
                values = group[metric].dropna().values
                if len(values) > 0:
                    row[f"{metric}_mean"] = float(np.mean(values))
                    row[f"{metric}_std"] = float(np.std(values))
                else:
                    row[f"{metric}_mean"] = np.nan
                    row[f"{metric}_std"] = np.nan

        if position_col and position_col in group.columns:
            row["position"] = group[position_col].mode().iloc[0] if not group[position_col].mode().empty else "?"

        records.append(row)

    return pd.DataFrame(records)


def compute_global_baselines(
    player_baselines: dict[str, pd.DataFrame],
    min_match_minutes: int = 180,
    player_col: str = "player_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-player and per-position fallback baselines.

    Parameters
    ----------
    player_baselines : dict
        {match_id: baseline_df} from compute_player_baselines.
    min_match_minutes : int, default 180
        Minimum total tracked minutes to keep per-player baseline.
        Players below this threshold default to position-level averages.

    Returns
    -------
    player_level : pd.DataFrame
        Full player baselines (filtered to those with enough minutes).
    position_level : pd.DataFrame
        Position-level fallback baselines.
    """
    # Combine all match baselines
    all_baselines = pd.concat(player_baselines.values(), ignore_index=True)

    # Count total minutes per player
    # Each frame = 0.04s, so minutes = n_frames * 0.04 / 60
    player_minutes = (
        all_baselines.groupby(player_col)["n_frames"]
        .sum()
        .apply(lambda x: x * 0.04 / 60)
    )

    # Compute per-player averages across all matches
    metric_cols = [c for c in all_baselines.columns
                   if c.endswith("_mean") or c.endswith("_std")]
    player_level = (
        all_baselines.groupby(player_col)[metric_cols]
        .mean()
        .reset_index()
    )
    player_level["total_minutes"] = player_level[player_col].map(
        player_minutes
    )

    # Players with enough minutes
    player_level = player_level[
        player_level["total_minutes"] >= min_match_minutes
    ].copy()

    # Position-level fallback
    if "position" in all_baselines.columns:
        position_level = (
            all_baselines.groupby("position")[metric_cols]
            .mean()
            .reset_index()
        )
    else:
        position_level = (
            all_baselines[metric_cols]
            .mean()
            .to_frame()
            .T
        )
        position_level["position"] = "ALL"

    return player_level, position_level
