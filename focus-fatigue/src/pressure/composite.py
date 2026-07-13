"""Pressure Composite — orchestrates Model 1 indicators into unified scores.

Combines four load indicators into a weighted pressure composite per
5-minute block for each defender:

    pressure = 1 + Σ(indicator_i / baseline_i)

Then classifies blocks:
    - Top quartile → high pressure
    - Bottom quartile → low pressure (control)
"""

from typing import Optional

import numpy as np
import pandas as pd

from .config import PressureConfig, DEFAULT_CONFIG


def compute_block_baselines(
    block_indicators: pd.DataFrame,
    config: Optional[PressureConfig] = None,
    player_col: str = "player_id",
    team_col: str = "team_id_opta",
) -> pd.DataFrame:
    """Compute per-player baselines from early-match blocks.

    Baselines are computed from the first N minutes (default 15 = first 3 blocks)
    of each player's data in each match. This captures their "fresh" state.

    Parameters
    ----------
    block_indicators : pd.DataFrame
        Combined per-block indicator values with columns:
        block_id, player_id, team_id_opta, and indicator columns.
    config : PressureConfig, optional
    player_col, team_col : str

    Returns
    -------
    pd.DataFrame
        One row per player with baseline values for each indicator:
        player_id, team_id_opta, n_baseline_blocks,
        opponents_nearby_baseline, depth_baseline,
        reorientation_rate_baseline, transition_rate_baseline
    """
    if config is None:
        config = DEFAULT_CONFIG

    # First N minutes = first few 5-minute blocks
    baseline_minutes = config.baseline_minutes  # 15
    n_baseline_blocks = max(1, baseline_minutes // config.block_window_minutes)

    # Determine which indicators are available
    indicator_cols = [c for c in block_indicators.columns
                      if c.startswith(("opponents_nearby", "depth", "reorientation", "transition"))
                      and not c.startswith(("opponents_nearby_std", "depth_std"))
                      and c.endswith(("_mean", "_count", "rate"))]

    # Extract block numbers
    block_indicators = block_indicators.copy()
    block_indicators["block_num"] = block_indicators["block_id"].apply(
        lambda x: int(x.split("_")[1])
    )

    # Use first N blocks per phase 1 for baseline
    # (phase 1 blocks have IDs like "1_0", "1_1", etc.)
    baseline_condition = (block_indicators["block_num"] < n_baseline_blocks)

    baseline_only = block_indicators[baseline_condition].copy()
    if len(baseline_only) == 0:
        # Fallback: use all available blocks
        baseline_only = block_indicators.copy()

    records = []
    for (pid, tid), grp in baseline_only.groupby([player_col, team_col]):
        row = {
            player_col: int(pid),
            team_col: int(tid),
            "n_baseline_blocks": len(grp),
        }
        for col in indicator_cols:
            vals = grp[col].dropna().values
            if len(vals) > 0 and np.mean(vals) > 0:
                row[f"{col}_baseline"] = float(np.mean(vals))
            else:
                row[f"{col}_baseline"] = np.nan
        records.append(row)

    return pd.DataFrame(records)


def compute_pressure_composite(
    block_indicators: pd.DataFrame,
    player_baselines: pd.DataFrame,
    config: Optional[PressureConfig] = None,
    player_col: str = "player_id",
    team_col: str = "team_id_opta",
) -> pd.DataFrame:
    """Compute per-block pressure composite for each defender.

    pressure = 1 + Σ(indicator_i / baseline_i)

    Where indicator_i is the player's value for that indicator in
    the current block, and baseline_i is their per-match baseline
    (from compute_block_baselines).

    Parameters
    ----------
    block_indicators : pd.DataFrame
        Per-block per-player indicator values.
    player_baselines : pd.DataFrame
        Per-player baseline values (from compute_block_baselines).
    config : PressureConfig, optional
    player_col, team_col : str

    Returns
    -------
    pd.DataFrame
        block_indicators with additional columns:
        - contribution_*: individual indicator contributions
        - pressure_score: weighted composite
        - n_contributing_indicators: how many indicators contributed
    """
    if config is None:
        config = DEFAULT_CONFIG

    result = block_indicators.copy()

    # Merge baselines
    merge_cols = [player_col, team_col]
    result = result.merge(
        player_baselines,
        on=merge_cols,
        how="left",
        suffixes=("", "_baseline"),
    )

    # Identify indicator columns and their baselines
    indicator_pairs = [
        ("opponents_nearby_mean", "opponents_nearby_mean_baseline"),
        ("depth_mean", "depth_mean_baseline"),
        ("reorientation_count", "reorientation_count_baseline"),
        ("transition_count", "transition_count_baseline"),
    ]

    # Also check for rate-based indicators
    if "reorientation_rate" in result.columns:
        indicator_pairs.append(
            ("reorientation_rate", "reorientation_rate_baseline")
        )
    if "transition_rate" in result.columns:
        indicator_pairs.append(
            ("transition_rate", "transition_rate_baseline")
        )

    result["pressure_score"] = 1.0
    result["n_contributing_indicators"] = 0

    for indicator_col, baseline_col in indicator_pairs:
        if indicator_col not in result.columns or baseline_col not in result.columns:
            continue

        contrib_col = f"contribution_{indicator_col}"
        # contribution = indicator / baseline (if baseline > 0)
        baseline_vals = result[baseline_col].values
        indicator_vals = result[indicator_col].values

        contrib = np.where(
            (baseline_vals > 0) & np.isfinite(baseline_vals) & np.isfinite(indicator_vals),
            indicator_vals / baseline_vals,
            0.0,
        )
        result[contrib_col] = contrib

        # Add to pressure score
        result["pressure_score"] += contrib

        # Count non-zero contributions
        result["n_contributing_indicators"] += (contrib > 0).astype(int)

    return result


def classify_pressure_blocks(
    pressure_df: pd.DataFrame,
    config: Optional[PressureConfig] = None,
) -> pd.DataFrame:
    """Classify blocks into high/low pressure categories.

    High pressure = top quartile (pressure_score > 75th percentile)
    Low pressure = bottom quartile (pressure_score < 25th percentile)

    Classification is done globally across all players and blocks
    to ensure quartile thresholds are meaningful.

    Parameters
    ----------
    pressure_df : pd.DataFrame
        Output from compute_pressure_composite.
    config : PressureConfig, optional

    Returns
    -------
    pd.DataFrame
        Input with additional columns:
        - pressure_quartile: quartile rank (1-4)
        - pressure_category: 'high', 'medium', or 'low'
    """
    if config is None:
        config = DEFAULT_CONFIG

    result = pressure_df.copy()

    # Compute global quartile thresholds
    scores = result["pressure_score"].dropna().values
    if len(scores) == 0:
        result["pressure_quartile"] = np.nan
        result["pressure_category"] = "unknown"
        return result

    high_thresh = np.percentile(scores, config.high_pressure_quantile * 100)
    low_thresh = np.percentile(scores, config.low_pressure_quantile * 100)

    # Assign quartile
    quartiles = pd.qcut(scores, q=4, labels=[1, 2, 3, 4], duplicates="drop")
    result["pressure_quartile"] = np.nan
    result.loc[result["pressure_score"].notna(), "pressure_quartile"] = quartiles

    # Assign category
    result["pressure_category"] = "medium"
    result.loc[result["pressure_score"] >= high_thresh, "pressure_category"] = "high"
    result.loc[result["pressure_score"] <= low_thresh, "pressure_category"] = "low"

    return result


def build_pressure_dataset(
    blocks: list[pd.DataFrame],
    proximity_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    reorientation_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    config: Optional[PressureConfig] = None,
) -> pd.DataFrame:
    """Merge all indicator aggregations into a unified pressure dataset.

    Parameters
    ----------
    blocks : list of pd.DataFrame
        Block-segmented tracking data.
    proximity_df : pd.DataFrame
        Opponent proximity per block (from opponent_proximity).
    depth_df : pd.DataFrame
        Defensive depth per block (from defensive_depth).
    reorientation_df : pd.DataFrame
        Reorientation counts per block (from reorientations).
    transition_df : pd.DataFrame
        Transition counts per block (from transitions).
    config : PressureConfig, optional

    Returns
    -------
    pd.DataFrame
        Unified dataset with one row per player per block.
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Start with block summary (for block metadata)
    from ..segments import block_summary
    summary = block_summary(blocks)

    # Merge all indicators
    dfs = []
    for indicator_df, prefix in [
        (proximity_df, "opp"),
        (depth_df, "depth"),
        (reorientation_df, "reo"),
        (transition_df, "trans"),
    ]:
        if len(indicator_df) > 0:
            dfs.append(indicator_df)

    if not dfs:
        return pd.DataFrame()

    # Merge all indicator DataFrames on block_id + player_id + team_id_opta
    merged = dfs[0]
    for df in dfs[1:]:
        merge_cols = ["block_id", "player_id", "team_id_opta"]
        merged = merged.merge(
            df[merge_cols + [c for c in df.columns if c not in merge_cols]],
            on=merge_cols,
            how="outer",
        )

    # Add block metadata
    merged = merged.merge(summary[["block_id", "phase", "block_num", "n_frames"]],
                          on="block_id", how="left")

    # Fill NaN transition counts with 0
    if "transition_count" in merged.columns:
        merged["transition_count"] = merged["transition_count"].fillna(0).astype(int)

    # Add reorientation rate (events per minute of block time)
    if "reorientation_count" in merged.columns and "n_frames" in merged.columns:
        block_minutes = merged["n_frames"] * config.frame_interval_s / 60.0
        merged["reorientation_rate"] = np.where(
            block_minutes > 0,
            merged["reorientation_count"] / block_minutes,
            0.0,
        )

    # Add transition rate (events per minute)
    if "transition_count" in merged.columns and "n_frames" in merged.columns:
        block_minutes = merged["n_frames"] * config.frame_interval_s / 60.0
        merged["transition_rate"] = np.where(
            block_minutes > 0,
            merged["transition_count"] / block_minutes,
            0.0,
        )

    return merged.sort_values(["player_id", "block_id"]).reset_index(drop=True)
