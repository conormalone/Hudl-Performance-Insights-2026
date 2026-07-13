"""Defensive Depth — Load Indicator 2.

Computes the distance from a player's own goal line per frame.
In DOP-normalised coordinates (all attacks left→right), the
defending team's goal is always at x = -57.

Per-frame: depth = x - goal_line_x
Per-block: mean + std of depth values for each defender.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .config import PressureConfig, DEFAULT_CONFIG


def compute_defensive_depth(
    df: pd.DataFrame,
    goal_line_x: float = -57.0,
    config: Optional[PressureConfig] = None,
    player_col: str = "player_id",
    team_col: str = "team_id_opta",
    x_col: str = "x",
) -> pd.DataFrame:
    """Compute distance from own goal line per frame for each player.

    In DOP-normalised coordinates, own goal is always at x = goal_line_x
    (left edge of pitch, constant -57 across all frames after canonicalisation).
    This means defenders closer to their own goal have low depth values,
    while defenders pushed up the pitch have high depth values (exposed = pressure).

    Parameters
    ----------
    df : pd.DataFrame
        Tracking data for a single match, DOP-normalised.
    goal_line_x : float, default -57.0
        x-coordinate of the defending team's goal line.
    config : PressureConfig, optional
        Overrides goal_line_x if provided.
    player_col : str, default 'player_id'
    team_col : str, default 'team_id_opta'
    x_col : str, default 'x'

    Returns
    -------
    pd.DataFrame
        Original rows with added column 'defensive_depth' containing
        the distance from own goal line. Only computed for outfield
        players (ball and GKs excluded).
    """
    if config is not None:
        goal_line_x = config.goal_line_x

    # Identify GKs: jersey 1 + near goal line
    result = df.copy()
    result["defensive_depth"] = np.nan

    # Exclude ball
    outfield = result[result[player_col] != -1].copy()

    # Apply the heuristic: for any outfield player (not GK),
    # depth = x - goal_line_x (always >= 0 for valid pitch coords)
    is_gk = _flag_gks(outfield)

    # Compute depth for non-GK players
    non_gk_mask = ~is_gk
    outfield_idx = outfield.index[non_gk_mask]
    result.loc[outfield_idx, "defensive_depth"] = (
        outfield.loc[outfield_idx, x_col].values - goal_line_x
    )

    return result


def _flag_gks(df: pd.DataFrame) -> pd.Series:
    """Flag goalkeepers using jersey number + position heuristic.

    Returns boolean series True for GK rows.
    """
    gk_records = {}
    for (pid, tid), grp in df.groupby(["player_id", "team_id_opta"]):
        jersey = grp["jersey_no"].iloc[0]
        if jersey == 1:
            mean_x = grp["x"].mean()
            if abs(mean_x) > 45:
                gk_records[(pid, tid)] = True

    return df.apply(
        lambda r: gk_records.get((r["player_id"], r["team_id_opta"]), False),
        axis=1,
    )


def aggregate_defensive_depth_to_blocks(
    blocks: list[pd.DataFrame],
    config: Optional[PressureConfig] = None,
) -> pd.DataFrame:
    """Aggregate defensive depth per player per block.

    Parameters
    ----------
    blocks : list of pd.DataFrame
        Tracking data split into blocks. Each must have 'defensive_depth'.

    Returns
    -------
    pd.DataFrame
        One row per player per block with:
        block_id, player_id, team_id_opta, n_frames_depth,
        depth_mean, depth_std
    """
    records = []
    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        depth_df = blk[blk["defensive_depth"].notna()]
        if len(depth_df) == 0:
            continue

        for (pid, tid), grp in depth_df.groupby(["player_id", "team_id_opta"]):
            vals = grp["defensive_depth"].values
            records.append({
                "block_id": bid,
                "player_id": int(pid),
                "team_id_opta": int(tid),
                "n_frames_depth": len(vals),
                "depth_mean": float(np.mean(vals)),
                "depth_std": float(np.std(vals)),
            })

    return pd.DataFrame(records)
