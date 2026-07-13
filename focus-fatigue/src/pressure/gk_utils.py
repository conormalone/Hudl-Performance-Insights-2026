"""Shared goalkeeper detection utility.

Unifies GK detection logic across all Model 1 modules to avoid
the maintenance liability of three separate implementations (H3 fix).

Strategy (in order of preference):
1. Use the raw `goalkeeper` column if available (0% nulls in sample data)
2. Fall back to heuristic: jersey_no == 1 + position near goal line + centred y
"""

import numpy as np
import pandas as pd

from typing import Optional


def flag_goalkeepers(
    df: pd.DataFrame,
    player_col: str = "player_id",
    team_col: str = "team_id_opta",
    jersey_col: str = "jersey_no",
    gk_col: str = "goalkeeper",
    x_col: str = "x",
    y_col: str = "y",
    goal_x_threshold: float = 45.0,
    centre_y_threshold: float = 10.0,
) -> pd.Series:
    """Return a boolean Series flagging goalkeeper rows in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Data. Should contain at minimum player_id, team_id_opta, and
        either a boolean `goalkeeper` column (preferred) or jersey_no.
    player_col, team_col, jersey_col, gk_col : str
        Column names.
    x_col, y_col : str
        Position columns for heuristic fallback.
    goal_x_threshold : float
        Absolute x-value above which a player is considered near the
        goal line (heuristic fallback).
    centre_y_threshold : float
        Absolute y-value below which a player is considered centred
        (heuristic fallback).

    Returns
    -------
    pd.Series
        Boolean, True for goalkeeper rows.
    """
    # Strategy 1: Use raw goalkeeper column if available
    if gk_col in df.columns and df[gk_col].dtype in (bool, np.bool_, "bool"):
        return df[gk_col].astype(bool)

    if gk_col in df.columns:
        # Check if it's a nullable boolean (Int64, etc.)
        try:
            return df[gk_col].astype(bool)
        except (ValueError, TypeError):
            pass

    # Strategy 2: Heuristic fallback
    return _heuristic_gk_detection(
        df, player_col, team_col, jersey_col,
        x_col, y_col, goal_x_threshold, centre_y_threshold,
    )


def _heuristic_gk_detection(
    df: pd.DataFrame,
    player_col: str = "player_id",
    team_col: str = "team_id_opta",
    jersey_col: str = "jersey_no",
    x_col: str = "x",
    y_col: str = "y",
    goal_x_threshold: float = 45.0,
    centre_y_threshold: float = 10.0,
) -> pd.Series:
    """Heuristic GK detection: jersey 1 + near goal line + centred y.

    This matches the logic used in load_tracking.py's _resolve_goalkeeper,
    which is the most conservative (fewest false positives) of the three
    original implementations.
    """
    # Compute per-player mean positions
    gk_records = {}
    for (pid, tid), grp in df.groupby([player_col, team_col]):
        if jersey_col not in grp.columns:
            continue
        jersey = grp[jersey_col].iloc[0]
        if jersey == 1:
            mean_x = grp[x_col].mean()
            mean_y = grp[y_col].mean()
            near_goal = abs(mean_x) > goal_x_threshold
            near_centre = abs(mean_y) < centre_y_threshold
            if near_goal and near_centre:
                gk_records[(pid, tid)] = True

    return df.apply(
        lambda r: gk_records.get((r[player_col], r[team_col]), False),
        axis=1,
    )
