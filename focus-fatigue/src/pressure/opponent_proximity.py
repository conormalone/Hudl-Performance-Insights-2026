"""Opponent Proximity — Load Indicator 1.

Counts the number of opposing players within a configurable radius
of each defender, per frame. Aggregated to per-block mean count.

Uses scipy.spatial.distance.cdist for efficient per-frame computation.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .config import PressureConfig, DEFAULT_CONFIG


from .gk_utils import flag_goalkeepers

def _flag_goalkeepers(df: pd.DataFrame) -> set:
    """Identify goalkeeper (player_id, team_id_opta) pairs.

    Delegates to the shared gk_utils.flag_goalkeepers for consistent
    detection across all Model 1 modules (H3 fix).
    """
    is_gk = flag_goalkeepers(df)
    gk_set = set()
    gk_rows = df[is_gk]
    for _, row in gk_rows.iterrows():
        gk_set.add((int(row["player_id"]), int(row["team_id_opta"])))
    return gk_set


def compute_opponent_proximity(
    df: pd.DataFrame,
    radius: float = 7.0,
    config: Optional[PressureConfig] = None,
    frame_col: str = "frame_count",
    player_col: str = "player_id",
    team_col: str = "team_id_opta",
    pos_cols: tuple[str, str] = ("x", "y"),
) -> pd.DataFrame:
    """Per-frame count of opponents within radius of each player.

    Uses scipy cdist for efficient cross-team distance computation
    within each frame. Processes all frames via pandas groupby.

    Parameters
    ----------
    df : pd.DataFrame
        Tracking data for a single match (DOP-normalised).
    radius : float, default 7.0
        Distance threshold in metres.
    frame_col : str, default 'frame_count'
    player_col : str, default 'player_id'
    team_col : str, default 'team_id_opta'
    pos_cols : tuple, default ('x', 'y')

    Returns
    -------
    pd.DataFrame
        Original rows with added column 'opponents_nearby'.
    """
    if config is not None:
        radius = config.opponent_radius

    col_x, col_y = pos_cols
    result = df.copy()
    result["opponents_nearby"] = 0.0

    # Filter: outfield players only (exclude ball and GKs)
    outfield = result[result[player_col] != -1].copy()
    gk_set = _flag_goalkeepers(outfield)
    gk_mask = outfield.apply(
        lambda r: (r[player_col], r[team_col]) in gk_set, axis=1
    )
    outfield = outfield[~gk_mask].copy()

    if len(outfield) == 0:
        return result

    # Group by frame and compute cross-team distances
    frame_groups = outfield.groupby(frame_col, sort=True)

    proximity_values = {}  # {(frame, pid, tid): count}

    for frame_id, frame_df in frame_groups:
        n = len(frame_df)
        if n < 2:
            continue

        teams = frame_df[team_col].values.astype(np.int64)
        xs = frame_df[col_x].values.astype(np.float64)
        ys = frame_df[col_y].values.astype(np.float64)
        pids = frame_df[player_col].values.astype(np.int64)

        unique_teams = np.unique(teams)
        if len(unique_teams) < 2:
            continue

        team_a, team_b = unique_teams[0], unique_teams[1]
        a_mask = teams == team_a
        b_mask = teams == team_b

        xs_a, ys_a = xs[a_mask], ys[a_mask]
        xs_b, ys_b = xs[b_mask], ys[b_mask]

        if len(xs_a) == 0 or len(xs_b) == 0:
            continue

        # Cross-distance between teams
        a_pos = np.column_stack([xs_a, ys_a])
        b_pos = np.column_stack([xs_b, ys_b])
        dists = cdist(a_pos, b_pos)

        counts_a = (dists <= radius).sum(axis=1)
        counts_b = (dists <= radius).sum(axis=0)

        # Store for writing back
        a_pids = pids[a_mask]
        a_tids = teams[a_mask]
        for i in range(len(a_pids)):
            proximity_values[(frame_id, int(a_pids[i]), int(a_tids[i]))] = int(counts_a[i])

        b_pids = pids[b_mask]
        for i in range(len(b_pids)):
            proximity_values[(frame_id, int(b_pids[i]), int(team_b))] = int(counts_b[i])

    # Write back to result DataFrame (vectorized)
    for (fid, pid, tid), cnt in proximity_values.items():
        mask = (
            (result[frame_col] == fid)
            & (result[player_col] == pid)
            & (result[team_col] == tid)
        )
        result.loc[mask, "opponents_nearby"] = cnt

    return result


def aggregate_opponent_proximity_to_blocks(
    blocks: list[pd.DataFrame],
    config: Optional[PressureConfig] = None,
) -> pd.DataFrame:
    """Aggregate opponent proximity per player per block.

    Parameters
    ----------
    blocks : list of pd.DataFrame
        Tracking data blocks with 'opponents_nearby' column.

    Returns
    -------
    pd.DataFrame
        One row per defender per block with:
        block_id, player_id, team_id_opta, n_frames_opp,
        opponents_nearby_mean, opponents_nearby_std
    """
    records = []
    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        # Only non-NaN, non-ball rows
        valid = blk[blk["player_id"] != -1].copy()
        if len(valid) == 0:
            continue

        for (pid, tid), grp in valid.groupby(["player_id", "team_id_opta"]):
            opp_vals = grp["opponents_nearby"].values
            if len(opp_vals) == 0:
                continue
            records.append({
                "block_id": bid,
                "player_id": int(pid),
                "team_id_opta": int(tid),
                "n_frames_opp": len(opp_vals),
                "opponents_nearby_mean": float(np.mean(opp_vals)),
                "opponents_nearby_std": float(np.std(opp_vals)),
            })

    return pd.DataFrame(records)
