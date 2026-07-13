"""Transition Count — Load Indicator 4.

Counts team_in_possession changes (flips) that occur within each
defender's zone (within a configurable radius of their position).

Since we don't have event data, we use the tracking data's
`team_in_possession` column. A transition occurs when this value
changes from one non-NaN team ID to another (e.g., 147 → 862).
"""

from typing import Optional

import numpy as np
import pandas as pd

from .config import PressureConfig, DEFAULT_CONFIG


BALL_PLAYER_ID = -1


def detect_transition_frames(
    df: pd.DataFrame,
    team_in_possession_col: str = "team_in_possession",
    frame_col: str = "frame_count",
    phase_col: str = "phase",
) -> pd.DataFrame:
    """Detect frames where a possession transition occurs.

    A transition = team_in_possession changes between non-NaN values.
    Returns a DataFrame with one row per transition frame.

    Parameters
    ----------
    df : pd.DataFrame
        Tracking data for a single match (can be the full dataset).
    team_in_possession_col : str
    frame_col : str
    phase_col : str

    Returns
    -------
    pd.DataFrame
        Columns: frame (frame_count), phase, prev_team, new_team,
                 ball_x, ball_y
        One row per transition frame.
    """
    # Get ball-level data: one row per frame (ball or any row)
    # First, get team_in_possession per frame from the ball row
    ball_df = df[df["player_id"] == BALL_PLAYER_ID].copy()

    if len(ball_df) == 0:
        # Fallback: use first row per frame
        ball_df = df.groupby([phase_col, frame_col]).first().reset_index()

    ball_df = ball_df.sort_values([phase_col, frame_col])

    transitions = []

    for phase in sorted(ball_df[phase_col].unique()):
        phase_ball = ball_df[ball_df[phase_col] == phase].copy()
        tip = phase_ball[team_in_possession_col].values
        frames = phase_ball[frame_col].values

        # Find flips: non-NaN -> different non-NaN
        prev_valid = None
        prev_frame = None
        prev_phase = None

        for i in range(len(tip)):
            current_val = tip[i]
            if pd.notna(current_val):
                if prev_valid is not None and current_val != prev_valid:
                    # Transition detected!
                    # Get ball position at this frame
                    ball_row = phase_ball.iloc[i]
                    transitions.append({
                        "frame": int(frames[i]),
                        "phase": int(phase),
                        "prev_team": int(prev_valid),
                        "new_team": int(current_val),
                        "ball_x": float(ball_row.get("x", np.nan)),
                        "ball_y": float(ball_row.get("y", np.nan)),
                    })
                prev_valid = int(current_val)
                prev_frame = int(frames[i])

    if not transitions:
        return pd.DataFrame(
            columns=["frame", "phase", "prev_team", "new_team", "ball_x", "ball_y"]
        )

    return pd.DataFrame(transitions)


def count_zone_transitions(
    df: pd.DataFrame,
    transition_frames: pd.DataFrame,
    radius: float = 7.0,
    config: Optional[PressureConfig] = None,
    player_col: str = "player_id",
    team_col: str = "team_id_opta",
    frame_col: str = "frame_count",
    pos_cols: tuple[str, str] = ("x", "y"),
) -> pd.DataFrame:
    """Count how many transitions occurred in each defender's zone.

    A transition is "in the zone" of a defender if the ball was within
    `radius` metres of that defender at the transition frame.

    Parameters
    ----------
    df : pd.DataFrame
        Full tracking data for a match.
    transition_frames : pd.DataFrame
        Output from detect_transition_frames.
    radius : float, default 7.0
        Zone radius in metres.
    config : PressureConfig, optional
    player_col : str, default 'player_id'
    team_col : str, default 'team_id_opta'
    frame_col : str, default 'frame_count'
    pos_cols : tuple, default ('x', 'y')

    Returns
    -------
    pd.DataFrame
        One row per player per block with:
        block_id, player_id, team_id_opta, transition_count
        Only includes players who had at least one zone transition
        (players with zero transitions are implicit).
    """
    if config is not None:
        radius = config.transition_zone_radius

    if len(transition_frames) == 0:
        return pd.DataFrame(
            columns=["block_id", "player_id", "team_id_opta", "transition_count"]
        )

    col_x, col_y = pos_cols
    records = []

    for _, trans_row in transition_frames.iterrows():
        t_frame = trans_row["frame"]
        ball_x = trans_row["ball_x"]
        ball_y = trans_row["ball_y"]

        if pd.isna(ball_x) or pd.isna(ball_y):
            continue

        # Get player positions at this frame
        frame_players = df[
            (df[frame_col] == t_frame) & (df[player_col] != BALL_PLAYER_ID)
        ]

        for _, prow in frame_players.iterrows():
            px = prow[col_x]
            py = prow[col_y]
            dist = np.sqrt((px - ball_x) ** 2 + (py - ball_y) ** 2)

            if dist <= radius:
                records.append({
                    "frame": t_frame,
                    "player_id": int(prow[player_col]),
                    "team_id_opta": int(prow[team_col]),
                    "ball_dist": float(dist),
                })

    return pd.DataFrame(records)


def aggregate_transitions_to_blocks(
    blocks: list[pd.DataFrame],
    transition_records: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate zone transition counts per player per block.

    Parameters
    ----------
    blocks : list of pd.DataFrame
        Block-segmented tracking data.
    transition_records : pd.DataFrame
        Output from count_zone_transitions (or empty DataFrame).

    Returns
    -------
    pd.DataFrame
        One row per player per block with:
        block_id, player_id, team_id_opta, transition_count
    """
    if len(transition_records) == 0:
        # Return empty result with correct columns
        blocks_with_ids = [(blk["block_id"].iloc[0], blk) for blk in blocks]
        return pd.DataFrame(
            columns=["block_id", "player_id", "team_id_opta", "transition_count"]
        )

    # Map frames to blocks
    frame_to_block = {}
    for blk in blocks:
        bid = blk["block_id"].iloc[0]
        for f in blk["frame_count"].unique():
            frame_to_block[int(f)] = bid

    # Group by player and block
    agg = {}
    for _, row in transition_records.iterrows():
        f = int(row["frame"])
        bid = frame_to_block.get(f)
        if bid is None:
            continue
        key = (bid, int(row["player_id"]), int(row["team_id_opta"]))
        agg[key] = agg.get(key, 0) + 1

    records = [
        {"block_id": k[0], "player_id": k[1], "team_id_opta": k[2], "transition_count": v}
        for k, v in agg.items()
    ]

    return pd.DataFrame(records)
