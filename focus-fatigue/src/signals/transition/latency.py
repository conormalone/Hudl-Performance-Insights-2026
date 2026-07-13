"""Defender reaction latency computation.

For each possession transition, measure how quickly every defender
recognises and reacts by computing the time between the transition
event and when they begin moving purposefully toward their own goal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import TransitionConfig, DEFAULT_TRANSITION_CONFIG

# ═══════════════════════════════════════════════════════════════════════════
# Per-Transition Reaction Time
# ═══════════════════════════════════════════════════════════════════════════


def _pre_transition_heading(
    df_player: pd.DataFrame,
    trans_frame: int,
    smoothing_frames: int,
    heading_col: str = "heading",
    frame_col: str = "frame_count",
) -> float:
    """Compute the mean heading before the transition for a single player.

    Parameters
    ----------
    df_player : pd.DataFrame
        Player data sorted by frame.
    trans_frame : int
        The transition frame number.
    smoothing_frames : int
        Number of frames before transition to average over.
    heading_col : str
        Column containing heading angles (degrees).
    frame_col : str
        Column containing frame numbers.

    Returns
    -------
    float
        Mean heading angle before transition, or NaN if insufficient data.
    """
    # Only look at frames before the transition
    before_transition = df_player[
        (df_player[heading_col].notna()) &
        (df_player[frame_col] < trans_frame)
    ]
    before = before_transition.iloc[-smoothing_frames:]
    if len(before) < 2:
        return np.nan
    return float(before[heading_col].mean())


def compute_reaction_time(
    df: pd.DataFrame,
    trans_df: pd.DataFrame,
    config: TransitionConfig,
    own_goal_direction: str = "left",
    frame_col: str = "frame_count",
    player_col: str = "player_id",
    team_col: str = "team_id_opta",
) -> pd.DataFrame:
    """Compute reaction latency for each defender on each transition.

    **Definition of reaction**: The first frame after a possession flip
    where the defender simultaneously satisfies:

    1. **Speed condition**: ``v_mag > min_reaction_speed`` (m/s)
    2. **Goalward movement**: Their smoothed velocity x-component points
       toward their own goal (``vx_smooth < 0`` when ``own_goal_direction='left'``)
    3. **Reorientation**: Their heading has changed by more than
       ``reorientation_threshold_deg`` from their pre-transition heading.

    If no frame satisfies all three within ``reaction_window_s``, the
    reaction is marked as invalid (``valid = False``).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``frame_count``, ``player_id``, ``team_id_opta``,
        ``vx_smooth``, ``vy_smooth``, ``v_mag``, ``heading``, and positional
        coordinates.
    trans_df : pd.DataFrame
        Transition events from :func:`detect_transitions` (must have
        ``transition_id`` and ``frame`` columns).
    config : TransitionConfig
        Configuration with reaction window, speed threshold, smoothing,
        and reorientation parameters.
    own_goal_direction : str
        Direction of own goal in DOP-normalised coordinates.
        ``'left'`` means the own goal is at negative x.
    frame_col : str
        Column name for frame count.
    player_col : str
        Column name for player ID.
    team_col : str
        Column name for team ID.

    Returns
    -------
    pd.DataFrame
        One row per (transition_id, player_id) combination with columns:

        - transition_id : int
        - player_id : int
        - team_id_opta : int
        - reaction_time_s : float — seconds from transition to detected
          reaction (NaN if invalid)
        - pre_transition_speed : float — speed (m/s) just before transition
        - post_transition_speed : float — speed (m/s) at reaction time
        - heading_change_deg : float — absolute heading change from
          pre-transition average
        - valid : bool — whether a reaction was detected within the window
    """
    fps = config.frames_per_second
    window_frames = int(config.reaction_window_s * fps)
    min_speed = config.min_reaction_speed
    smoothing = config.direction_smoothing_frames
    reorient_thresh = config.reorientation_threshold_deg
    max_reaction_frames = int(config.max_reaction_time_s * fps)

    # Determine goalward direction sign
    goalward_sign = -1.0  # vx < 0 means moving left (toward own goal)
    if own_goal_direction == "right":
        goalward_sign = 1.0

    records: list[dict] = []

    # Group player data by player_id for fast lookup
    player_groups = {pid: grp.sort_values(frame_col)
                     for pid, grp in df.groupby(player_col)}

    # Build a frame → player data lookup to avoid full scans
    # frame → {player_id: row}
    frame_player_map: dict[int, dict[int, pd.Series]] = {}
    for _, row in df.iterrows():
        f = int(row[frame_col])
        if f not in frame_player_map:
            frame_player_map[f] = {}
        frame_player_map[f][int(row[player_col])] = row

    for _, trans_row in trans_df.iterrows():
        tid = int(trans_row["transition_id"])
        trans_frame = int(trans_row["frame"])

        # Identify which players to analyse: those on the gaining team
        # (they just lost possession and need to react)
        gaining_team = int(trans_row["gaining_team"])

        # Get all player IDs on the gaining team
        gaining_players = df[
            (df[team_col] == gaining_team) &
            (df[player_col] != BALL_PLAYER_ID)
        ][player_col].unique()

        for pid in gaining_players:
            pid = int(pid)

            # -- Pre-transition speed --
            pre_speed = _pre_transition_speed(
                df, trans_frame, pid, smoothing, frame_col, player_col
            )

            # -- Pre-transition heading --
            pid_df = player_groups.get(pid)
            if pid_df is None:
                records.append(_empty_record(tid, pid, gaining_team))
                continue

            pre_heading = _pre_transition_heading(
                pid_df, trans_frame, smoothing, frame_col=frame_col
            )
            if pd.isna(pre_heading):
                records.append(_empty_record(tid, pid, gaining_team))
                continue

            # -- Scan forward for reaction --
            end_frame = min(
                trans_frame + window_frames,
                trans_frame + max_reaction_frames,
                int(pid_df[frame_col].max())
            )

            reaction_frame: Optional[int] = None
            reaction_speed: float = 0.0
            reaction_heading: float = 0.0

            for f in range(trans_frame + 1, end_frame + 1):
                player_at_frame = frame_player_map.get(f, {}).get(pid)
                if player_at_frame is None:
                    continue

                vmag = player_at_frame.get("v_mag", 0.0)
                if pd.isna(vmag) or vmag < min_speed:
                    continue

                vx = player_at_frame.get("vx_smooth", player_at_frame.get("vx", 0.0))
                if pd.isna(vx):
                    continue

                # Check goalward movement
                if vx * goalward_sign >= 0:
                    continue

                # Check reorientation
                heading = player_at_frame.get("heading", np.nan)
                if pd.isna(heading):
                    continue

                heading_change_rad = abs(heading - pre_heading)
                # heading is in radians; convert to degrees to compare against threshold
                heading_change = float(np.degrees(heading_change_rad))
                # Normalise to [0, 180]
                heading_change = heading_change % 360
                if heading_change > 180:
                    heading_change = 360 - heading_change

                if heading_change < reorient_thresh:
                    continue

                # All conditions met — this is the reaction frame
                reaction_frame = f
                reaction_speed = float(vmag)
                reaction_heading = float(heading)
                break

            if reaction_frame is not None:
                reaction_time = (reaction_frame - trans_frame) / fps
                heading_change = abs(reaction_heading - pre_heading)
                heading_change = heading_change % 360
                if heading_change > 180:
                    heading_change = 360 - heading_change

                records.append({
                    "transition_id": tid,
                    "player_id": pid,
                    "team_id_opta": gaining_team,
                    "reaction_time_s": round(reaction_time, 3),
                    "pre_transition_speed": round(pre_speed, 3),
                    "post_transition_speed": round(reaction_speed, 3),
                    "heading_change_deg": round(heading_change, 1),
                    "valid": True,
                })
            else:
                records.append({
                    "transition_id": tid,
                    "player_id": pid,
                    "team_id_opta": gaining_team,
                    "reaction_time_s": np.nan,
                    "pre_transition_speed": round(pre_speed, 3),
                    "post_transition_speed": 0.0,
                    "heading_change_deg": 0.0,
                    "valid": False,
                })

    columns = [
        "transition_id", "player_id", "team_id_opta",
        "reaction_time_s", "pre_transition_speed",
        "post_transition_speed", "heading_change_deg", "valid",
    ]
    if not records:
        return pd.DataFrame({
            "transition_id": pd.Series(dtype="int"),
            "player_id": pd.Series(dtype="int"),
            "team_id_opta": pd.Series(dtype="int"),
            "reaction_time_s": pd.Series(dtype="float"),
            "pre_transition_speed": pd.Series(dtype="float"),
            "post_transition_speed": pd.Series(dtype="float"),
            "heading_change_deg": pd.Series(dtype="float"),
            "valid": pd.Series(dtype="bool"),
        })

    return pd.DataFrame(records, columns=columns)


def _pre_transition_speed(
    df: pd.DataFrame,
    trans_frame: int,
    player_id: int,
    smoothing_frames: int,
    frame_col: str = "frame_count",
    player_col: str = "player_id",
) -> float:
    """Get the average speed of a player just before the transition.

    Parameters
    ----------
    df : pd.DataFrame
        Full tracking data.
    trans_frame : int
        Transition frame number.
    player_id : int
        Player ID.
    smoothing_frames : int
        Frames to average over.
    frame_col : str
        Frame column name.
    player_col : str
        Player column name.

    Returns
    -------
    float
        Mean speed over the smoothing window, or 0.0 if insufficient data.
    """
    player_data = df[
        (df[player_col] == player_id) &
        (df[frame_col] >= trans_frame - smoothing_frames) &
        (df[frame_col] < trans_frame)
    ]

    if len(player_data) == 0:
        return 0.0

    speed = player_data["v_mag"].mean()
    return float(speed) if pd.notna(speed) else 0.0


def _empty_record(
    tid: int,
    player_id: int,
    team_id: int,
) -> dict:
    """Build an empty/invalid record for a player with no reaction data."""
    return {
        "transition_id": tid,
        "player_id": player_id,
        "team_id_opta": team_id,
        "reaction_time_s": np.nan,
        "pre_transition_speed": 0.0,
        "post_transition_speed": 0.0,
        "heading_change_deg": 0.0,
        "valid": False,
    }


BALL_PLAYER_ID = -1


# ═══════════════════════════════════════════════════════════════════════════
# Per-Block Aggregation
# ═══════════════════════════════════════════════════════════════════════════


def aggregate_latency_by_block(
    latency_df: pd.DataFrame,
    blocks: list[pd.DataFrame],
    config: TransitionConfig,
    game_id: str = "",
) -> pd.DataFrame:
    """Aggregate reaction times into per-block, per-player signal values.

    For each block and each player, the function computes:

    - ``mean_reaction_time`` — mean of valid reaction times
    - ``p90_reaction_time`` — 90th percentile of valid reaction times
    - ``n_transitions`` — number of transitions in this block for this
      player
    - ``transition_types`` — breakdown (counts) of expected vs surprise
      transitions

    These are mapped to the standard signal output schema columns.

    Parameters
    ----------
    latency_df : pd.DataFrame
        Output from :func:`compute_reaction_time`. Must contain columns
        ``transition_id``, ``player_id``, ``team_id_opta``,
        ``reaction_time_s``, ``valid``.
    blocks : list of pd.DataFrame
        Block-segmented tracking data from :func:`split_into_blocks`.
        Each DataFrame has a ``block_id`` column and ``frame_count`` column.
    config : TransitionConfig
        Configuration object (used for frame rate consistency).
    game_id : str
        Match identifier.

    Returns
    -------
    pd.DataFrame
        Standardised output with columns:
        ``game_id``, ``block_id``, ``phase``, ``player_id``, ``team_id_opta``,
        ``signal_name`` (``'transition_latency'``), ``signal_value`` (mean
        reaction time), ``n_frames``.
    """
    if len(latency_df) == 0 or not latency_df["valid"].any():
        # Return empty DataFrame with correct schema and dtypes
        empty = pd.DataFrame({
            "game_id": pd.Series(dtype="str"),
            "block_id": pd.Series(dtype="str"),
            "phase": pd.Series(dtype="int"),
            "player_id": pd.Series(dtype="int"),
            "team_id_opta": pd.Series(dtype="int"),
            "signal_name": pd.Series(dtype="str"),
            "signal_value": pd.Series(dtype="float"),
            "n_frames": pd.Series(dtype="int"),
        })
        return empty

    # Build a frame → block_id mapping from DataFrame blocks
    frame_to_block: dict[int, tuple[str, int]] = {}
    for blk in blocks:
        bid = str(blk["block_id"].iloc[0])
        phase = int(blk["block_id"].iloc[0].split("_")[0])
        start_frame = int(blk["frame_count"].min())
        end_frame = int(blk["frame_count"].max())
        for f in range(start_frame, end_frame + 1):
            frame_to_block[f] = (bid, phase)

    # We need to map each transition to a block.
    # latency_df has transition_id; we need the transitions' frames.
    # Since latency_df doesn't carry the transition frame, we assume
    # the caller provides a merged version or we infer from the data.
    # For now, we require that latency_df has a 'frame' column or
    # 'transition_frame' column. If not, we return empty.

    # Add a transition_frame if we can infer it — caller responsibility
    # to include it. Here we check and gracefully handle absence.
    tframe_col = None
    for candidate in ("frame", "transition_frame"):
        if candidate in latency_df.columns:
            tframe_col = candidate
            break

    if tframe_col is None:
        # Fallback: cannot map to blocks. Return empty result warning.
        return pd.DataFrame(
            columns=[
                "game_id", "block_id", "phase", "player_id",
                "team_id_opta", "signal_name", "signal_value", "n_frames",
            ]
        )

    # Merge block info
    valid = latency_df[latency_df["valid"]].copy()
    valid["block_id"] = valid[tframe_col].map(
        lambda f: frame_to_block.get(int(f), (None, None))[0]
    )
    valid["phase"] = valid[tframe_col].map(
        lambda f: frame_to_block.get(int(f), (None, None))[1]
    )

    # Drop rows that fell outside any block
    valid = valid.dropna(subset=["block_id"])

    if len(valid) == 0:
        return pd.DataFrame({
            "game_id": pd.Series(dtype="str"),
            "block_id": pd.Series(dtype="str"),
            "phase": pd.Series(dtype="int"),
            "player_id": pd.Series(dtype="int"),
            "team_id_opta": pd.Series(dtype="int"),
            "signal_name": pd.Series(dtype="str"),
            "signal_value": pd.Series(dtype="float"),
            "n_frames": pd.Series(dtype="int"),
        })

    # Aggregate per player per block
    agg = (
        valid.groupby(["block_id", "phase", "player_id", "team_id_opta"])
        .agg(
            mean_reaction_time=("reaction_time_s", "mean"),
            p90_reaction_time=("reaction_time_s", lambda x: x.quantile(0.90)),
            n_transitions=("transition_id", "nunique"),
        )
        .reset_index()
    )

    # Build standard output
    output_records = []
    for _, row in agg.iterrows():
        # Use the block's frame count for n_frames
        block_frames = 0
        for blk in blocks:
            if str(blk["block_id"].iloc[0]) == row["block_id"]:
                block_frames = len(blk)
                break

        output_records.append({
            "game_id": game_id,
            "block_id": row["block_id"],
            "phase": int(row["phase"]),
            "player_id": int(row["player_id"]),
            "team_id_opta": int(row["team_id_opta"]),
            "signal_name": "transition_latency",
            "signal_value": round(float(row["mean_reaction_time"]), 3),
            "n_frames": block_frames,
        })

    return pd.DataFrame(output_records)
