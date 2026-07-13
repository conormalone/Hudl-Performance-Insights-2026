"""Defender reaction latency to shift triggers for Signal 2.

Measures how quickly defenders react to ball speed spikes and
aggressive opponent runs by computing the time between trigger
onset and when the defender begins moving goalward.

Follows the same approach as Signal 5 (transition latency) but
for different trigger types.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import ShiftLatencyConfig, DEFAULT_SHIFT_LATENCY_CONFIG

BALL_PLAYER_ID = -1


# ═══════════════════════════════════════════════════════════════════════════
# Pre-trigger Baseline
# ═══════════════════════════════════════════════════════════════════════════


def _pre_trigger_speed(
    df: pd.DataFrame,
    trigger_frame: int,
    player_id: int,
    smoothing_frames: int,
    frame_col: str = "frame_count",
    player_col: str = "player_id",
) -> float:
    """Get average speed before the trigger for a single player."""
    player_data = df[
        (df[player_col] == player_id) &
        (df[frame_col] >= trigger_frame - smoothing_frames) &
        (df[frame_col] < trigger_frame)
    ]

    if len(player_data) == 0:
        return 0.0

    speed = player_data["v_mag"].mean()
    return float(speed) if pd.notna(speed) else 0.0


def _pre_trigger_heading(
    df_player: pd.DataFrame,
    trigger_frame: int,
    smoothing_frames: int,
    heading_col: str = "heading",
    frame_col: str = "frame_count",
) -> float:
    """Compute the mean heading before the trigger."""
    before = df_player[
        (df_player[heading_col].notna()) &
        (df_player[frame_col] < trigger_frame)
    ]
    before_window = before.iloc[-smoothing_frames:]
    if len(before_window) < 2:
        return np.nan
    return float(before_window[heading_col].mean())


# ═══════════════════════════════════════════════════════════════════════════
# Per-Trigger Reaction Time
# ═══════════════════════════════════════════════════════════════════════════


def compute_shift_reaction_time(
    df: pd.DataFrame,
    trigger_df: pd.DataFrame,
    config: ShiftLatencyConfig,
    own_goal_direction: str = "left",
    frame_col: str = "frame_count",
    player_col: str = "player_id",
    team_col: str = "team_id_opta",
) -> pd.DataFrame:
    """Compute defender reaction latency for each trigger event.

    Measures the time between a trigger event (ball speed spike or
    opponent run) and each defender's first reaction frame. A reaction
    is defined as the first frame where the defender simultaneously:
    1. Moves above min speed
    2. Moves toward their own goal (vx towards own goal)
    3. Shows a heading change from their pre-trigger baseline

    Parameters
    ----------
    df : pd.DataFrame
        Full tracking data with smoothed velocities.
    trigger_df : pd.DataFrame
        Trigger events from triggers.detect_all_triggers.
        Must have columns: trigger_id, frame, phase, trigger_type.
    config : ShiftLatencyConfig
        Configuration.
    own_goal_direction : str
        'left' or 'right' in DOP-normalised coordinates.
    frame_col, player_col, team_col : str
        Column names.

    Returns
    -------
    pd.DataFrame
        One row per (trigger_id, player_id) with columns:
        - trigger_id, player_id, team_id_opta
        - reaction_time_s, pre_trigger_speed, post_trigger_speed
        - heading_change_deg, valid (bool)
        - trigger_type, trigger_frame
    """
    fps = config.frames_per_second
    window_frames = int(config.reaction_window_s * fps)
    min_speed = config.min_reaction_speed
    smoothing = config.direction_smoothing_frames
    reorient_thresh = config.reorientation_threshold_deg
    max_reaction_frames = int(config.max_reaction_time_s * fps)

    goalward_sign = -1.0  # vx < 0 = moving left = toward own goal
    if own_goal_direction == "right":
        goalward_sign = 1.0

    # Build frame → player data lookup for fast access
    frame_player_map: dict[int, dict[int, pd.Series]] = {}
    for _, row in df.iterrows():
        f = int(row[frame_col])
        if f not in frame_player_map:
            frame_player_map[f] = {}
        frame_player_map[f][int(row[player_col])] = row

    # Group player data by player_id
    player_groups = {
        pid: grp.sort_values(frame_col)
        for pid, grp in df.groupby(player_col)
    }

    records: list[dict] = []

    for _, trig_row in trigger_df.iterrows():
        tid = int(trig_row["trigger_id"])
        trig_frame = int(trig_row["frame"])
        trig_type = str(trig_row["trigger_type"])

        # Determine which team should react (team NOT in possession = defending)
        # Get the ball's team_in_possession at this frame
        ball_at_frame = frame_player_map.get(trig_frame, {}).get(BALL_PLAYER_ID)
        in_possession_team = None
        if ball_at_frame is not None:
            in_possession_team = ball_at_frame.get("team_in_possession", None)

        if pd.isna(in_possession_team):
            # Cannot determine — analyse all outfield players
            defending_teams = df[
                (df[player_col] != BALL_PLAYER_ID)
            ][team_col].unique()
        else:
            # The defending team = team NOT in possession
            all_teams = df[
                (df[player_col] != BALL_PLAYER_ID)
            ][team_col].unique()
            defending_teams = [t for t in all_teams if int(t) != int(in_possession_team)]

        for def_team in defending_teams:
            def_team = int(def_team)
            defending_players = df[
                (df[team_col] == def_team) &
                (df[player_col] != BALL_PLAYER_ID)
            ][player_col].unique()

            for pid in defending_players:
                pid = int(pid)

                # Pre-trigger speed
                pre_speed = _pre_trigger_speed(
                    df, trig_frame, pid, smoothing, frame_col, player_col
                )

                # Pre-trigger heading
                pid_df = player_groups.get(pid)
                if pid_df is None:
                    records.append(_empty_record(tid, pid, def_team, trig_type, trig_frame))
                    continue

                pre_heading = _pre_trigger_heading(
                    pid_df, trig_frame, smoothing, frame_col=frame_col
                )
                if pd.isna(pre_heading):
                    records.append(_empty_record(tid, pid, def_team, trig_type, trig_frame))
                    continue

                # Scan forward for reaction
                end_frame = min(
                    trig_frame + window_frames,
                    trig_frame + max_reaction_frames,
                    int(pid_df[frame_col].max())
                )

                reaction_frame: Optional[int] = None
                reaction_speed: float = 0.0
                reaction_heading: float = 0.0

                for f in range(trig_frame + 1, end_frame + 1):
                    player_at_frame = frame_player_map.get(f, {}).get(pid)
                    if player_at_frame is None:
                        continue

                    vmag = player_at_frame.get("v_mag", 0.0)
                    if pd.isna(vmag) or vmag < min_speed:
                        continue

                    vx = player_at_frame.get(
                        "vx_smooth",
                        player_at_frame.get("vx", 0.0)
                    )
                    if pd.isna(vx):
                        continue

                    # Check goalward movement
                    if vx * goalward_sign >= 0:
                        continue

                    # Check heading change
                    heading = player_at_frame.get("heading", np.nan)
                    if pd.isna(heading):
                        continue

                    # Radians → degrees for threshold comparison
                    heading_change_rad = abs(heading - pre_heading)
                    heading_change_deg = float(np.degrees(heading_change_rad)) % 360
                    if heading_change_deg > 180:
                        heading_change_deg = 360 - heading_change_deg

                    if heading_change_deg < reorient_thresh:
                        continue

                    # All conditions met!
                    reaction_frame = f
                    reaction_speed = float(vmag)
                    reaction_heading = float(heading)
                    break

                if reaction_frame is not None:
                    reaction_time = (reaction_frame - trig_frame) / fps
                    heading_change = abs(reaction_heading - pre_heading)
                    heading_change = float(np.degrees(heading_change)) % 360
                    if heading_change > 180:
                        heading_change = 360 - heading_change

                    records.append({
                        "trigger_id": tid,
                        "player_id": pid,
                        "team_id_opta": def_team,
                        "reaction_time_s": round(reaction_time, 3),
                        "pre_trigger_speed": round(pre_speed, 3),
                        "post_trigger_speed": round(reaction_speed, 3),
                        "heading_change_deg": round(heading_change, 1),
                        "valid": True,
                        "trigger_type": trig_type,
                        "trigger_frame": trig_frame,
                    })
                else:
                    records.append(_empty_record(tid, pid, def_team, trig_type, trig_frame))

    columns = [
        "trigger_id", "player_id", "team_id_opta",
        "reaction_time_s", "pre_trigger_speed",
        "post_trigger_speed", "heading_change_deg",
        "valid", "trigger_type", "trigger_frame",
    ]

    if not records:
        return pd.DataFrame({c: pd.Series(dtype="float64" if c in (
            "reaction_time_s", "pre_trigger_speed", "post_trigger_speed",
            "heading_change_deg",
        ) else "int64" if c in ("trigger_id", "player_id", "team_id_opta",
                                "trigger_frame") else "bool" if c == "valid"
        else "object") for c in columns})

    return pd.DataFrame(records, columns=columns)


def _empty_record(
    tid: int,
    player_id: int,
    team_id: int,
    trigger_type: str,
    trigger_frame: int,
) -> dict:
    """Build an invalid record for a player with no reaction."""
    return {
        "trigger_id": tid,
        "player_id": player_id,
        "team_id_opta": team_id,
        "reaction_time_s": np.nan,
        "pre_trigger_speed": 0.0,
        "post_trigger_speed": 0.0,
        "heading_change_deg": 0.0,
        "valid": False,
        "trigger_type": trigger_type,
        "trigger_frame": trigger_frame,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Block Aggregation
# ═══════════════════════════════════════════════════════════════════════════


def aggregate_shift_latency_by_block(
    latency_df: pd.DataFrame,
    blocks: list[dict],
    config: ShiftLatencyConfig,
    game_id: str = "",
) -> pd.DataFrame:
    """Aggregate shift reaction latencies into per-block, per-player values.

    For each block and each player:
    - mean_reaction_time: mean of valid reaction times
    - p90_reaction_time: 90th percentile
    - n_triggers: number of trigger events in this block
    - trigger_types: breakdown by type

    Parameters
    ----------
    latency_df : pd.DataFrame
        Output from compute_shift_reaction_time.
    blocks : list of dict
        Block definitions with block_id, phase, start_frame, end_frame.
    config : ShiftLatencyConfig
        Configuration.
    game_id : str
        Match identifier.

    Returns
    -------
    pd.DataFrame
        Standardised signal output with columns:
        game_id, block_id, phase, player_id, team_id_opta,
        signal_name ('shift_latency'), signal_value, n_frames.
    """
    # Build frame → block lookup
    frame_to_block: dict[int, tuple[str, int]] = {}
    for blk in blocks:
        bid = blk["block_id"]
        phase = blk["phase"]
        for f in range(blk["start_frame"], blk["end_frame"] + 1):
            frame_to_block[f] = (bid, phase)

    # Map each reaction to a block via trigger_frame
    valid = latency_df[latency_df["valid"]].copy()
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

    valid["block_id"] = valid["trigger_frame"].map(
        lambda f: frame_to_block.get(int(f), (None, None))[0]
    )
    valid["phase"] = valid["trigger_frame"].map(
        lambda f: frame_to_block.get(int(f), (None, None))[1]
    )

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
            n_triggers=("trigger_id", "nunique"),
        )
        .reset_index()
    )

    output_records = []
    for _, row in agg.iterrows():
        block_frames = 0
        for blk in blocks:
            if blk["block_id"] == row["block_id"]:
                block_frames = blk.get("end_frame", 0) - blk.get("start_frame", 0)
                break

        output_records.append({
            "game_id": game_id,
            "block_id": row["block_id"],
            "phase": int(row["phase"]),
            "player_id": int(row["player_id"]),
            "team_id_opta": int(row["team_id_opta"]),
            "signal_name": "shift_latency",
            "signal_value": round(float(row["mean_reaction_time"]), 3),
            "n_frames": block_frames,
        })

    return pd.DataFrame(output_records)
