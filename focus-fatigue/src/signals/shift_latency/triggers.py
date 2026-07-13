"""Trigger detection for Signal 2 — Shift Latency.

Detects two types of triggers that should elicit a defensive reaction:
1. Ball speed spikes (sudden fast pass/long ball)
2. Opponent runs (attacker accelerating toward goal)

Each trigger event represents a moment when defenders should react
(shift their position/orientation toward own goal).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import ShiftLatencyConfig, DEFAULT_SHIFT_LATENCY_CONFIG

# ── Constants ───────────────────────────────────────────────────────────────

BALL_PLAYER_ID = -1
"""Ball marker in tracking data."""


# ═══════════════════════════════════════════════════════════════════════════
# Ball Speed Spike Detection
# ═══════════════════════════════════════════════════════════════════════════


def detect_ball_speed_spikes(
    df: pd.DataFrame,
    config: ShiftLatencyConfig,
    frame_col: str = "frame_count",
    vx_col: str = "vx_smooth",
    vy_col: str = "vy_smooth",
) -> pd.DataFrame:
    """Detect frames where ball speed spikes above threshold.

    A ball speed spike occurs when the instantaneous ball speed
    exceeds ``ball_speed_spike_threshold`` m/s. Nearby spikes (within
    ``min_spike_gap_frames``) are grouped into single events.

    Parameters
    ----------
    df : pd.DataFrame
        Full tracking data with smoothed ball velocity columns.
    config : ShiftLatencyConfig
        Configuration with spike threshold and grouping parameters.
    frame_col : str
        Column name for frame count.
    vx_col, vy_col : str
        Velocity column names.

    Returns
    -------
    pd.DataFrame
        One row per spike event with columns:
            - spike_id: int — zero-based unique id
            - frame: int — frame of the spike onset
            - phase: int — match phase
            - peak_speed: float — ball speed at peak (m/s)
            - ball_x, ball_y: float — ball position at peak
            - trigger_type: str — always 'ball_speed_spike'
    """
    threshold = config.ball_speed_spike_threshold
    min_gap = config.min_spike_gap_frames
    smoothing = config.ball_speed_smoothing_frames

    # Get ball data only
    ball_df = df[df["player_id"] == BALL_PLAYER_ID].copy()

    if len(ball_df) == 0:
        return pd.DataFrame(
            columns=["spike_id", "frame", "phase", "peak_speed",
                     "ball_x", "ball_y", "trigger_type"]
        )

    ball_df = ball_df.sort_values(frame_col)

    # Compute ball speed from smooth velocity components
    if vx_col in ball_df.columns and vy_col in ball_df.columns:
        vx = ball_df[vx_col].values.astype(np.float64)
        vy = ball_df[vy_col].values.astype(np.float64)
        ball_speed = np.sqrt(vx ** 2 + vy ** 2)
    else:
        vx = ball_df.get("vx", ball_df.get("speed_x", np.full(len(ball_df), np.nan))).values
        vy = ball_df.get("vy", ball_df.get("speed_y", np.full(len(ball_df), np.nan))).values
        ball_speed = np.sqrt(vx.astype(np.float64) ** 2 + vy.astype(np.float64) ** 2)

    # Smooth with rolling average
    ball_speed_smooth = pd.Series(ball_speed).rolling(
        window=smoothing, center=True, min_periods=1
    ).mean().values

    # Detect spikes
    is_spike = ball_speed_smooth >= threshold

    # Group consecutive spike frames
    spike_groups: list[list[int]] = []
    current_group: list[int] = []

    for i in range(len(is_spike)):
        if is_spike[i]:
            current_group.append(i)
        else:
            if current_group:
                spike_groups.append(current_group)
                current_group = []

    if current_group:
        spike_groups.append(current_group)

    # Merge nearby groups (within min_gap_frames)
    if spike_groups:
        merged_groups: list[list[int]] = [spike_groups[0]]
        for grp in spike_groups[1:]:
            prev_last_idx = merged_groups[-1][-1]
            this_first_idx = grp[0]
            frames_gap = (ball_df.iloc[this_first_idx][frame_col] -
                          ball_df.iloc[prev_last_idx][frame_col])
            if frames_gap <= min_gap:
                merged_groups[-1].extend(grp)
            else:
                merged_groups.append(grp)
    else:
        merged_groups = []

    # Build output
    records = []
    for sid, grp_indices in enumerate(merged_groups):
        # Find the peak within the group
        group_speeds = ball_speed_smooth[grp_indices]
        peak_idx = grp_indices[np.argmax(group_speeds)]
        peak_row = ball_df.iloc[peak_idx]

        records.append({
            "spike_id": sid,
            "frame": int(peak_row[frame_col]),
            "phase": int(peak_row.get("phase", 1)),
            "peak_speed": float(group_speeds.max()),
            "ball_x": float(peak_row.get("x", np.nan)),
            "ball_y": float(peak_row.get("y", np.nan)),
            "trigger_type": "ball_speed_spike",
        })

    if not records:
        return pd.DataFrame(
            columns=["spike_id", "frame", "phase", "peak_speed",
                     "ball_x", "ball_y", "trigger_type"]
        )

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# Opponent Run Detection
# ═══════════════════════════════════════════════════════════════════════════


def detect_opponent_runs(
    df: pd.DataFrame,
    config: ShiftLatencyConfig,
    frame_col: str = "frame_count",
    vx_col: str = "vx_smooth",
    vy_col: str = "vy_smooth",
    player_col: str = "player_id",
    team_col: str = "team_id_opta",
) -> pd.DataFrame:
    """Detect aggressive opponent runs (attacker sprints toward goal).

    An aggressive run is defined as an attacker (any outfield player
    on the team currently in possession) moving at speed > threshold
    toward the defending team's goal.

    Parameters
    ----------
    df : pd.DataFrame
        Full tracking data. Must have possession indicator.
    config : ShiftLatencyConfig
        Configuration with speed threshold.
    frame_col, vx_col, vy_col, player_col, team_col : str
        Column names.
    team_in_possession_col : str
        Which team has the ball.

    Returns
    -------
    pd.DataFrame
        One row per run event with columns:
            - run_id: int — zero-based unique id
            - frame: int — peak frame
            - phase: int
            - attacker_id: int
            - attacker_team: int
            - atk_speed: float — peak run speed (m/s)
            - atk_x, atk_y: float — position
            - trigger_type: str — always 'opponent_run'
    """
    speed_threshold = config.opponent_run_speed_threshold

    if "team_in_possession" not in df.columns:
        return pd.DataFrame(
            columns=["run_id", "frame", "phase", "attacker_id",
                     "attacker_team", "atk_speed", "atk_x", "atk_y",
                     "trigger_type"]
        )

    # Get all outfield players who are on the team in possession
    # and moving fast toward the opponent's goal
    outfield = df[
        (df[player_col] != BALL_PLAYER_ID) &
        (df["team_in_possession"].notna()) &
        (df[team_col] == df["team_in_possession"])
    ].copy()

    if len(outfield) == 0:
        return pd.DataFrame(
            columns=["run_id", "frame", "phase", "attacker_id",
                     "attacker_team", "atk_speed", "atk_x", "atk_y",
                     "trigger_type"]
        )

    # Compute speed
    vx = outfield.get(vx_col, outfield.get("vx", 0)).values.astype(np.float64)
    vy = outfield.get(vy_col, outfield.get("vy", 0)).values.astype(np.float64)
    outfield["atk_speed"] = np.sqrt(vx ** 2 + vy ** 2)

    # Filter: moving fast (sprinting)
    sprinting = outfield[outfield["atk_speed"] >= speed_threshold].copy()

    if len(sprinting) == 0:
        return pd.DataFrame(
            columns=["run_id", "frame", "phase", "attacker_id",
                     "attacker_team", "atk_speed", "atk_x", "atk_y",
                     "trigger_type"]
        )

    # For each attacker, group consecutive sprint frames
    records = []
    for (pid, team), grp in sprinting.groupby([player_col, team_col]):
        grp = grp.sort_values(frame_col)
        sprint_groups: list[list[int]] = []
        current: list[int] = []

        for i in range(len(grp)):
            if current:
                gap = grp.iloc[i][frame_col] - grp.iloc[current[-1]][frame_col]
                if gap > 5:  # More than 5 frame gap = new sprint
                    sprint_groups.append(current)
                    current = [i]
                else:
                    current.append(i)
            else:
                current = [i]

        if current:
            sprint_groups.append(current)

        for run_id_offset, grp_indices in enumerate(sprint_groups):
            run_speeds = grp.iloc[grp_indices]["atk_speed"].values
            peak_idx_grp = grp_indices[np.argmax(run_speeds)]
            peak_row = grp.iloc[peak_idx_grp]

            records.append({
                "run_id": len(records),
                "frame": int(peak_row[frame_col]),
                "phase": int(peak_row.get("phase", 1)),
                "attacker_id": int(pid),
                "attacker_team": int(team),
                "atk_speed": float(run_speeds.max()),
                "atk_x": float(peak_row.get("x", np.nan)),
                "atk_y": float(peak_row.get("y", np.nan)),
                "trigger_type": "opponent_run",
            })

    if not records:
        return pd.DataFrame(
            columns=["run_id", "frame", "phase", "attacker_id",
                     "attacker_team", "atk_speed", "atk_x", "atk_y",
                     "trigger_type"]
        )

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# Combined Trigger Detection
# ═══════════════════════════════════════════════════════════════════════════


def detect_all_triggers(
    df: pd.DataFrame,
    config: ShiftLatencyConfig,
) -> pd.DataFrame:
    """Detect all shift triggers (ball speed spikes + opponent runs).

    Returns a combined DataFrame of trigger events sorted by frame,
    with a unified ``trigger_id``.

    Parameters
    ----------
    df : pd.DataFrame
        Full tracking data.
    config : ShiftLatencyConfig
        Configuration.

    Returns
    -------
    pd.DataFrame
        All trigger events with a unified ``trigger_id``.
        Columns: trigger_id, frame, phase, trigger_type,
        and type-specific columns.
    """
    spikes = detect_ball_speed_spikes(df, config)
    runs = detect_opponent_runs(df, config)

    # Normalise columns
    spikes["trigger_id"] = spikes["spike_id"]
    runs["trigger_id"] = runs["run_id"] + len(spikes) if len(spikes) > 0 else runs["run_id"]

    # Rename type-specific columns to generic names
    spike_cols = spikes.rename(columns={"peak_speed": "trigger_magnitude", "ball_x": "x", "ball_y": "y"})
    run_cols = runs.rename(columns={"atk_speed": "trigger_magnitude", "atk_x": "x", "atk_y": "y",
                                     "attacker_id": "actor_id", "attacker_team": "actor_team"})

    # Keep common columns
    common = ["trigger_id", "frame", "phase", "trigger_type", "trigger_magnitude", "x", "y"]
    for col in common:
        if col not in spike_cols:
            spike_cols[col] = np.nan
        if col not in run_cols:
            run_cols[col] = np.nan

    combined = pd.concat(
        [spike_cols[common], run_cols[common]],
        ignore_index=True,
    )
    combined = combined.sort_values("frame").reset_index(drop=True)
    combined["trigger_id"] = range(len(combined))

    return combined
