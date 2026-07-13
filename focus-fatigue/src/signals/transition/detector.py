"""Possession transition detection and classification.

Detects frames where ``team_in_possession`` flips from one team to another
and classifies the nature of each transition (expected vs surprise).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import TransitionConfig, DEFAULT_TRANSITION_CONFIG

# ── Constants ───────────────────────────────────────────────────────────────

BALL_PLAYER_ID = -1
"""Player ID used for the ball in tracking data."""


# ═══════════════════════════════════════════════════════════════════════════
# Transition Detection
# ═══════════════════════════════════════════════════════════════════════════


def detect_transitions(
    df: pd.DataFrame,
    team_in_possession_col: str = "team_in_possession",
    min_gap_frames: int = 10,
    frame_col: str = "frame_count",
    phase_col: str = "phase",
    config: Optional[TransitionConfig] = None,
) -> pd.DataFrame:
    """Detect possession transition events.

    A transition occurs when ``team_in_possession`` flips from one non-NaN
    team ID to a different non-NaN team ID. Nearby flips (within
    ``min_gap_frames``) are grouped into a single event, keeping the first
    flip frame as the transition timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``team_in_possession`` column and the standard
        tracking frame/phase columns.
    team_in_possession_col : str
        Column name for which team has possession.
    min_gap_frames : int
        Minimum gap (in frames) between separate transition events.
        Flips closer together are merged into one event.
    frame_col : str
        Column name for frame count.
    phase_col : str
        Column name for match phase.
    config : TransitionConfig | None
        If provided, ``min_gap_frames`` is overridden by ``config.min_gap_frames``.

    Returns
    -------
    pd.DataFrame
        One row per transition event with columns:
            - transition_id : int      — unique zero-based identifier
            - frame : int              — frame number of the flip
            - time_s : float           — elapsed match time in seconds
            - losing_team : int        — team ID that lost possession
            - gaining_team : int       — team ID that gained possession
            - ball_x : float           — ball x-coordinate at transition
            - ball_y : float           — ball y-coordinate at transition
            - phase : int              — match phase

        Returns an empty DataFrame with the correct columns when no
        transitions are found.
    """
    if config is not None:
        min_gap_frames = config.min_gap_frames

    # Get one row per frame with team_in_possession (prefer ball row)
    ball_df = df[df["player_id"] == BALL_PLAYER_ID].copy()
    if len(ball_df) == 0:
        # Fallback: first row per frame
        ball_df = (
            df.groupby([phase_col, frame_col], as_index=False)
            .first()
        )

    ball_df = ball_df.sort_values([phase_col, frame_col])

    raw_flips: list[dict] = []

    for phase in sorted(ball_df[phase_col].unique()):
        phase_ball = ball_df[ball_df[phase_col] == phase]
        tip = phase_ball[team_in_possession_col].values
        frames = phase_ball[frame_col].values
        xs = phase_ball.get("x", phase_ball.get("x_smooth", [np.nan] * len(phase_ball))).values
        ys = phase_ball.get("y", phase_ball.get("y_smooth", [np.nan] * len(phase_ball))).values

        prev_valid: Optional[int] = None
        prev_frame: Optional[int] = None

        for i in range(len(tip)):
            current_val = tip[i]
            if pd.notna(current_val):
                cur = int(current_val)
                if prev_valid is not None and cur != prev_valid:
                    raw_flips.append({
                        "frame": int(frames[i]),
                        "phase": int(phase),
                        "losing_team": prev_valid,
                        "gaining_team": cur,
                        "ball_x": float(xs[i]) if pd.notna(xs[i]) else np.nan,
                        "ball_y": float(ys[i]) if pd.notna(ys[i]) else np.nan,
                    })
                prev_valid = cur
                prev_frame = int(frames[i])

    if not raw_flips:
        return pd.DataFrame(
            columns=[
                "transition_id", "frame", "time_s",
                "losing_team", "gaining_team",
                "ball_x", "ball_y", "phase",
            ]
        )

    flips_df = pd.DataFrame(raw_flips).sort_values(["phase", "frame"]).reset_index(drop=True)

    # ── Group nearby flips ─────────────────────────────────────────────
    # Assign each flip to a transition event by iterating in order.
    # Merge if the gap to the *current group start* is <= min_gap_frames.
    groups: list[list[int]] = []
    current_group: list[int] = [0]
    prev_idx = 0

    for idx in range(1, len(flips_df)):
        gap = flips_df.iloc[idx]["frame"] - flips_df.iloc[prev_idx]["frame"]
        same_phase = (
            flips_df.iloc[idx]["phase"] == flips_df.iloc[prev_idx]["phase"]
        )
        if gap <= min_gap_frames and same_phase:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
            prev_idx = idx

    groups.append(current_group)

    # Build transition events: keep the first flip in each group
    records: list[dict] = []
    for tid, group_indices in enumerate(groups):
        first = flips_df.iloc[group_indices[0]]
        frame = int(first["frame"])
        records.append({
            "transition_id": tid,
            "frame": frame,
            "time_s": frame / 25.0,  # default fps; caller can adjust
            "losing_team": int(first["losing_team"]),
            "gaining_team": int(first["gaining_team"]),
            "ball_x": float(first["ball_x"]),
            "ball_y": float(first["ball_y"]),
            "phase": int(first["phase"]),
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# Transition Type Classification
# ═══════════════════════════════════════════════════════════════════════════


def classify_transition_type(
    df: pd.DataFrame,
    trans_df: pd.DataFrame,
    max_surprise_frames: int = 5,
    frames_per_second: int = 25,
    surprise_speed_threshold: float = 10.0,
    config: Optional[TransitionConfig] = None,
    frame_col: str = "frame_count",
) -> pd.DataFrame:
    """Classify transitions as ``'expected'`` or ``'surprise'``.

    A surprise transition is one where the ball is moving fast (>10 m/s)
    toward the losing team's goal in the frames just before the flip.
    This typically indicates a counter-attack that caught the defending
    team off-guard.

    **Logic**: For each transition, sample the ball's velocity at
    ``max_surprise_frames`` frames before the transition frame. If the
    ball speed exceeds ``surprise_speed_threshold`` and the ball is
    moving toward the losing team's own goal (i.e. in the direction
    the losing team is defending), classify as ``'surprise'``.

    When ball data is not available, the transition is marked
    ``'unknown'``.

    Parameters
    ----------
    df : pd.DataFrame
        Full tracking data. Must contain ball rows (``player_id == -1``)
        with columns ``vx_smooth`` / ``vy_smooth`` (or ``vx`` / ``vy``).
    trans_df : pd.DataFrame
        Transition events from :func:`detect_transitions`.
    max_surprise_frames : int
        Number of frames before the transition to look at for ball
        velocity sampling.
    frames_per_second : int
        Frame rate of the tracking data.
    surprise_speed_threshold : float
        Ball speed (m/s) above which a transition is considered surprise.
    config : TransitionConfig | None
        If provided, overrides the keyword defaults.
    frame_col : str
        Column name for frame count.

    Returns
    -------
    pd.DataFrame
        The input ``trans_df`` with an additional column ``transition_type``
        containing ``'expected'``, ``'surprise'``, or ``'unknown'``.
    """
    if config is not None:
        max_surprise_frames = config.max_surprise_frames
        frames_per_second = config.frames_per_second
        surprise_speed_threshold = config.surprise_ball_speed_threshold

    if len(trans_df) == 0:
        return trans_df.copy().assign(transition_type=pd.Series(dtype="object"))

    # Build a lookup of ball velocity per frame
    ball_df = df[df["player_id"] == BALL_PLAYER_ID].copy()
    if len(ball_df) == 0:
        # No ball data — mark all unknown
        result = trans_df.copy()
        result["transition_type"] = "unknown"
        return result

    # Determine which velocity columns are available
    vx_col = "vx_smooth" if "vx_smooth" in ball_df.columns else "vx"
    vy_col = "vy_smooth" if "vy_smooth" in ball_df.columns else "vy"

    ball_vel = ball_df[[frame_col, vx_col, vy_col]].copy()
    ball_vel.columns = [frame_col, "ball_vx", "ball_vy"]

    result = trans_df.copy()
    types: list[str] = []

    for _, trans_row in trans_df.iterrows():
        trans_frame = int(trans_row["frame"])
        losing_team = int(trans_row["losing_team"])

        # Sample ball velocity before the transition
        sample_frame = max(0, trans_frame - max_surprise_frames)

        vel_rows = ball_vel[ball_vel[frame_col] == sample_frame]
        if len(vel_rows) == 0:
            types.append("unknown")
            continue

        bvx = vel_rows.iloc[0]["ball_vx"]
        bvy = vel_rows.iloc[0]["ball_vy"]

        if pd.isna(bvx) or pd.isna(bvy):
            types.append("unknown")
            continue

        # Velocity is in m/s after smoothing.py np.gradient/dt conversion.
        # Do NOT multiply by frames_per_second — that would inflate values 25x
        # and incorrectly classify nearly all transitions as "surprise".
        speed = np.sqrt(bvx**2 + bvy**2)

        if speed <= surprise_speed_threshold:
            types.append("expected")
            continue

        # Determine ball direction relative to the losing team's goal.
        # The losing team is defending their own goal. After DOP
        # normalisation, the attacking direction is left → right for
        # all teams. The losing team's own goal is at x = -57.
        # A surprise counter-attack means the ball is moving toward
        # the losing goal (negative x direction) fast.
        # We infer direction from ball_team and opponent patterns:
        # If ball_vx < 0 (moving left) AND losing_team was defending
        # the left goal, this is a surprise counter.
        # DOP normalisation makes "own goal = left" for all teams.
        if bvx < 0:
            types.append("surprise")
        else:
            types.append("expected")

    result["transition_type"] = types
    return result
