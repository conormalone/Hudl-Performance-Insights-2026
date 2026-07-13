"""Pressing event detection for Signal 3 — Pressing Accuracy.

Identifies frames where a defender is actively pressing an opponent
based on speed, movement direction, and intercept probability criteria.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PressingConfig

# ── Public Functions ───────────────────────────────────────────────────────


def detect_pressing_events(
    df: pd.DataFrame,
    tti_df: pd.DataFrame,
    config: PressingConfig,
) -> pd.DataFrame:
    """Identify frames where a defender is actively pressing an opponent.

    A defender is considered *pressing* when **all** of the following
    criteria are met:

    1. **Speed criterion:** The defender's ground speed exceeds
       ``press_speed_threshold`` (≥ 2.0 m/s by default).
    2. **Direction criterion:** The defender's velocity vector points
       within ``press_angle_threshold`` degrees of the direction to the
       nearest attacker (i.e. their movement is goal-directed toward
       the opponent).
    3. **Intercept criterion:** The defender has a non-zero intercept
       probability (they are moving in a way that could realistically
       cut off the pass).

    Parameters
    ----------
    df : pd.DataFrame
        Full tracking data. Must contain: ``frame_count``, ``player_id``,
        ``vx_smooth``, ``vy_smooth``, ``x``, ``y``.
    tti_df : pd.DataFrame
        Output of :func:`src.signals.pressing.tti.compute_tti`, containing
        per-frame per-defender TTI estimates.
    config : PressingConfig
        Configuration with pressing detection thresholds.

    Returns
    -------
    pd.DataFrame
        The original ``tti_df`` augmented with a boolean column
        ``is_pressing`` indicating whether the defender is actively
        pressing in that frame. Also includes ``def_speed`` (m/s) and
        ``pressing_angle`` (degrees) for downstream analysis.
    """
    # ── Guard: empty input ─────────────────────────────────────────────
    if len(tti_df) == 0:
        return tti_df.assign(
            is_pressing=False, def_speed=0.0, pressing_angle=0.0
        ).reset_index(drop=True)

    # ── Merge velocity data from full tracking ─────────────────────────
    vel_cols = ["frame_count", "player_id", "vx_smooth", "vy_smooth"]
    missing = [c for c in vel_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Tracking DataFrame missing columns: {missing}"
        )

    vel_df = df[vel_cols].copy()
    vel_df["def_speed"] = np.sqrt(
        vel_df["vx_smooth"] ** 2 + vel_df["vy_smooth"] ** 2
    )

    result = tti_df.merge(
        vel_df[["frame_count", "player_id", "def_speed", "vx_smooth", "vy_smooth"]],
        on=["frame_count", "player_id"],
        how="left",
    )

    # ── Merge defender and attacker positions ──────────────────────────
    pos_cols = ["frame_count", "player_id", "x", "y"]
    def_pos = df[pos_cols].rename(
        columns={"player_id": "_def_pid", "x": "_def_x", "y": "_def_y"}
    )
    att_pos = df[pos_cols].rename(
        columns={"player_id": "_att_pid", "x": "_att_x", "y": "_att_y"}
    )

    result = result.merge(
        def_pos,
        left_on=["frame_count", "player_id"],
        right_on=["frame_count", "_def_pid"],
        how="left",
    )

    result = result.merge(
        att_pos,
        left_on=["frame_count", "closest_attacker_id"],
        right_on=["frame_count", "_att_pid"],
        how="left",
    )

    # ── Compute direction angle ────────────────────────────────────────
    dx = result["_att_x"] - result["_def_x"]
    dy = result["_att_y"] - result["_def_y"]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    vx = result["vx_smooth"].values
    vy = result["vy_smooth"].values
    v_mag = np.maximum(
        np.sqrt(vx ** 2 + vy ** 2), config.speed_guard
    )
    cos_theta = (vx * dx + vy * dy) / (v_mag * np.maximum(dist, 1e-6))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))

    result["pressing_angle"] = angle_deg

    # ── Apply criteria ─────────────────────────────────────────────────
    speed_ok = result["def_speed"] >= config.press_speed_threshold
    direction_ok = angle_deg <= config.press_angle_threshold
    intercept_ok = result["intercept_probability"] > 0.0

    result["is_pressing"] = speed_ok & direction_ok & intercept_ok

    # ── Clean temporary columns ────────────────────────────────────────
    drop_cols = [
        "_def_pid", "_def_x", "_def_y",
        "_att_pid", "_att_x", "_att_y",
    ]
    result = result.drop(columns=[c for c in drop_cols if c in result.columns],
                         errors="ignore")

    return result.reset_index(drop=True)
