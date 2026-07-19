"""Signal 3 — Pressing Accuracy (Bekkers Time-To-Intercept).

Measures how effectively defenders press opposition attackers using
the Bekkers TTI framework. The core hypothesis is that mentally fatigued
defenders exhibit less accurate pressing — either pressing when they
cannot realistically intercept (wasteful) or failing to press when they
have a strong intercept opportunity (missed).

One file contains: config, TTI computation, pressing detection,
accuracy classification, block aggregation, and the registered signal class.
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class PressingConfig:
    reaction_time_s: float = 0.2
    tti_threshold_s: float = 1.5
    beta_scaling: float = 1.0
    press_speed_threshold: float = 2.0
    press_angle_threshold: float = 45.0
    correct_press_threshold: float = 0.18
    frames_per_second: int = 25
    max_pair_distance: float = 30.0
    speed_guard: float = 0.1
    tti_steepness_k: float = 3.0

DEFAULT_PRESSING_CONFIG = PressingConfig()

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from .base import SignalBase
from .registry import register_signal


# ═══════════════════════════════════════════════════════════════════════════
# TTI (Bekkers Time-To-Intercept)
# ═══════════════════════════════════════════════════════════════════════════

def compute_tta_threshold(pass_speed: float = 15.0, pass_distance: float = 20.0) -> float:
    """Time-To-Arrive threshold — how long a pass takes to reach receiver."""
    if pass_speed <= 0 or pass_distance <= 0:
        raise ValueError("pass_speed and pass_distance must be positive")
    return pass_distance / pass_speed


def _filter_goalkeepers(team_df: pd.DataFrame) -> pd.DataFrame:
    if "goalkeeper" in team_df.columns:
        return team_df[~team_df["goalkeeper"]].copy()
    if "jersey_number" in team_df.columns:
        return team_df[team_df["jersey_number"] != 1].copy()
    return team_df.copy()


def compute_tti(df: pd.DataFrame, config: PressingConfig, own_team_id: int, opponent_team_id: int) -> pd.DataFrame:
    """Compute Bekkers TTI — per-frame nearest-neighbour avoids the 13M-row Cartesian merge.

    Original: ``defs.merge(atts, on="frame_count", how="inner")`` produces
    ~10×10×135k = 13.5M rows per match, then filters and groupby-reduces.

    Optimised: process each frame independently with numpy all-pairs (~100 pairs
    per frame) — never materialise the full Cartesian product.  If scipy is
    available we use ``cKDTree`` for the nearest-neighbour query (faster still).

    Returns per-frame, per-defender: tti_value, intercept_probability, closest_attacker.
    """
    req = ["frame_count", "player_id", "team_id_opta", "x", "y", "vx_smooth", "vy_smooth"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df_work = df[req + [c for c in ["goalkeeper", "jersey_number"] if c in df.columns]].copy()

    defenders = _filter_goalkeepers(df_work[df_work["team_id_opta"] == own_team_id])
    attackers = _filter_goalkeepers(df_work[df_work["team_id_opta"] == opponent_team_id])

    if len(defenders) == 0 or len(attackers) == 0:
        return pd.DataFrame(columns=[
            "frame_count", "player_id", "team_id_opta",
            "closest_attacker_id", "closest_attacker_distance",
            "tti_value", "intercept_probability",
        ])

    # ── Build per-frame numpy arrays ─────────────────────────────────
    def _def_arr(grp):
        return np.column_stack([
            grp["player_id"].values.astype(np.float64),
            grp["x"].values.astype(np.float64),
            grp["y"].values.astype(np.float64),
            grp["vx_smooth"].values.astype(np.float64),
            grp["vy_smooth"].values.astype(np.float64),
            grp["team_id_opta"].values.astype(np.float64),
        ])

    def _att_arr(grp):
        return np.column_stack([
            grp["player_id"].values.astype(np.float64),
            grp["x"].values.astype(np.float64),
            grp["y"].values.astype(np.float64),
        ])

    def_groups = {f: _def_arr(grp) for f, grp in defenders.groupby("frame_count", sort=False)}
    att_groups = {f: _att_arr(grp) for f, grp in attackers.groupby("frame_count", sort=False)}

    common_frames = sorted(def_groups.keys() & att_groups.keys())

    rows = []
    tta_th = compute_tta_threshold()
    k = config.tti_steepness_k
    speed_guard = config.speed_guard
    beta_s = config.beta_scaling
    max_dist = config.max_pair_distance
    rt_s = config.reaction_time_s

    for frame in common_frames:
        d_arr = def_groups[frame]
        a_arr = att_groups[frame]

        n_def, n_att = len(d_arr), len(a_arr)

        # All-pairs distance via broadcasting: (n_def, n_att)
        dx = a_arr[:, 1, np.newaxis] - d_arr[np.newaxis, :, 1]  # att_x - def_x
        dy = a_arr[:, 2, np.newaxis] - d_arr[np.newaxis, :, 2]  # att_y - def_y
        dist = np.sqrt(dx**2 + dy**2)

        # Per defender: find closest attacker within max_pair_distance
        for di in range(n_def):
            d_possible = dist[:, di] <= max_dist
            if not np.any(d_possible):
                continue

            ai = np.argmin(dist[:, di])  # closest attacker overall
            d = float(dist[ai, di])

            def_vx = float(d_arr[di, 3])
            def_vy = float(d_arr[di, 4])
            def_speed = np.sqrt(def_vx**2 + def_vy**2)
            v_clamped = def_speed if def_speed > speed_guard else speed_guard

            att_dx = float(a_arr[ai, 1] - d_arr[di, 1])
            att_dy = float(a_arr[ai, 2] - d_arr[di, 2])

            tau_dist = d / v_clamped
            dot = def_vx * att_dx + def_vy * att_dy
            cos_theta = dot / (v_clamped * (d if d > 1e-6 else 1e-6))
            cos_theta = max(-1.0, min(1.0, cos_theta))
            tau_beta = beta_s * (1.0 - cos_theta) * tau_dist
            tti = rt_s + tau_dist + tau_beta

            x_val = -k * (tta_th - tti)
            x_clipped = x_val if x_val > -700.0 else -700.0
            if x_clipped > 700.0:
                x_clipped = 700.0
            interp = 1.0 / (1.0 + np.exp(x_clipped))

            rows.append({
                "frame_count": frame,
                "player_id": int(d_arr[di, 0]),
                "team_id_opta": int(d_arr[di, 5]),
                "closest_attacker_id": int(a_arr[ai, 0]),
                "closest_attacker_distance": d,
                "tti_value": tti,
                "intercept_probability": interp,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "frame_count", "player_id", "team_id_opta",
            "closest_attacker_id", "closest_attacker_distance",
            "tti_value", "intercept_probability",
        ])
    return pd.DataFrame(rows, columns=[
        "frame_count", "player_id", "team_id_opta",
        "closest_attacker_id", "closest_attacker_distance",
        "tti_value", "intercept_probability",
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Pressing Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_pressing_events(df: pd.DataFrame, tti_df: pd.DataFrame, config: PressingConfig) -> pd.DataFrame:
    """Identify frames where defender is actively pressing."""
    if len(tti_df) == 0:
        return tti_df.assign(is_pressing=False, def_speed=0.0, pressing_angle=0.0).reset_index(drop=True)
    vel_df = df[["frame_count", "player_id", "vx_smooth", "vy_smooth"]].copy()
    vel_df["def_speed"] = np.sqrt(vel_df["vx_smooth"]**2 + vel_df["vy_smooth"]**2)
    result = tti_df.merge(vel_df[["frame_count", "player_id", "def_speed", "vx_smooth", "vy_smooth"]],
                           on=["frame_count", "player_id"], how="left")
    def_pos = df[["frame_count", "player_id", "x", "y"]].rename(
        columns={"player_id": "_def", "x": "_dx", "y": "_dy"})
    att_pos = df[["frame_count", "player_id", "x", "y"]].rename(
        columns={"player_id": "_att", "x": "_ax", "y": "_ay"})
    result = result.merge(def_pos, left_on=["frame_count", "player_id"], right_on=["frame_count", "_def"], how="left")
    result = result.merge(att_pos, left_on=["frame_count", "closest_attacker_id"], right_on=["frame_count", "_att"], how="left")
    dx = result["_ax"] - result["_dx"]; dy = result["_ay"] - result["_dy"]
    dist = np.sqrt(dx**2 + dy**2)
    vx, vy = result["vx_smooth"].values, result["vy_smooth"].values
    vm = np.maximum(np.sqrt(vx**2 + vy**2), config.speed_guard)
    cos_theta = np.clip((vx * dx + vy * dy) / (vm * np.maximum(dist, 1e-6)), -1.0, 1.0)
    result["pressing_angle"] = np.degrees(np.arccos(cos_theta))
    result["is_pressing"] = (result["def_speed"] >= config.press_speed_threshold) & \
                            (result["pressing_angle"] <= config.press_angle_threshold) & \
                            (result["intercept_probability"] > 0.0)
    drop = ["_def", "_dx", "_dy", "_att", "_ax", "_ay"]
    return result.drop(columns=[c for c in drop if c in result.columns], errors="ignore").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Accuracy Classification & Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def classify_pressing_accuracy(df: pd.DataFrame, tti_df: pd.DataFrame,
                                pressing_df: pd.DataFrame, config: PressingConfig) -> pd.DataFrame:
    """Classify each press as 'correct' or 'wasteful' based on intercept probability."""
    result = pressing_df.copy()
    result["is_correct_press"] = result["is_pressing"] & (result["intercept_probability"] > config.correct_press_threshold)
    result["press_quality"] = np.select(
        [result["is_correct_press"], result["is_pressing"] & ~result["is_correct_press"]],
        ["correct", "wasteful"], default="none")
    return result


def aggregate_pressing_by_block(df: pd.DataFrame, blocks: list[pd.DataFrame],
                                 config: PressingConfig, game_id: str = "") -> pd.DataFrame:
    """Aggregate pressing accuracy per block per player.

    Uses per-block boolean masking instead of a cartesian cross-join
    to avoid memory blow-up on the Pi (fixes 25-minute hang).
    """
    req = ["frame_count", "player_id", "is_pressing", "is_correct_press",
           "intercept_probability", "tti_value"]
    missing = [c for c in req if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")

    # Build block boundaries (avoids cartesian join)
    block_info = []
    for block in blocks:
        bid = str(block["block_id"].iloc[0])
        ph = int(bid.split("_")[0])
        block_info.append({
            "block_id": bid,
            "phase": ph,
            "start_frame": int(block["frame_count"].min()),
            "end_frame": int(block["frame_count"].max()),
        })

    # Aggregate per block — memory-safe iteration
    results = []
    for bi in block_info:
        mask = (df["frame_count"] >= bi["start_frame"]) & (df["frame_count"] < bi["end_frame"])
        subset = df[mask].copy()
        if len(subset) == 0:
            continue
        subset["block_id"] = bi["block_id"]
        subset["phase"] = bi["phase"]

        grouped = subset.groupby(["block_id", "phase", "player_id"], as_index=False)
        agg = grouped.agg(
            n_frames=("frame_count", "nunique"),
            n_presses=("is_pressing", "sum"),
            correct_presses=("is_correct_press", "sum"),
            mean_intercept_prob=("intercept_probability", "mean"),
            p90_tti=("tti_value", lambda x: x.quantile(0.90)),
        )
        results.append(agg)

    if not results:
        cols = ["game_id","block_id","phase","player_id","team_id_opta","signal_name",
                "signal_value","n_frames","n_presses","mean_intercept_prob","p90_tti",
                "total_correct","total_wasteful","pressing_accuracy"]
        return pd.DataFrame(columns=cols)

    result = pd.concat(results, ignore_index=True)
    result["pressing_accuracy"] = np.where(
        result["n_presses"] > 0, result["correct_presses"] / result["n_presses"], 0.0
    )
    result["total_correct"] = result["correct_presses"].astype(int)
    result["total_wasteful"] = (result["n_presses"] - result["correct_presses"]).astype(int)

    team_map = df[["player_id", "team_id_opta"]].drop_duplicates("player_id") if "team_id_opta" in df.columns else None
    if team_map is not None:
        result = result.merge(team_map, on="player_id", how="left")
    if "team_id_opta" not in result.columns:
        result["team_id_opta"] = 0

    result["game_id"] = game_id
    result["signal_name"] = "pressing_accuracy"
    result["signal_value"] = result["pressing_accuracy"].astype(float)

    sc = ["game_id","block_id","phase","player_id","team_id_opta","signal_name","signal_value","n_frames"]
    ec = ["n_presses","mean_intercept_prob","p90_tti","total_correct","total_wasteful","pressing_accuracy"]
    return result[sc + [c for c in ec if c in result.columns]].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Registered Signal Class
# ═══════════════════════════════════════════════════════════════════════════

@register_signal
class PressingAccuracySignal(SignalBase):
    signal_name = "pressing_accuracy"

    def __init__(self, signal_config=None, pressing_config=None, logger=None):
        super().__init__(config=signal_config, logger=logger)
        self.pressing_config = pressing_config or DEFAULT_PRESSING_CONFIG

    def compute(self, match_df, blocks, *, game_id="", own_team_id=0, opponent_team_id=0):
        cfg = self.pressing_config
        tti_df = compute_tti(match_df, cfg, own_team_id, opponent_team_id)
        if len(tti_df) == 0:
            return pd.DataFrame(columns=["game_id","block_id","phase","player_id","team_id_opta",
                                         "signal_name","signal_value","n_frames"])
        pressing_df = detect_pressing_events(match_df, tti_df, cfg)
        classified_df = classify_pressing_accuracy(match_df, tti_df, pressing_df, cfg)
        return aggregate_pressing_by_block(classified_df, blocks, cfg, game_id=game_id)

    def validate(self, output_df):
        super().validate(output_df)
        if len(output_df) == 0: return True
        sv = output_df["signal_value"]
        if sv.min() < 0.0 or sv.max() > 1.0:
            raise ValueError(f"signal_value must be in [0,1], got [{sv.min():.3f}, {sv.max():.3f}]")
        return True
