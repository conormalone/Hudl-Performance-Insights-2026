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
    """Compute Bekkers TTI for all defender-attacker pairs per frame.

    Returns per-frame, per-defender: tti_value, intercept_probability, closest_attacker.
    """
    req = ["frame_count", "player_id", "team_id_opta", "x", "y", "vx_smooth", "vy_smooth"]
    missing = [c for c in req if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    df = df[req + [c for c in ["goalkeeper", "jersey_number"] if c in df.columns]].copy()
    defenders = _filter_goalkeepers(df[df["team_id_opta"] == own_team_id])
    attackers = _filter_goalkeepers(df[df["team_id_opta"] == opponent_team_id])
    defs = defenders[["frame_count", "player_id", "x", "y", "vx_smooth", "vy_smooth"]].rename(
        columns={"player_id": "defender_id", "x": "def_x", "y": "def_y",
                 "vx_smooth": "def_vx", "vy_smooth": "def_vy"})
    atts = attackers[["frame_count", "player_id", "x", "y"]].rename(
        columns={"player_id": "attacker_id", "x": "att_x", "y": "att_y"})
    pairs = defs.merge(atts, on="frame_count", how="inner")
    if len(pairs) == 0:
        return pd.DataFrame(columns=["frame_count", "player_id", "team_id_opta",
                                     "closest_attacker_id", "closest_attacker_distance",
                                     "tti_value", "intercept_probability"])
    dx = pairs["att_x"] - pairs["def_x"]; dy = pairs["att_y"] - pairs["def_y"]
    dist = np.sqrt(dx**2 + dy**2)
    dm = dist <= config.max_pair_distance
    pairs = pairs[dm].copy(); dx = dx[dm]; dy = dy[dm]; dist = dist[dm]
    if len(pairs) == 0:
        return pd.DataFrame(columns=["frame_count", "player_id", "team_id_opta",
                                     "closest_attacker_id", "closest_attacker_distance",
                                     "tti_value", "intercept_probability"])
    def_speed = np.sqrt(pairs["def_vx"]**2 + pairs["def_vy"]**2)
    v_clamped = np.maximum(def_speed, config.speed_guard)
    tau_dist = dist / v_clamped
    dot = pairs["def_vx"] * dx + pairs["def_vy"] * dy
    cos_theta = np.clip(dot / (v_clamped * dist), -1.0, 1.0)
    tau_beta = config.beta_scaling * (1.0 - cos_theta) * tau_dist
    tti_value = config.reaction_time_s + tau_dist + tau_beta
    tta_threshold = compute_tta_threshold()
    intercept_prob = 1.0 / (1.0 + np.exp(-config.tti_steepness_k * (tta_threshold - tti_value)))
    pairs["distance"] = dist; pairs["tti_value"] = tti_value
    pairs["intercept_probability"] = intercept_prob; pairs["team_id_opta"] = own_team_id
    idx_min = pairs.groupby(["frame_count", "defender_id"])["tti_value"].idxmin()
    result = pairs.loc[idx_min].rename(columns={"defender_id": "player_id", "attacker_id": "closest_attacker_id",
                                                 "distance": "closest_attacker_distance"})
    return result[["frame_count", "player_id", "team_id_opta", "closest_attacker_id",
                   "closest_attacker_distance", "tti_value", "intercept_probability"]].reset_index(drop=True)


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
    """Aggregate pressing accuracy per block per player."""
    req = ["frame_count", "player_id", "is_pressing", "is_correct_press", "intercept_probability", "tti_value"]
    missing = [c for c in req if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    block_records = []
    for block in blocks:
        bid = str(block["block_id"].iloc[0]); ph = int(bid.split("_")[0])
        block_records.append({"block_id": bid, "phase": ph,
                              "start_frame": int(block["frame_count"].min()),
                              "end_frame": int(block["frame_count"].max())})
    bdf = pd.DataFrame(block_records)
    df["_key"] = 1; bdf["_key"] = 1
    merged = df.merge(bdf, on="_key").drop(columns="_key")
    merged = merged[(merged["frame_count"] >= merged["start_frame"]) & (merged["frame_count"] < merged["end_frame"])].copy()
    if len(merged) == 0:
        return pd.DataFrame(columns=["game_id","block_id","phase","player_id","team_id_opta","signal_name",
                                     "signal_value","n_frames","n_presses","mean_intercept_prob","p90_tti",
                                     "total_correct","total_wasteful","pressing_accuracy"])
    grouped = merged.groupby(["block_id", "phase", "player_id"], as_index=False)
    agg = grouped.agg(n_frames=("frame_count","nunique"), n_presses=("is_pressing","sum"),
                      correct_presses=("is_correct_press","sum"),
                      mean_intercept_prob=("intercept_probability","mean"),
                      p90_tti=("tti_value",lambda x: x.quantile(0.90)))
    agg["pressing_accuracy"] = np.where(agg["n_presses"] > 0, agg["correct_presses"] / agg["n_presses"], 0.0)
    agg["total_correct"] = agg["correct_presses"].astype(int)
    agg["total_wasteful"] = (agg["n_presses"] - agg["correct_presses"]).astype(int)
    team_map = df[["player_id", "team_id_opta"]].drop_duplicates("player_id") if "team_id_opta" in df.columns else None
    output = agg.merge(team_map, on="player_id", how="left") if team_map is not None else agg.copy()
    if "team_id_opta" not in output.columns: output["team_id_opta"] = 0
    output["game_id"] = game_id; output["signal_name"] = "pressing_accuracy"
    output["signal_value"] = output["pressing_accuracy"].astype(float)
    sc = ["game_id","block_id","phase","player_id","team_id_opta","signal_name","signal_value","n_frames"]
    ec = ["n_presses","mean_intercept_prob","p90_tti","total_correct","total_wasteful","pressing_accuracy"]
    return output[sc + [c for c in ec if c in output.columns]].reset_index(drop=True)


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
