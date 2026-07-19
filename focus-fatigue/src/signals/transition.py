"""Signal 5 — Transition Recognition.

Measures how quickly defenders recognise and react to possession
transitions (turnovers). The core hypothesis is that cognitively
fatigued defenders exhibit longer perception-reaction delays while
their physical sprint speed remains intact.

One file contains: config, transition detection, type classification,
reaction latency, block aggregation, and the registered signal class.
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TransitionConfig:
    reaction_window_s: float = 2.0
    min_reaction_speed: float = 0.5
    direction_smoothing_frames: int = 5
    reorientation_threshold_deg: float = 45.0
    max_reaction_time_s: float = 5.0
    frames_per_second: int = 25
    surprise_ball_speed_threshold: float = 10.0
    max_surprise_frames: int = 5
    min_gap_frames: int = 10

DEFAULT_TRANSITION_CONFIG = TransitionConfig()

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from .base import SignalBase
from .registry import register_signal

BALL_PLAYER_ID = -1


# ═══════════════════════════════════════════════════════════════════════════
# Transition Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_transitions(df: pd.DataFrame, team_in_possession_col="team_in_possession",
                        min_gap_frames=10, config=None) -> pd.DataFrame:
    """Detect possession transition events (team_in_possession flips)."""
    if config is not None: min_gap_frames = config.min_gap_frames
    ball_df = df[df["player_id"] == BALL_PLAYER_ID].copy()
    if len(ball_df) == 0:
        ball_df = df.groupby(["phase", "frame_count"], as_index=False).first()
    ball_df = ball_df.sort_values(["phase", "frame_count"])
    raw_flips = []
    for phase in sorted(ball_df["phase"].unique()):
        pb = ball_df[ball_df["phase"] == phase]
        tip, frames, xs, ys = pb[team_in_possession_col].values, pb["frame_count"].values, \
                              pb.get("x", pb.get("x_smooth", [np.nan]*len(pb))).values, \
                              pb.get("y", pb.get("y_smooth", [np.nan]*len(pb))).values
        prev_valid = None
        for i in range(len(tip)):
            cv = tip[i]
            if pd.notna(cv):
                cur = int(cv)
                if prev_valid is not None and cur != prev_valid:
                    raw_flips.append({"frame": int(frames[i]), "phase": int(phase),
                                      "losing_team": prev_valid, "gaining_team": cur,
                                      "ball_x": float(xs[i]) if pd.notna(xs[i]) else np.nan,
                                      "ball_y": float(ys[i]) if pd.notna(ys[i]) else np.nan})
                prev_valid = cur
    if not raw_flips:
        return pd.DataFrame(columns=["transition_id", "frame", "time_s", "losing_team",
                                     "gaining_team", "ball_x", "ball_y", "phase"])
    flips_df = pd.DataFrame(raw_flips).sort_values(["phase", "frame"]).reset_index(drop=True)
    groups = [[0]]; prev_idx = 0
    for idx in range(1, len(flips_df)):
        gap = flips_df.iloc[idx]["frame"] - flips_df.iloc[prev_idx]["frame"]
        same = flips_df.iloc[idx]["phase"] == flips_df.iloc[prev_idx]["phase"]
        if gap <= min_gap_frames and same: groups[-1].append(idx)
        else: groups.append([idx]); prev_idx = idx
    records = []
    for tid, gi in enumerate(groups):
        first = flips_df.iloc[gi[0]]
        records.append({"transition_id": tid, "frame": int(first["frame"]),
                        "time_s": int(first["frame"]) / 25.0,
                        "losing_team": int(first["losing_team"]), "gaining_team": int(first["gaining_team"]),
                        "ball_x": float(first["ball_x"]), "ball_y": float(first["ball_y"]),
                        "phase": int(first["phase"])})
    return pd.DataFrame(records)


def classify_transition_type(df: pd.DataFrame, trans_df: pd.DataFrame, config=None) -> pd.DataFrame:
    """Classify transitions as expected/surprise based on ball speed before the flip."""
    if config is not None:
        max_sf = config.max_surprise_frames; sst = config.surprise_ball_speed_threshold
    else:
        max_sf = 5; sst = 10.0
    if len(trans_df) == 0: return trans_df.copy().assign(transition_type=pd.Series(dtype="object"))
    ball_df = df[df["player_id"] == BALL_PLAYER_ID].copy()
    if len(ball_df) == 0:
        result = trans_df.copy(); result["transition_type"] = "unknown"; return result
    vx_col = "vx_smooth" if "vx_smooth" in ball_df.columns else "vx"
    vy_col = "vy_smooth" if "vy_smooth" in ball_df.columns else "vy"
    ball_vel = ball_df[["frame_count", vx_col, vy_col]].copy()
    ball_vel.columns = ["frame_count", "ball_vx", "ball_vy"]
    result = trans_df.copy(); types = []
    for _, tr in trans_df.iterrows():
        tf = int(tr["frame"]); sf = max(0, tf - max_sf)
        vr = ball_vel[ball_vel["frame_count"] == sf]
        if len(vr) == 0: types.append("unknown"); continue
        bvx, bvy = vr.iloc[0]["ball_vx"], vr.iloc[0]["ball_vy"]
        if pd.isna(bvx) or pd.isna(bvy): types.append("unknown"); continue
        speed = np.sqrt(bvx**2 + bvy**2)
        if speed <= sst: types.append("expected")
        elif bvx < 0: types.append("surprise")
        else: types.append("expected")
    result["transition_type"] = types
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Reaction Latency
# ═══════════════════════════════════════════════════════════════════════════

def _build_player_arrays(df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    """Pre-compute per-player numpy arrays for fast reaction time lookup."""
    arrays: dict[int, dict[str, np.ndarray]] = {}
    for pid, grp in df.groupby("player_id"):
        g = grp.sort_values("frame_count")
        vx_col = "vx_smooth" if "vx_smooth" in g.columns else "vx"
        arrays[int(pid)] = {
            "frame": g["frame_count"].values.astype(np.int64),
            "v_mag": g["v_mag"].values.astype(np.float64),
            "heading": g["heading"].values.astype(np.float64),
            "vx": g[vx_col].values.astype(np.float64),
        }
    return arrays


def _get_team_mapping(df: pd.DataFrame) -> dict[int, int]:
    """Build {player_id: team_id_opta} lookup."""
    mapping = {}
    for _, row in df[["player_id", "team_id_opta"]].drop_duplicates(subset="player_id").iterrows():
        pid = int(row["player_id"])
        if pid == BALL_PLAYER_ID:
            continue
        mapping[pid] = int(row["team_id_opta"])
    return mapping


def compute_reaction_time(df: pd.DataFrame, trans_df: pd.DataFrame, config: TransitionConfig,
                           own_goal_direction: str = "left") -> pd.DataFrame:
    """
    Vectorized reaction time computation for possession transitions.

    Replaces frame-by-frame Python loops with numpy vectorized operations
    on pre-computed per-player arrays.
    """
    fps = config.frames_per_second
    win_fr = int(config.reaction_window_s * fps)
    min_spd = config.min_reaction_speed
    smoothing = config.direction_smoothing_frames
    reo_thresh = config.reorientation_threshold_deg
    reo_thresh_rad = np.radians(reo_thresh)
    max_rf = int(config.max_reaction_time_s * fps)
    gw_sign = -1.0 if own_goal_direction == "left" else 1.0

    # Pre-compute per-player arrays
    player_arrays = _build_player_arrays(df)
    player_team = _get_team_mapping(df)

    records: list[dict] = []
    for _, tr in trans_df.iterrows():
        tid = int(tr["transition_id"])
        tf = int(tr["frame"])
        gt = int(tr["gaining_team"])

        # Find players on the gaining team
        def_pids = [pid for pid, pt in player_team.items() if pt == gt]
        if not def_pids:
            continue

        for pid in def_pids:
            pa = player_arrays.get(pid)
            if pa is None:
                records.append({"transition_id": tid, "player_id": pid, "team_id_opta": gt,
                                "reaction_time_s": np.nan, "pre_transition_speed": 0.0,
                                "post_transition_speed": 0.0, "heading_change_deg": 0.0, "valid": False})
                continue

            # ── Pre-transition speed (vectorized) ──
            pre_start = np.searchsorted(pa["frame"], tf - smoothing)
            pre_end = np.searchsorted(pa["frame"], tf)
            if pre_end > pre_start:
                pre_spd = float(np.nanmean(pa["v_mag"][pre_start:pre_end]))
            else:
                pre_spd = 0.0

            # ── Pre-transition heading ──
            before_end = np.searchsorted(pa["frame"], tf)
            before_start = max(0, before_end - smoothing)
            if before_end - before_start >= 2:
                bw_heading = pa["heading"][before_start:before_end]
                bw_heading = bw_heading[~np.isnan(bw_heading)]
                if len(bw_heading) >= 2:
                    pre_h = float(np.mean(bw_heading))
                else:
                    records.append({"transition_id": tid, "player_id": pid, "team_id_opta": gt,
                                    "reaction_time_s": np.nan, "pre_transition_speed": round(pre_spd, 3),
                                    "post_transition_speed": 0.0, "heading_change_deg": 0.0, "valid": False})
                    continue
            else:
                records.append({"transition_id": tid, "player_id": pid, "team_id_opta": gt,
                                "reaction_time_s": np.nan, "pre_transition_speed": round(pre_spd, 3),
                                "post_transition_speed": 0.0, "heading_change_deg": 0.0, "valid": False})
                continue

            # ── Vectorized reaction search ──
            end_f = min(tf + win_fr, tf + max_rf, int(pa["frame"][-1]))
            start_idx = np.searchsorted(pa["frame"], tf + 1)
            end_idx = np.searchsorted(pa["frame"], end_f + 1)

            if start_idx >= end_idx:
                records.append({"transition_id": tid, "player_id": pid, "team_id_opta": gt,
                                "reaction_time_s": np.nan, "pre_transition_speed": round(pre_spd, 3),
                                "post_transition_speed": 0.0, "heading_change_deg": 0.0, "valid": False})
                continue

            # Slice the reaction window
            vm_slice = pa["v_mag"][start_idx:end_idx]
            vx_slice = pa["vx"][start_idx:end_idx]
            hdg_slice = pa["heading"][start_idx:end_idx]
            frame_slice = pa["frame"][start_idx:end_idx]

            # Vectorized conditions
            # 1. Valid v_mag >= threshold
            cond_vm = ~np.isnan(vm_slice) & (vm_slice >= min_spd)
            # 2. Moving toward opponent goal
            cond_vx = ~np.isnan(vx_slice) & (vx_slice * gw_sign < 0)
            # 3. Valid heading
            cond_hdg = ~np.isnan(hdg_slice)

            combined = cond_vm & cond_vx & cond_hdg
            if not combined.any():
                records.append({"transition_id": tid, "player_id": pid, "team_id_opta": gt,
                                "reaction_time_s": np.nan, "pre_transition_speed": round(pre_spd, 3),
                                "post_transition_speed": 0.0, "heading_change_deg": 0.0, "valid": False})
                continue

            # Heading difference (preserving original transition logic: % 360, not [-pi, pi])
            hd = hdg_slice[combined] - pre_h
            hd_deg = np.degrees(np.abs(hd)) % 360
            hd_deg[hd_deg > 180] = 360 - hd_deg[hd_deg > 180]

            # 4. Heading change >= reorientation threshold
            cond_reorient = hd_deg >= reo_thresh
            if not cond_reorient.any():
                records.append({"transition_id": tid, "player_id": pid, "team_id_opta": gt,
                                "reaction_time_s": np.nan, "pre_transition_speed": round(pre_spd, 3),
                                "post_transition_speed": 0.0, "heading_change_deg": 0.0, "valid": False})
                continue

            # Find first matching frame
            match_indices = np.where(combined)[0]
            reorient_indices = match_indices[cond_reorient]
            first_idx = reorient_indices[0]

            reaction_frame = int(frame_slice[first_idx])
            react_vm = float(vm_slice[first_idx])
            react_hdg = float(hdg_slice[first_idx])

            rt = (reaction_frame - tf) / fps
            hd_val = abs(react_hdg - pre_h)
            hd_deg_val = float(np.degrees(hd_val)) % 360
            if hd_deg_val > 180:
                hd_deg_val = 360 - hd_deg_val

            records.append({"transition_id": tid, "player_id": pid, "team_id_opta": gt,
                            "reaction_time_s": round(rt, 3), "pre_transition_speed": round(pre_spd, 3),
                            "post_transition_speed": round(react_vm, 3), "heading_change_deg": round(hd_deg_val, 1),
                            "valid": True})

    cols = ["transition_id", "player_id", "team_id_opta", "reaction_time_s", "pre_transition_speed",
            "post_transition_speed", "heading_change_deg", "valid"]
    if not records:
        return pd.DataFrame({c: pd.Series(dtype="int" if c in ("transition_id","player_id","team_id_opta") else
                                          "float" if c in ("reaction_time_s","pre_transition_speed","post_transition_speed","heading_change_deg") else "bool")
                             for c in cols})
    return pd.DataFrame(records, columns=cols)


def aggregate_latency_by_block(latency_df, blocks, config, game_id=""):
    """Aggregate transition reaction times per block per player."""
    frame_to_block = {}
    for blk in blocks:
        bid = str(blk["block_id"].iloc[0]); ph = int(bid.split("_")[0])
        for f in range(int(blk["frame_count"].min()), int(blk["frame_count"].max()) + 1):
            frame_to_block[f] = (bid, ph)
    tfc = None
    for c in ("frame", "transition_frame"):
        if c in latency_df.columns: tfc = c; break
    if tfc is None:
        return pd.DataFrame(columns=["game_id","block_id","phase","player_id","team_id_opta",
                                     "signal_name","signal_value","n_frames"])
    valid = latency_df[latency_df["valid"]].copy()
    if len(valid) == 0:
        return pd.DataFrame({c: pd.Series(dtype="str" if c in ("game_id","block_id","signal_name") else "int" if c in ("phase","player_id","team_id_opta","n_frames") else "float") for c in ["game_id","block_id","phase","player_id","team_id_opta","signal_name","signal_value","n_frames"]})
    valid["block_id"] = valid[tfc].map(lambda f: frame_to_block.get(int(f), (None, None))[0])
    valid["phase"] = valid[tfc].map(lambda f: frame_to_block.get(int(f), (None, None))[1])
    valid = valid.dropna(subset=["block_id"])
    if len(valid) == 0:
        return pd.DataFrame({c: pd.Series(dtype="str" if c in ("game_id","block_id","signal_name") else "int" if c in ("phase","player_id","team_id_opta","n_frames") else "float") for c in ["game_id","block_id","phase","player_id","team_id_opta","signal_name","signal_value","n_frames"]})
    agg = valid.groupby(["block_id", "phase", "player_id", "team_id_opta"]).agg(
        mean_rt=("reaction_time_s","mean"), p90_rt=("reaction_time_s",lambda x: x.quantile(0.90)),
        n_trans=("transition_id","nunique")).reset_index()
    records = []
    for _, row in agg.iterrows():
        bf = 0
        for blk in blocks:
            if str(blk["block_id"].iloc[0]) == row["block_id"]: bf = len(blk); break
        records.append({"game_id": game_id, "block_id": row["block_id"], "phase": int(row["phase"]),
                        "player_id": int(row["player_id"]), "team_id_opta": int(row["team_id_opta"]),
                        "signal_name": "transition_latency", "signal_value": round(float(row["mean_rt"]), 3),
                        "n_frames": bf})
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# Registered Signal Class
# ═══════════════════════════════════════════════════════════════════════════

@register_signal
class TransitionRecognitionSignal(SignalBase):
    signal_name = "transition_latency"

    def __init__(self, signal_config=None, transition_config=None, logger=None):
        super().__init__(config=signal_config, logger=logger)
        self.transition_config = transition_config or DEFAULT_TRANSITION_CONFIG

    def compute(self, match_df, blocks, *, game_id="", own_goal_direction="left",
                team_in_possession_col="team_in_possession"):
        cfg = self.transition_config
        trans_df = detect_transitions(match_df, team_in_possession_col, config=cfg)
        if len(trans_df) == 0:
            return pd.DataFrame(columns=["game_id","block_id","phase","player_id","team_id_opta","signal_name","signal_value","n_frames"])
        trans_df = classify_transition_type(match_df, trans_df, cfg)
        lat_df = compute_reaction_time(match_df, trans_df, cfg, own_goal_direction)
        trans_meta = trans_df[["transition_id", "frame", "transition_type"]].copy()
        trans_meta["frame"] = trans_meta["frame"].astype(int)
        lat_df = lat_df.merge(trans_meta, on="transition_id", how="left")
        return aggregate_latency_by_block(lat_df, blocks, cfg, game_id=game_id)

    def validate(self, output_df):
        super().validate(output_df)
        if len(output_df) == 0: return True
        sv = output_df["signal_value"]
        if sv.min() < 0: raise ValueError(f"signal_value negative (min={sv.min():.3f})")
        if sv.max() > self.transition_config.max_reaction_time_s:
            raise ValueError("signal_value exceeds max_reaction_time_s")
        return True
