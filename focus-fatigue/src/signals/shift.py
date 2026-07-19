"""Signal 2 — Shift Latency.

Measures how quickly defenders react to sudden shifts in play:
ball speed spikes (fast passes/long balls) and aggressive opponent runs.
The core hypothesis is that cognitively fatigued defenders exhibit
slower reaction times to these sudden events.

One file contains: config, trigger detection, reaction latency, and
the registered signal class.
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ShiftLatencyConfig:
    ball_speed_spike_threshold: float = 15.0
    ball_speed_smoothing_frames: int = 5
    min_spike_gap_frames: int = 25
    opponent_run_speed_threshold: float = 5.0
    opponent_run_acceleration_window: int = 10
    reaction_window_s: float = 3.0
    min_reaction_speed: float = 0.5
    direction_smoothing_frames: int = 5
    reorientation_threshold_deg: float = 30.0
    max_reaction_time_s: float = 5.0
    frames_per_second: int = 25

DEFAULT_SHIFT_LATENCY_CONFIG = ShiftLatencyConfig()

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from .base import SignalBase
from .registry import register_signal

BALL_PLAYER_ID = -1


# ═══════════════════════════════════════════════════════════════════════════
# Trigger Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_ball_speed_spikes(df: pd.DataFrame, config: ShiftLatencyConfig) -> pd.DataFrame:
    """Detect ball speed spike events."""
    ball_df = df[df["player_id"] == BALL_PLAYER_ID].copy()
    if len(ball_df) == 0:
        return pd.DataFrame(columns=["spike_id", "frame", "phase", "peak_speed", "ball_x", "ball_y", "trigger_type"])
    ball_df = ball_df.sort_values("frame_count")
    vx = ball_df.get("vx_smooth", ball_df.get("vx", ball_df.get("speed_x", np.nan))).values
    vy = ball_df.get("vy_smooth", ball_df.get("vy", ball_df.get("speed_y", np.nan))).values
    bs = np.sqrt(vx.astype(np.float64)**2 + vy.astype(np.float64)**2)
    bs_smooth = pd.Series(bs).rolling(window=config.ball_speed_smoothing_frames, center=True, min_periods=1).mean().values
    is_spike = bs_smooth >= config.ball_speed_spike_threshold
    groups = []; cur = []
    for i in range(len(is_spike)):
        if is_spike[i]: cur.append(i)
        else:
            if cur: groups.append(cur); cur = []
    if cur: groups.append(cur)
    if not groups: return pd.DataFrame(columns=["spike_id", "frame", "phase", "peak_speed", "ball_x", "ball_y", "trigger_type"])
    merged = [groups[0]]
    for g in groups[1:]:
        gap = ball_df.iloc[g[0]]["frame_count"] - ball_df.iloc[merged[-1][-1]]["frame_count"]
        merged[-1].extend(g) if gap <= config.min_spike_gap_frames else merged.append(g)
    records = []
    for sid, gi in enumerate(merged):
        speeds = bs_smooth[gi]; pi = gi[np.argmax(speeds)]; pr = ball_df.iloc[pi]
        records.append({"spike_id": sid, "frame": int(pr["frame_count"]), "phase": int(pr.get("phase", 1)),
                        "peak_speed": float(speeds.max()), "ball_x": float(pr.get("x", np.nan)),
                        "ball_y": float(pr.get("y", np.nan)), "trigger_type": "ball_speed_spike"})
    return pd.DataFrame(records)


def detect_opponent_runs(df: pd.DataFrame, config: ShiftLatencyConfig) -> pd.DataFrame:
    """Detect aggressive opponent runs."""
    if "team_in_possession" not in df.columns:
        return pd.DataFrame(columns=["run_id", "frame", "phase", "attacker_id", "attacker_team",
                                     "atk_speed", "atk_x", "atk_y", "trigger_type"])
    outfield = df[(df["player_id"] != BALL_PLAYER_ID) & df["team_in_possession"].notna() &
                  (df["team_id_opta"] == df["team_in_possession"])].copy()
    if len(outfield) == 0: return pd.DataFrame(columns=["run_id", "frame", "phase", "attacker_id",
                                                         "attacker_team", "atk_speed", "atk_x", "atk_y", "trigger_type"])
    vx = outfield.get("vx_smooth", outfield.get("vx", 0)).values.astype(np.float64)
    vy = outfield.get("vy_smooth", outfield.get("vy", 0)).values.astype(np.float64)
    outfield["atk_speed"] = np.sqrt(vx**2 + vy**2)
    sprinting = outfield[outfield["atk_speed"] >= config.opponent_run_speed_threshold].copy()
    if len(sprinting) == 0: return pd.DataFrame(columns=["run_id", "frame", "phase", "attacker_id",
                                                         "attacker_team", "atk_speed", "atk_x", "atk_y", "trigger_type"])
    records = []
    for (pid, team), grp in sprinting.groupby(["player_id", "team_id_opta"]):
        grp = grp.sort_values("frame_count")
        cur_g = []; all_g = []
        for i in range(len(grp)):
            if cur_g and (grp.iloc[i]["frame_count"] - grp.iloc[cur_g[-1]]["frame_count"]) > 5:
                all_g.append(cur_g); cur_g = [i]
            else: cur_g.append(i)
        if cur_g: all_g.append(cur_g)
        for gi in all_g:
            sp = grp.iloc[gi]["atk_speed"].values; pi_grp = gi[np.argmax(sp)]; pr = grp.iloc[pi_grp]
            records.append({"run_id": len(records), "frame": int(pr["frame_count"]), "phase": int(pr.get("phase", 1)),
                            "attacker_id": int(pid), "attacker_team": int(team), "atk_speed": float(sp.max()),
                            "atk_x": float(pr.get("x", np.nan)), "atk_y": float(pr.get("y", np.nan)),
                            "trigger_type": "opponent_run"})
    if not records: return pd.DataFrame(columns=["run_id", "frame", "phase", "attacker_id",
                                                 "attacker_team", "atk_speed", "atk_x", "atk_y", "trigger_type"])
    return pd.DataFrame(records)


def detect_all_triggers(df: pd.DataFrame, config: ShiftLatencyConfig) -> pd.DataFrame:
    """Combine ball speed spikes and opponent runs into unified trigger events."""
    spikes = detect_ball_speed_spikes(df, config)
    runs = detect_opponent_runs(df, config)
    if len(spikes) == 0 and len(runs) == 0:
        return pd.DataFrame(columns=["trigger_id", "frame", "phase", "trigger_type", "trigger_magnitude", "x", "y"])
    spikes["trigger_id"] = spikes["spike_id"]; runs["trigger_id"] = runs["run_id"] + (len(spikes) if len(spikes) > 0 else 0)
    sc = spikes.rename(columns={"peak_speed": "trigger_magnitude", "ball_x": "x", "ball_y": "y"})
    rc = runs.rename(columns={"atk_speed": "trigger_magnitude", "atk_x": "x", "atk_y": "y",
                               "attacker_id": "actor_id", "attacker_team": "actor_team"})
    common = ["trigger_id", "frame", "phase", "trigger_type", "trigger_magnitude", "x", "y"]
    for c in common:
        if c not in sc: sc[c] = np.nan
        if c not in rc: rc[c] = np.nan
    combined = pd.concat([sc[common], rc[common]], ignore_index=True).sort_values("frame").reset_index(drop=True)
    combined["trigger_id"] = range(len(combined))
    return combined


# ═══════════════════════════════════════════════════════════════════════════
# Reaction Latency
# ═══════════════════════════════════════════════════════════════════════════

def _build_player_arrays(df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    """Pre-compute per-player numpy arrays for fast reaction time lookup.

    Returns {player_id: {"frame": ndarray, "v_mag": ndarray,
                          "heading": ndarray, "vx": ndarray}}
    """
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


def compute_shift_reaction_time(df: pd.DataFrame, trigger_df: pd.DataFrame,
                                config: ShiftLatencyConfig,
                                own_goal_direction: str = "left") -> pd.DataFrame:
    """
    Vectorized reaction time computation for shift triggers.

    Replaces frame-by-frame Python loops with numpy vectorized operations
    on pre-computed per-player arrays. For each trigger+defender pair,
    the reaction window is sliced out of the player's array and all
    conditions are checked in one vectorized pass.
    """
    fps = config.frames_per_second
    win_fr = int(config.reaction_window_s * fps)
    min_spd = config.min_reaction_speed
    smoothing = config.direction_smoothing_frames
    reorient_thresh = config.reorientation_threshold_deg
    reorient_thresh_rad = np.radians(reorient_thresh)
    max_reaction_fr = int(config.max_reaction_time_s * fps)
    gw_sign = -1.0 if own_goal_direction == "left" else 1.0

    # Pre-compute per-player arrays (avoids repeated groupby & DataFrame slicing)
    player_arrays = _build_player_arrays(df)
    player_team = _get_team_mapping(df)

    # Build ball-frame lookup for team_in_possession
    ball_df = df[df["player_id"] == BALL_PLAYER_ID][["frame_count", "team_in_possession"]].drop_duplicates("frame_count")
    ball_frames = ball_df.set_index("frame_count")["team_in_possession"].to_dict()

    # Group triggers by type for cleaner output
    trigger_types = {}
    for _, tr in trigger_df.iterrows():
        trigger_types[int(tr["trigger_id"])] = str(tr["trigger_type"])

    all_player_ids = set(a for a in player_arrays.keys() if a != BALL_PLAYER_ID)

    records: list[dict] = []
    for _, tr in trigger_df.iterrows():
        tid = int(tr["trigger_id"])
        tf = int(tr["frame"])

        # Determine defending teams
        in_poss_team = ball_frames.get(tf)
        if in_poss_team is None or (isinstance(in_poss_team, float) and np.isnan(in_poss_team)):
            def_teams = set(player_team.values())
        else:
            in_poss_val = int(in_poss_team)
            def_teams = {t for t in set(player_team.values()) if t != in_poss_val}

        for dt in def_teams:
            # Find defenders on this team
            def_pids = [pid for pid, pt in player_team.items() if pt == dt]
            if not def_pids:
                continue

            for pid in def_pids:
                pa = player_arrays.get(pid)
                if pa is None:
                    records.append({"trigger_id": tid, "player_id": pid, "team_id_opta": dt,
                                    "reaction_time_s": np.nan, "pre_trigger_speed": 0.0,
                                    "post_trigger_speed": 0.0, "heading_change_deg": 0.0,
                                    "valid": False, "trigger_type": trigger_types.get(tid, ""),
                                    "trigger_frame": tf})
                    continue

                # ── Pre-trigger speed (vectorized) ──
                pre_start = np.searchsorted(pa["frame"], tf - smoothing)
                pre_end = np.searchsorted(pa["frame"], tf)
                if pre_end > pre_start:
                    pre_spd = float(np.nanmean(pa["v_mag"][pre_start:pre_end]))
                else:
                    pre_spd = 0.0

                # ── Pre-trigger heading (from last `smoothing` frames) ──
                before_end = np.searchsorted(pa["frame"], tf)
                before_start = max(0, before_end - smoothing)
                if before_end - before_start >= 2:
                    bw_heading = pa["heading"][before_start:before_end]
                    bw_heading = bw_heading[~np.isnan(bw_heading)]
                    if len(bw_heading) >= 2:
                        pre_hdg = float(np.mean(bw_heading))
                    else:
                        records.append({"trigger_id": tid, "player_id": pid, "team_id_opta": dt,
                                        "reaction_time_s": np.nan, "pre_trigger_speed": round(pre_spd, 3),
                                        "post_trigger_speed": 0.0, "heading_change_deg": 0.0,
                                        "valid": False, "trigger_type": trigger_types.get(tid, ""),
                                        "trigger_frame": tf})
                        continue
                else:
                    records.append({"trigger_id": tid, "player_id": pid, "team_id_opta": dt,
                                    "reaction_time_s": np.nan, "pre_trigger_speed": round(pre_spd, 3),
                                    "post_trigger_speed": 0.0, "heading_change_deg": 0.0,
                                    "valid": False, "trigger_type": trigger_types.get(tid, ""),
                                    "trigger_frame": tf})
                    continue

                # ── Vectorized reaction search ──
                end_f = min(tf + win_fr, tf + max_reaction_fr, int(pa["frame"][-1]))
                start_idx = np.searchsorted(pa["frame"], tf + 1)
                end_idx = np.searchsorted(pa["frame"], end_f + 1)

                if start_idx >= end_idx:
                    records.append({"trigger_id": tid, "player_id": pid, "team_id_opta": dt,
                                    "reaction_time_s": np.nan, "pre_trigger_speed": round(pre_spd, 3),
                                    "post_trigger_speed": 0.0, "heading_change_deg": 0.0,
                                    "valid": False, "trigger_type": trigger_types.get(tid, ""),
                                    "trigger_frame": tf})
                    continue

                # Slice the reaction window
                vm_slice = pa["v_mag"][start_idx:end_idx]
                vx_slice = pa["vx"][start_idx:end_idx]
                hdg_slice = pa["heading"][start_idx:end_idx]
                frame_slice = pa["frame"][start_idx:end_idx]

                # Vectorized conditions (all at once, no per-frame Python loop)
                # 1. Valid v_mag >= threshold
                cond_vm = ~np.isnan(vm_slice) & (vm_slice >= min_spd)
                # 2. Moving toward opponent goal
                cond_vx = ~np.isnan(vx_slice) & (vx_slice * gw_sign < 0)
                # 3. Valid heading
                cond_hdg = ~np.isnan(hdg_slice)

                combined = cond_vm & cond_vx & cond_hdg
                if not combined.any():
                    records.append({"trigger_id": tid, "player_id": pid, "team_id_opta": dt,
                                    "reaction_time_s": np.nan, "pre_trigger_speed": round(pre_spd, 3),
                                    "post_trigger_speed": 0.0, "heading_change_deg": 0.0,
                                    "valid": False, "trigger_type": trigger_types.get(tid, ""),
                                    "trigger_frame": tf})
                    continue

                # Heading difference normalized to [-pi, pi]
                hd_diff = hdg_slice[combined] - pre_hdg
                hd_diff = (hd_diff + np.pi) % (2 * np.pi) - np.pi
                hd_deg = np.degrees(np.abs(hd_diff))

                # 4. Heading change >= reorientation threshold
                cond_reorient = hd_deg >= reorient_thresh
                if not cond_reorient.any():
                    records.append({"trigger_id": tid, "player_id": pid, "team_id_opta": dt,
                                    "reaction_time_s": np.nan, "pre_trigger_speed": round(pre_spd, 3),
                                    "post_trigger_speed": 0.0, "heading_change_deg": 0.0,
                                    "valid": False, "trigger_type": trigger_types.get(tid, ""),
                                    "trigger_frame": tf})
                    continue

                # Find first matching frame
                match_indices = np.where(combined)[0]
                reorient_indices = match_indices[cond_reorient]
                first_idx = reorient_indices[0]

                reaction_frame = int(frame_slice[first_idx])
                react_vm = float(vm_slice[first_idx])
                react_hdg = float(hdg_slice[first_idx])

                rt = (reaction_frame - tf) / fps
                hd = abs(react_hdg - pre_hdg)
                hd_deg = float(np.degrees(hd)) % 360
                if hd_deg > 180:
                    hd_deg = 360 - hd_deg

                records.append({"trigger_id": tid, "player_id": pid, "team_id_opta": dt,
                                "reaction_time_s": round(rt, 3), "pre_trigger_speed": round(pre_spd, 3),
                                "post_trigger_speed": round(react_vm, 3), "heading_change_deg": round(hd_deg, 1),
                                "valid": True, "trigger_type": trigger_types.get(tid, ""),
                                "trigger_frame": tf})

    cols = ["trigger_id", "player_id", "team_id_opta", "reaction_time_s", "pre_trigger_speed",
            "post_trigger_speed", "heading_change_deg", "valid", "trigger_type", "trigger_frame"]
    if not records:
        return pd.DataFrame({c: pd.Series(dtype="float64" if c in ("reaction_time_s", "pre_trigger_speed",
                                     "post_trigger_speed", "heading_change_deg") else "int64" if c in
                                     ("trigger_id", "player_id", "team_id_opta", "trigger_frame") else
                                     "bool" if c == "valid" else "object") for c in cols})
    return pd.DataFrame(records, columns=cols)


def aggregate_shift_latency_by_block(latency_df: pd.DataFrame, blocks: list[dict],
                                      config: ShiftLatencyConfig, game_id: str = "") -> pd.DataFrame:
    frame_to_block = {}
    for blk in blocks:
        bid = blk["block_id"]; ph = blk["phase"]
        for f in range(blk["start_frame"], blk["end_frame"] + 1):
            frame_to_block[f] = (bid, ph)
    valid = latency_df[latency_df["valid"]].copy()
    if len(valid) == 0:
        return pd.DataFrame({c: pd.Series(dtype="str" if c in ("game_id","block_id","signal_name") else
                                          "int" if c in ("phase","player_id","team_id_opta","n_frames") else "float")
                             for c in ["game_id","block_id","phase","player_id","team_id_opta","signal_name","signal_value","n_frames"]})
    valid["block_id"] = valid["trigger_frame"].map(lambda f: frame_to_block.get(int(f), (None, None))[0])
    valid["phase"] = valid["trigger_frame"].map(lambda f: frame_to_block.get(int(f), (None, None))[1])
    valid = valid.dropna(subset=["block_id"])
    if len(valid) == 0:
        return pd.DataFrame({c: pd.Series(dtype="str" if c in ("game_id","block_id","signal_name") else
                                          "int" if c in ("phase","player_id","team_id_opta","n_frames") else "float")
                             for c in ["game_id","block_id","phase","player_id","team_id_opta","signal_name","signal_value","n_frames"]})
    agg = valid.groupby(["block_id", "phase", "player_id", "team_id_opta"]).agg(
        mean_reaction_time=("reaction_time_s", "mean"),
        p90_reaction_time=("reaction_time_s", lambda x: x.quantile(0.90)),
        n_triggers=("trigger_id", "nunique")).reset_index()
    records = []
    for _, row in agg.iterrows():
        bf = 0
        for blk in blocks:
            if blk["block_id"] == row["block_id"]: bf = blk.get("end_frame", 0) - blk.get("start_frame", 0); break
        records.append({"game_id": game_id, "block_id": row["block_id"], "phase": int(row["phase"]),
                        "player_id": int(row["player_id"]), "team_id_opta": int(row["team_id_opta"]),
                        "signal_name": "shift_latency", "signal_value": round(float(row["mean_reaction_time"]), 3),
                        "n_frames": bf})
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# Registered Signal Class
# ═══════════════════════════════════════════════════════════════════════════

@register_signal
class ShiftLatencySignal(SignalBase):
    signal_name = "shift_latency"

    def __init__(self, signal_config=None, shift_config=None, logger=None):
        super().__init__(config=signal_config, logger=logger)
        self.shift_config = shift_config or DEFAULT_SHIFT_LATENCY_CONFIG

    def compute(self, match_df, blocks, *, game_id="", own_goal_direction="left"):
        cfg = self.shift_config
        trigger_df = detect_all_triggers(match_df, cfg)
        if len(trigger_df) == 0:
            return pd.DataFrame(columns=["game_id","block_id","phase","player_id","team_id_opta","signal_name","signal_value","n_frames"])
        latency_df = compute_shift_reaction_time(match_df, trigger_df, cfg, own_goal_direction)
        return aggregate_shift_latency_by_block(latency_df, blocks, cfg, game_id=game_id)

    def validate(self, output_df):
        super().validate(output_df)
        if len(output_df) == 0: return True
        sv = output_df["signal_value"]
        if sv.min() < 0: raise ValueError(f"signal_value negative (min={sv.min():.3f})")
        if sv.max() > self.shift_config.max_reaction_time_s:
            raise ValueError(f"signal_value exceeds max_reaction_time_s")
        return True
