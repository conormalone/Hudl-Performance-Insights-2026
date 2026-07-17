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

def _build_player_arrays(df: pd.DataFrame) -> dict:
    """Precompute per-player numpy arrays for fast forward-scan.

    Returns dict: {player_id: {"frames": np.array, "v_mag": np.array,
                                "heading": np.array, "vx": np.array}}
    """
    result = {}
    for pid, grp in df.groupby("player_id"):
        grp = grp.sort_values("frame_count")
        arr = {
            "frames": grp["frame_count"].values.astype(np.int64),
            "v_mag": grp["v_mag"].values.astype(np.float64),
            "heading": grp["heading"].values.astype(np.float64),
            "vx": grp.get("vx_smooth", grp.get("vx", grp.get("speed_x", np.zeros(len(grp))))).values.astype(np.float64),
        }
        result[int(pid)] = arr
    return result


def compute_shift_reaction_time(df: pd.DataFrame, trigger_df: pd.DataFrame,
                                config: ShiftLatencyConfig,
                                own_goal_direction: str = "left") -> pd.DataFrame:
    """Compute reaction times from ball-speed spikes / opponent runs.

    Uses precomputed numpy arrays and ``np.searchsorted`` instead of
    Python-dict-per-frame lookups (approx 2-4x faster than the original
    which scanned frame-by-frame with dict access).
    """
    fps = config.frames_per_second
    win_fr = int(config.reaction_window_s * fps)
    min_spd = config.min_reaction_speed
    smoothing = config.direction_smoothing_frames
    reorient_thresh = config.reorientation_threshold_deg
    max_reaction_fr = int(config.max_reaction_time_s * fps)
    gw_sign = -1.0 if own_goal_direction == "left" else 1.0

    if len(trigger_df) == 0:
        return pd.DataFrame(columns=[
            "trigger_id", "player_id", "team_id_opta",
            "reaction_time_s", "pre_trigger_speed",
            "post_trigger_speed", "heading_change_deg",
            "valid", "trigger_type", "trigger_frame",
        ])

    # ── Precompute player arrays ──────────────────────────────────────
    player_arrays = _build_player_arrays(df)

    # ── Identify "defensive" teams at each trigger ────────────────────
    all_teams = sorted(
        int(t) for t in df[df["player_id"] != BALL_PLAYER_ID]["team_id_opta"].unique()
    )

    # ── Iterate triggers (outer loop, ~50 iterations) ─────────────────
    records: list[dict] = []

    for _, tr in trigger_df.iterrows():
        tid = int(tr["trigger_id"])
        tf = int(tr["frame"])
        in_poss_team = tr.get("team_in_possession")

        # Defensive teams = {all teams} \ {in_possession_team}
        def_teams = [t for t in all_teams if pd.isna(in_poss_team) or t != int(in_poss_team)]

        for dt in def_teams:
            # Get defenders on this team
            defenders = df[
                (df["team_id_opta"] == dt)
                & (df["player_id"] != BALL_PLAYER_ID)
            ]["player_id"].unique()

            for raw_pid in defenders:
                pid = int(raw_pid)

                pa = player_arrays.get(pid)
                if pa is None:
                    records.append({
                        "trigger_id": tid, "player_id": pid, "team_id_opta": dt,
                        "reaction_time_s": np.nan, "pre_trigger_speed": 0.0,
                        "post_trigger_speed": 0.0, "heading_change_deg": 0.0,
                        "valid": False, "trigger_type": str(tr["trigger_type"]),
                        "trigger_frame": tf,
                    })
                    continue

                frames = pa["frames"]
                v_mag = pa["v_mag"]
                heading = pa["heading"]
                vx = pa["vx"]

                # ── Pre-trigger speed ──────────────────────────────
                trig_idx = np.searchsorted(frames, tf, side="left")
                start_idx = max(0, trig_idx - smoothing)
                pre_spd = float(np.mean(v_mag[start_idx:trig_idx])) if trig_idx > start_idx else 0.0

                # ── Pre-trigger heading ────────────────────────────
                if trig_idx - start_idx >= 2:
                    pre_hdg = float(np.mean(heading[start_idx:trig_idx]))
                else:
                    records.append({
                        "trigger_id": tid, "player_id": pid, "team_id_opta": dt,
                        "reaction_time_s": np.nan, "pre_trigger_speed": round(pre_spd, 3),
                        "post_trigger_speed": 0.0, "heading_change_deg": 0.0,
                        "valid": False, "trigger_type": str(tr["trigger_type"]),
                        "trigger_frame": tf,
                    })
                    continue

                # ── Forward scan ─────────────────────────────────────
                end_idx = min(
                    trig_idx + win_fr,
                    trig_idx + max_reaction_fr,
                    len(frames),
                )

                scan_frames = frames[trig_idx + 1:end_idx]
                scan_vmag = v_mag[trig_idx + 1:end_idx]
                scan_heading = heading[trig_idx + 1:end_idx]
                scan_vx = vx[trig_idx + 1:end_idx]

                reaction_frame = None
                react_spd = 0.0
                react_hdg = 0.0

                for i in range(len(scan_frames)):
                    vm = scan_vmag[i]
                    if np.isnan(vm) or vm < min_spd:
                        continue

                    v = scan_vx[i]
                    if np.isnan(v) or v * gw_sign >= 0:
                        continue

                    h = scan_heading[i]
                    if np.isnan(h):
                        continue

                    hd = abs(h - pre_hdg)
                    hd = (hd + np.pi) % (2 * np.pi) - np.pi
                    hd_deg = float(np.degrees(abs(hd)))
                    if hd_deg < reorient_thresh:
                        continue

                    # Found reaction
                    reaction_frame = int(scan_frames[i])
                    react_spd = float(vm)
                    react_hdg = float(h)
                    break

                if reaction_frame is not None:
                    rt = (reaction_frame - tf) / fps
                    hd = abs(react_hdg - pre_hdg)
                    hd_deg = float(np.degrees(hd)) % 360
                    if hd_deg > 180:
                        hd_deg = 360 - hd_deg
                    records.append({
                        "trigger_id": tid, "player_id": pid, "team_id_opta": dt,
                        "reaction_time_s": round(rt, 3),
                        "pre_trigger_speed": round(pre_spd, 3),
                        "post_trigger_speed": round(react_spd, 3),
                        "heading_change_deg": round(hd_deg, 1),
                        "valid": True,
                        "trigger_type": str(tr["trigger_type"]),
                        "trigger_frame": tf,
                    })
                else:
                    records.append({
                        "trigger_id": tid, "player_id": pid, "team_id_opta": dt,
                        "reaction_time_s": np.nan,
                        "pre_trigger_speed": round(pre_spd, 3),
                        "post_trigger_speed": 0.0,
                        "heading_change_deg": 0.0,
                        "valid": False,
                        "trigger_type": str(tr["trigger_type"]),
                        "trigger_frame": tf,
                    })

    cols = [
        "trigger_id", "player_id", "team_id_opta", "reaction_time_s",
        "pre_trigger_speed", "post_trigger_speed", "heading_change_deg",
        "valid", "trigger_type", "trigger_frame",
    ]
    if not records:
        return pd.DataFrame({c: pd.Series(
            dtype="float64" if c in ("reaction_time_s", "pre_trigger_speed",
                                     "post_trigger_speed", "heading_change_deg")
            else "int64" if c in ("trigger_id", "player_id", "team_id_opta", "trigger_frame")
            else "bool" if c == "valid" else "object"
        ) for c in cols})
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
