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
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from .base import SignalBase
from .registry import register_signal

BALL_PLAYER_ID = -1

# ── Debug flag: set True to print per-sub-function timings ─────────────
# Set to True when profiling, False for production.
_DEBUG_TIMING = False


def _log_timing(label: str, t0: float) -> float:
    """Print elapsed time if _DEBUG_TIMING is True.
    Returns current time for chaining: t0 = _log_timing("...", t0)
    """
    if _DEBUG_TIMING:
        et = time.time() - t0
        print(f"    {label}: {et:.1f}s")
    return time.time()


# ═══════════════════════════════════════════════════════════════════════════
# Trigger Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_ball_speed_spikes(df: pd.DataFrame, config: ShiftLatencyConfig) -> pd.DataFrame:
    """Detect ball speed spike events."""
    # Pre-filter to only needed columns before copy
    _avail = [c for c in ("frame_count", "phase", "x", "y",
                          "vx_smooth", "vy_smooth", "vx", "vy", "speed_x", "speed_y")
              if c in df.columns]
    ball_df = df.loc[df["player_id"] == BALL_PLAYER_ID, _avail].copy()
    if len(ball_df) == 0:
        return pd.DataFrame(columns=["spike_id", "frame", "phase", "peak_speed", "ball_x", "ball_y", "trigger_type"])
    ball_df = ball_df.sort_values("frame_count")
    vx = ball_df.get("vx_smooth", ball_df.get("vx", ball_df.get("speed_x", np.nan))).values
    vy = ball_df.get("vy_smooth", ball_df.get("vy", ball_df.get("speed_y", np.nan))).values
    bs = np.sqrt(vx.astype(np.float64)**2 + vy.astype(np.float64)**2)
    bs_smooth = pd.Series(bs).rolling(window=config.ball_speed_smoothing_frames, center=True, min_periods=1).mean().values
    is_spike = bs_smooth >= config.ball_speed_spike_threshold
    if not is_spike.any():
        return pd.DataFrame(columns=["spike_id", "frame", "phase", "peak_speed", "ball_x", "ball_y", "trigger_type"])
    # Vectorised spike grouping
    edges = np.diff(is_spike.astype(np.int8), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    groups = [list(range(s, e)) for s, e in zip(starts, ends)]
    # Merge nearby groups
    merged = [groups[0]]
    fc_vals = ball_df["frame_count"].values
    for g in groups[1:]:
        gap = fc_vals[g[0]] - fc_vals[merged[-1][-1]]
        if gap <= config.min_spike_gap_frames:
            merged[-1].extend(g)
        else:
            merged.append(g)
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
    needed_cols = ["player_id", "team_id_opta", "frame_count", "phase", "x", "y",
                   "vx_smooth", "vy_smooth", "vx", "vy", "speed_x", "speed_y"]
    needed_cols = [c for c in needed_cols if c in df.columns]
    out_mask = (df["player_id"] != BALL_PLAYER_ID) & df["team_in_possession"].notna() & (df["team_id_opta"] == df["team_in_possession"])
    outfield = df.loc[out_mask, needed_cols].copy()
    if len(outfield) == 0:
        return pd.DataFrame(columns=["run_id", "frame", "phase", "attacker_id",
                                     "attacker_team", "atk_speed", "atk_x", "atk_y", "trigger_type"])
    vx = outfield.get("vx_smooth", outfield.get("vx", 0)).values.astype(np.float64)
    vy = outfield.get("vy_smooth", outfield.get("vy", 0)).values.astype(np.float64)
    atk_speed = np.sqrt(vx**2 + vy**2)
    sprint_mask = atk_speed >= config.opponent_run_speed_threshold
    sprinting = outfield.loc[sprint_mask].copy()
    if len(sprinting) == 0:
        return pd.DataFrame(columns=["run_id", "frame", "phase", "attacker_id",
                                     "attacker_team", "atk_speed", "atk_x", "atk_y", "trigger_type"])
    sprinting["atk_speed"] = atk_speed[sprint_mask]
    # Pre-sort once
    sprinting = sprinting.sort_values(["player_id", "team_id_opta", "frame_count"])

    records: list[dict] = []
    for (pid, team), grp in sprinting.groupby(["player_id", "team_id_opta"], sort=False):
        # Vectorised gap detection
        frames_arr = grp["frame_count"].values
        if len(frames_arr) == 0:
            continue
        gaps = np.diff(frames_arr) > 5
        split_pts = np.where(gaps)[0] + 1
        fragments = np.split(np.arange(len(grp)), split_pts)
        for gi in fragments:
            if len(gi) == 0:
                continue
            sp = grp["atk_speed"].values[gi]
            pi_grp = gi[np.argmax(sp)]
            pr = grp.iloc[pi_grp]
            records.append({"run_id": len(records), "frame": int(pr["frame_count"]),
                            "phase": int(pr.get("phase", 1)),
                            "attacker_id": int(pid), "attacker_team": int(team),
                            "atk_speed": float(sp.max()),
                            "atk_x": float(pr.get("x", np.nan)),
                            "atk_y": float(pr.get("y", np.nan)),
                            "trigger_type": "opponent_run"})
    if not records:
        return pd.DataFrame(columns=["run_id", "frame", "phase", "attacker_id",
                                     "attacker_team", "atk_speed", "atk_x", "atk_y", "trigger_type"])
    return pd.DataFrame(records)


def detect_all_triggers(df: pd.DataFrame, config: ShiftLatencyConfig) -> pd.DataFrame:
    """Combine ball speed spikes and opponent runs into unified trigger events."""
    _t_debug = time.time()
    spikes = detect_ball_speed_spikes(df, config)
    _t_debug = _log_timing("detect_ball_speed_spikes", _t_debug)
    runs = detect_opponent_runs(df, config)
    _t_debug = _log_timing("detect_opponent_runs", _t_debug)
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
    # Pre-extract only needed columns before groupby to avoid carrying
    # 40+ pressure columns through groupby overhead
    vx_col = "vx_smooth" if "vx_smooth" in df.columns else ("vx" if "vx" in df.columns else "speed_x")
    needed = ["player_id", "frame_count", "v_mag", "heading", vx_col]
    sub = df[needed]
    vx_col_idx = sub.columns.get_loc(vx_col)

    result = {}
    for pid, grp in sub.groupby("player_id", sort=False):
        grp = grp.sort_values("frame_count")
        arr = {
            "frames": grp["frame_count"].values.astype(np.int64),
            "v_mag": grp["v_mag"].values.astype(np.float64),
            "heading": grp["heading"].values.astype(np.float64),
            "vx": grp.iloc[:, vx_col_idx].values.astype(np.float64),
        }
        result[int(pid)] = arr
    return result


def compute_shift_reaction_time(df: pd.DataFrame, trigger_df: pd.DataFrame,
                                config: ShiftLatencyConfig,
                                own_goal_direction: str = "left") -> pd.DataFrame:
    """Compute reaction times from ball-speed spikes / opponent runs.

    Uses precomputed numpy arrays, ``np.searchsorted``, and a fully
    vectorised forward scan (boolean mask + ``np.argmax``) instead of
    Python-dict-per-frame lookups or Python-level frame scanning.
    Approx 50-100x faster than the original loop-based scan.
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

    t0 = time.time()

    # ── Precompute player arrays ──────────────────────────────────────
    player_arrays = _build_player_arrays(df)
    t0 = _log_timing("_build_player_arrays", t0)

    # ── Precompute per-team defender lists ONCE ───────────────────────
    # Hoisted from inside the trigger loop — previously filtered the FULL
    # 3.4M-row DataFrame ~100 times (50 triggers × 2 teams) just to get
    # player IDs. Now done once.
    all_teams = sorted(
        int(t) for t in df[df["player_id"] != BALL_PLAYER_ID]["team_id_opta"].unique()
    )
    team_players: dict[int, np.ndarray] = {}
    for t in all_teams:
        mask = (df["team_id_opta"] == t) & (df["player_id"] != BALL_PLAYER_ID)
        team_players[t] = df.loc[mask, "player_id"].unique()

    # ── Precompute frame→possession lookup ───────────────────────────
    # The original code looked up team_in_possession from the ball's row
    # at each trigger frame via a dict-of-dicts.  We build a flat dict
    # instead — much faster and correct.
    ball_possession: dict[int, int | None] = {}
    ball_df = df[df["player_id"] == BALL_PLAYER_ID]
    if len(ball_df) > 0:
        for _, brow in ball_df.iterrows():
            f = int(brow["frame_count"])
            v = brow.get("team_in_possession")
            ball_possession[f] = int(v) if pd.notna(v) else None

    t0 = _log_timing("team_players setup", t0)

    # ── Iterate triggers (outer loop, ~50 iterations) ─────────────────
    records: list[dict] = []
    n_triggers = len(trigger_df)

    for _, tr in trigger_df.iterrows():
        tid = int(tr["trigger_id"])
        tf = int(tr["frame"])
        in_poss_team = ball_possession.get(tf)

        # Defensive teams = {all teams} \ {in_possession_team}
        def_teams = [t for t in all_teams if in_poss_team is None or t != in_poss_team]

        for dt in def_teams:
            defenders = team_players[dt]

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

                # ── Forward scan (vectorised) ────────────────────────
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

                if len(scan_frames) > 0:
                    # Vectorised: precompute boolean conditions for ALL scan
                    # frames at once, then find the first frame where ALL
                    # conditions are True via np.argmax.
                    speed_ok = (scan_vmag >= min_spd) & ~np.isnan(scan_vmag)
                    dir_ok = (scan_vx * gw_sign < 0) & ~np.isnan(scan_vx)

                    hd_raw = np.abs(scan_heading - pre_hdg)
                    hd_wrapped = (hd_raw + np.pi) % (2 * np.pi) - np.pi
                    hd_deg = np.degrees(np.abs(hd_wrapped))
                    hdg_ok = (hd_deg >= reorient_thresh) & ~np.isnan(scan_heading)

                    all_ok = speed_ok & dir_ok & hdg_ok
                    idx = int(np.argmax(all_ok))
                    if all_ok[idx]:
                        reaction_frame = int(scan_frames[idx])
                        react_spd = float(scan_vmag[idx])
                        react_hdg = float(scan_heading[idx])

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
    # Build frame→block mapping using numpy array for O(1) vectorised lookup
    # instead of Python dict + for f in range(...) + lambda .map().
    if not blocks:
        return pd.DataFrame({c: pd.Series(dtype="str" if c in ("game_id","block_id","signal_name") else
                                          "int" if c in ("phase","player_id","team_id_opta","n_frames") else "float")
                             for c in ["game_id","block_id","phase","player_id","team_id_opta","signal_name","signal_value","n_frames"]})

    all_start = min(b["start_frame"] for b in blocks)
    all_end = max(b["end_frame"] for b in blocks)
    n_total_frames = all_end - all_start + 1

    # Build block_id lookup as an object array indexed by (frame - all_start)
    # Only used if there are valid frames to map
    valid = latency_df[latency_df["valid"]]
    if len(valid) == 0:
        return pd.DataFrame({c: pd.Series(dtype="str" if c in ("game_id","block_id","signal_name") else
                                          "int" if c in ("phase","player_id","team_id_opta","n_frames") else "float")
                             for c in ["game_id","block_id","phase","player_id","team_id_opta","signal_name","signal_value","n_frames"]})

    # Pre-allocate lookup arrays for O(1) frame→block mapping
    bid_lookup = np.empty(n_total_frames, dtype=object)
    phase_lookup = np.full(n_total_frames, -1, dtype=np.int64)
    bid_lookup[:] = None

    for blk in blocks:
        start = blk["start_frame"] - all_start
        end = blk["end_frame"] - all_start
        bid_lookup[start:end + 1] = blk["block_id"]
        phase_lookup[start:end + 1] = blk["phase"]

    # Also build a fast block_id → n_frames lookup
    block_nframes = {blk["block_id"]: blk.get("end_frame", 0) - blk.get("start_frame", 0)
                     for blk in blocks}

    idx_arr = valid["trigger_frame"].values.astype(np.int64) - all_start
    in_range = (idx_arr >= 0) & (idx_arr < n_total_frames)

    valid = valid.copy()
    valid["block_id"] = None
    valid["phase"] = -1
    if in_range.any():
        valid.loc[in_range, "block_id"] = bid_lookup[idx_arr[in_range]]
        valid.loc[in_range, "phase"] = phase_lookup[idx_arr[in_range]]

    valid = valid.dropna(subset=["block_id"])
    if len(valid) == 0:
        return pd.DataFrame({c: pd.Series(dtype="str" if c in ("game_id","block_id","signal_name") else
                                          "int" if c in ("phase","player_id","team_id_opta","n_frames") else "float")
                             for c in ["game_id","block_id","phase","player_id","team_id_opta","signal_name","signal_value","n_frames"]})

    agg = valid.groupby(["block_id", "phase", "player_id", "team_id_opta"], sort=False).agg(
        mean_reaction_time=("reaction_time_s", "mean"),
        p90_reaction_time=("reaction_time_s", lambda x: x.quantile(0.90)),
        n_triggers=("trigger_id", "nunique")).reset_index()

    records = []
    for _, row in agg.iterrows():
        bid = row["block_id"]
        bf = block_nframes.get(bid, 0)
        records.append({"game_id": game_id, "block_id": bid, "phase": int(row["phase"]),
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
        t0 = time.time()
        cfg = self.shift_config

        t_sub = time.time()
        trigger_df = detect_all_triggers(match_df, cfg)
        n_triggers = len(trigger_df)
        t_sub = _log_timing("detect_all_triggers", t_sub)

        if len(trigger_df) == 0:
            _log_timing(f"no triggers, total", t0)
            return pd.DataFrame(columns=["game_id","block_id","phase","player_id","team_id_opta","signal_name","signal_value","n_frames"])

        latency_df = compute_shift_reaction_time(match_df, trigger_df, cfg, own_goal_direction)
        # Count how many players were scanned (unique player_ids in latency_df)
        n_players_scanned = latency_df["player_id"].nunique() if len(latency_df) > 0 else 0
        _log_timing(f"compute_shift_reaction_time ({n_triggers} triggers, {n_players_scanned} players)", t_sub)

        result = aggregate_shift_latency_by_block(latency_df, blocks, cfg, game_id=game_id)
        _log_timing("aggregate_shift_latency_by_block", t_sub)

        _log_timing(f"total ({len(result)} rows)", t0)
        return result

    def validate(self, output_df):
        super().validate(output_df)
        if len(output_df) == 0: return True
        sv = output_df["signal_value"]
        if sv.min() < 0: raise ValueError(f"signal_value negative (min={sv.min():.3f})")
        if sv.max() > self.shift_config.max_reaction_time_s:
            raise ValueError(f"signal_value exceeds max_reaction_time_s")
        return True
