"""Signal 1 — Positional Drift.

Measures how far defenders drift from their expected (shape-model)
positions during out-of-possession phases. The core hypothesis is that
cognitively fatigued defenders exhibit degraded spatial awareness,
manifesting as larger positional errors relative to the ideal role
centroid provided by the shape.json model.

This file contains: shape file parsing and team mapping bridge,
per-frame drift computation, block aggregation, the signal class,
and configuration — all in one file.
"""

# ── Config ──────────────────────────────────────────────────────────────────

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class DriftConfig:
    shape_window_s: float = 60.0
    min_fit_score: float = 0.5
    frames_per_second: int = 25
    max_plausible_drift_m: float = 50.0

    @property
    def frames_per_minute(self) -> int:
        return self.frames_per_second * 60

DEFAULT_DRIFT_CONFIG = DriftConfig()

# ── Imports ─────────────────────────────────────────────────────────────────

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .base import SignalBase
from .config import DEFAULT_SIGNAL_CONFIG
from .registry import register_signal

# ── Coordinate conversion ──────────────────────────────────────────────────

_PITCH_LENGTH = 116.0
_PITCH_WIDTH = 78.0
_TEAM_MAPPINGS_PATH = Path("./data/raw/team_mappings/team_mappings.csv")

def _shape_x(v): return v / 100.0 * _PITCH_LENGTH - 57.0
def _shape_y(v): return v / 100.0 * _PITCH_WIDTH - 39.0


# ═══════════════════════════════════════════════════════════════════════════
# Team Mapping
# ═══════════════════════════════════════════════════════════════════════════

def load_team_mappings(path: Union[str, Path] = "") -> tuple[dict[str, int], dict[int, str]]:
    """Load UUID→OptaID and OptaID→UUID from team_mappings.csv."""
    path = Path(path) if path else Path(_TEAM_MAPPINGS_PATH)
    uuid_to_opta: dict[str, int] = {}
    opta_to_uuid: dict[int, str] = {}
    if not path.exists():
        return uuid_to_opta, opta_to_uuid
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            u = row.get("uuid", "").strip()
            o = row.get("opta_id", "").strip()
            if u and o:
                oid = int(o)
                uuid_to_opta[u] = oid
                opta_to_uuid[oid] = u
    return uuid_to_opta, opta_to_uuid


# ═══════════════════════════════════════════════════════════════════════════
# Shape Data Structures
# ═══════════════════════════════════════════════════════════════════════════

class ShapeRole:
    __slots__ = ("role_id", "role_name", "jersey_no", "player_uuid", "avg_x", "avg_y", "fit_score")
    def __init__(self, role_id="", role_name="", jersey_no=0, player_uuid="",
                 avg_x=0.0, avg_y=0.0, fit_score=0.0):
        self.role_id = role_id
        self.role_name = role_name
        self.jersey_no = jersey_no
        self.player_uuid = player_uuid
        self.avg_x = avg_x
        self.avg_y = avg_y
        self.fit_score = fit_score


class ShapeWindow:
    __slots__ = ("minute_window", "at_time", "phase", "formation_out", "formation_in",
                 "out_of_possession", "in_possession", "team_uuid")
    def __init__(self, minute_window=0, at_time="", phase=1, formation_out="",
                 formation_in="", out_of_possession=None, in_possession=None, team_uuid=""):
        self.minute_window = minute_window
        self.at_time = at_time
        self.phase = phase
        self.formation_out = formation_out
        self.formation_in = formation_in
        self.out_of_possession = out_of_possession or []
        self.in_possession = in_possession or []
        self.team_uuid = team_uuid


# ═══════════════════════════════════════════════════════════════════════════
# Shape File Parsing (auto-detects V1/V2 format)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_iso_timestamp(ts: str) -> Optional[datetime]:
    for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"]:
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _compute_minute_window(at_time: str, match_start: Optional[datetime], phase_num: int) -> int:
    ts = _parse_iso_timestamp(at_time)
    if ts is not None and match_start is not None:
        return max(0, int(ts.total_seconds() // 60))
    return 0 if phase_num == 1 else 45


def _parse_v2_shapes(raw: dict) -> list[ShapeWindow]:
    windows: list[ShapeWindow] = []
    periods = raw.get("periods", [])
    if not periods:
        return windows
    match_start = None
    first_start = periods[0].get("startTime", "")
    if first_start:
        match_start = _parse_iso_timestamp(first_start)
    for period in periods:
        phase_num = period.get("periodNumber", 1)
        for shape_entry in period.get("shapes", []):
            at_time = shape_entry.get("atTime", "")
            mw = _compute_minute_window(at_time, match_start, phase_num)
            def _parse_roles(raw_roles):
                return [ShapeRole(
                    role_id=r.get("roleId", ""),
                    role_name=r.get("roleDisplayName", r.get("roleId", "")),
                    jersey_no=int(r.get("jerseyNo", 0)),
                    player_uuid=r.get("playerUuid", ""),
                    avg_x=_shape_x(float(r.get("averageRolePositionX", 0.0))),
                    avg_y=_shape_y(float(r.get("averageRolePositionY", 0.0))),
                    fit_score=float(r.get("fitScore", 0.0)),
                ) for r in raw_roles]
            oop = shape_entry.get("outOfPossession", {})
            ip = shape_entry.get("inPossession", {})
            windows.append(ShapeWindow(
                minute_window=mw, at_time=at_time, phase=phase_num,
                formation_out=oop.get("formation", ""),
                formation_in=ip.get("formation", ""),
                out_of_possession=_parse_roles(oop.get("roles", [])),
                in_possession=_parse_roles(ip.get("roles", [])),
            ))
    windows.sort(key=lambda w: w.minute_window)
    return windows


def _parse_v1_shapes(raw: dict) -> list[ShapeWindow]:
    windows: list[ShapeWindow] = []
    for entry in raw.get("liveData", {}).get("shapes", []):
        period_id = int(entry.get("periodId", 1))
        for contestant in entry.get("contestant", []):
            cid = contestant.get("id", "")
            for state_key in ("inPossession", "outOfPossession"):
                state_data = contestant.get(state_key, {})
                for shape_entry in state_data.get("shape", []):
                    ps = shape_entry.get("periodStart", "00:00")
                    minute_start = int(ps.split(":")[0]) + (period_id - 1) * 45
                    roles_raw = shape_entry.get("shapeRole", {}).get("role", [])
                    roles = [ShapeRole(
                        role_id=r.get("id", ""),
                        role_name=r.get("roleDescription", ""),
                        jersey_no=int(r.get("shirtNumber", 0)),
                        player_uuid=r.get("playerId", ""),
                        avg_x=_shape_x(float(r.get("averageRolePositionX", 0.0))),
                        avg_y=_shape_y(float(r.get("averageRolePositionY", 0.0))),
                        fit_score=float(r.get("fitScore", 0.0)),
                    ) for r in roles_raw]
                    oop = roles if state_key == "outOfPossession" else []
                    ip = roles if state_key == "inPossession" else []
                    windows.append(ShapeWindow(
                        minute_window=minute_start, at_time="", phase=period_id,
                        formation_out=shape_entry.get("formation", "") if state_key == "outOfPossession" else "",
                        formation_in=shape_entry.get("formation", "") if state_key == "inPossession" else "",
                        out_of_possession=oop, in_possession=ip, team_uuid=cid,
                    ))
    windows.sort(key=lambda w: w.minute_window)
    return windows


def load_shape_file(shape_path: Union[str, Path]) -> dict[str, Any]:
    """Load and parse a shape.json file into structured dict.

    Returns dict with keys: match_id, date, competition,
    home_name, away_name, home_uuid, away_uuid, start_time,
    team_uuid_map, shapes_by_time, windows.
    """
    shape_path = Path(shape_path)
    with open(shape_path) as f:
        raw = json.load(f)
    mi = raw.get("matchInfo", {})
    contestants = mi.get("contestant", [])
    home = next((c for c in contestants if c.get("position") == "home"), {})
    away = next((c for c in contestants if c.get("position") == "away"), {})
    home_uuid = home.get("id", "")
    away_uuid = away.get("id", "")
    team_uuid_map = {}
    if home_uuid: team_uuid_map[home_uuid] = home.get("name", "")
    if away_uuid: team_uuid_map[away_uuid] = away.get("name", "")
    match_start = None
    periods = raw.get("periods", [])
    if periods and periods[0].get("startTime"):
        match_start = _parse_iso_timestamp(periods[0]["startTime"])
    v2 = _parse_v2_shapes(raw)
    windows = v2 if v2 else _parse_v1_shapes(raw)
    shapes_by_time: dict[int, list[ShapeWindow]] = {}
    for w in windows:
        shapes_by_time.setdefault(w.minute_window, []).append(w)
    return {
        "match_id": mi.get("id", shape_path.stem),
        "date": mi.get("date", ""),
        "competition": mi.get("competition", {}).get("name", ""),
        "home_name": home.get("name", ""),
        "away_name": away.get("name", ""),
        "home_uuid": home_uuid,
        "away_uuid": away_uuid,
        "start_time": match_start,
        "team_uuid_map": team_uuid_map,
        "shapes_by_time": shapes_by_time,
        "windows": windows,
    }


def build_player_role_map(
    tracking_df: pd.DataFrame,
    shapes: dict[str, Any],
    team_mappings_path: Union[str, Path] = "",
    min_fit_score: float = 0.5,
) -> dict[int, dict[int, dict[str, Any]]]:
    """Map tracking player_id → shape role per minute window.

    Returns {player_id: {minute: {role, role_name, expected_x, expected_y, fit_score, formation, phase}}}
    """
    uuid_to_opta, opta_to_uuid = load_team_mappings(team_mappings_path)
    jersey_col = "jersey_number" if "jersey_number" in tracking_df.columns else \
                 "jersey_no" if "jersey_no" in tracking_df.columns else None
    if jersey_col is None:
        raise ValueError("tracking_df must contain 'jersey_number' or 'jersey_no'")
    players = (tracking_df[["player_id", "team_id_opta", jersey_col]]
               .drop_duplicates(subset=["player_id"])
               .dropna(subset=["player_id", "team_id_opta", jersey_col])
               .astype({"player_id": int, "team_id_opta": int, jersey_col: int}))
    windows_by_minute: dict[int, list[ShapeWindow]] = {}
    for w in shapes.get("windows", []):
        windows_by_minute.setdefault(w.minute_window, []).append(w)
    role_map: dict[int, dict[int, dict[str, Any]]] = {}
    for _, prow in players.iterrows():
        pid, team_opta, jersey = int(prow["player_id"]), int(prow["team_id_opta"]), int(prow[jersey_col])
        team_uuid = opta_to_uuid.get(team_opta)
        if team_uuid is None:
            continue
        role_map[pid] = {}
        for minute_win, win_list in windows_by_minute.items():
            team_wins = [w for w in win_list if w.team_uuid == team_uuid]
            candidates = team_wins if team_wins else win_list
            matched = None; formation = ""; phase = 0
            for win in candidates:
                for role in win.out_of_possession:
                    if role.jersey_no == jersey and role.fit_score >= min_fit_score:
                        matched = role; formation = win.formation_out; phase = win.phase; break
                if matched: break
                for role in win.in_possession:
                    if role.jersey_no == jersey and role.fit_score >= min_fit_score:
                        matched = role; formation = win.formation_in; phase = win.phase; break
                if matched: break
            if matched:
                role_map[pid][minute_win] = {
                    "role": matched.role_id, "role_name": matched.role_name,
                    "expected_x": matched.avg_x, "expected_y": matched.avg_y,
                    "fit_score": matched.fit_score, "formation": formation, "phase": phase,
                }
    return role_map


# ═══════════════════════════════════════════════════════════════════════════
# Drift Computation
# ═══════════════════════════════════════════════════════════════════════════

def vectorise_expected_positions(
    df: pd.DataFrame,
    player_role_map: dict[int, dict[int, dict[str, Any]]],
    frames_per_minute: int = 1500,
    allow_forward_fill: bool = True,
) -> pd.DataFrame:
    if "player_id" not in df.columns or "frame" not in df.columns:
        raise ValueError("DataFrame must contain 'player_id' and 'frame' columns")
    out_x = np.full(len(df), np.nan, dtype=np.float64)
    out_y = np.full(len(df), np.nan, dtype=np.float64)
    out_fit = np.full(len(df), np.nan, dtype=np.float64)
    out_role = np.full(len(df), "", dtype=object)
    for pid, idxs in df.groupby("player_id", sort=False).indices.items():
        if pid not in player_role_map:
            continue
        pw = player_role_map[pid]
        if not pw:
            continue
        sm = sorted(pw.keys())
        min_m = sm[0]
        mws = df["frame"].iloc[idxs].values // frames_per_minute
        for li, mw in enumerate(mws):
            gi = idxs[li]
            if mw in pw:
                e = pw[mw]
                out_x[gi] = e["expected_x"]; out_y[gi] = e["expected_y"]
                out_fit[gi] = e["fit_score"]; out_role[gi] = e["role"]
            elif allow_forward_fill and mw >= min_m:
                for m in reversed(sm):
                    if m <= mw:
                        e = pw[m]
                        out_x[gi] = e["expected_x"]; out_y[gi] = e["expected_y"]
                        out_fit[gi] = e["fit_score"]; out_role[gi] = e["role"]
                        break
    df = df.copy()
    df["expected_x"] = out_x; df["expected_y"] = out_y
    df["fit_score"] = out_fit; df["shape_role"] = out_role
    return df


def compute_drift(
    df: pd.DataFrame,
    player_role_map: dict[int, dict[int, dict[str, Any]]],
    config: DriftConfig,
    team_in_possession_col: str = "team_in_possession",
    team_id_col: str = "team_id_opta",
) -> pd.DataFrame:
    """Compute positional drift per frame per player (out-of-possession only)."""
    result = vectorise_expected_positions(df, player_role_map, config.frames_per_minute)
    has_exp = result["expected_x"].notna() & result["expected_y"].notna()
    result["drift_m"] = np.where(has_exp,
        np.sqrt((result["x"] - result["expected_x"])**2 + (result["y"] - result["expected_y"])**2), np.nan)
    # Mask to out-of-possession only
    if team_in_possession_col in result.columns and team_id_col in result.columns:
        ball_in = result[team_in_possession_col].notna()
        result["drift_m"] = result["drift_m"].where(ball_in & (result[team_id_col] != result[team_in_possession_col]), np.nan)
    result.loc[result["drift_m"] > config.max_plausible_drift_m, "drift_m"] = np.nan
    low_fit = result["fit_score"].notna() & (result["fit_score"] < config.min_fit_score)
    result.loc[low_fit, "drift_m"] = np.nan
    return result


def aggregate_drift_by_block(
    df: pd.DataFrame, blocks: list[pd.DataFrame], config: DriftConfig,
    game_id: str = "", valid_only: bool = True,
) -> pd.DataFrame:
    """Aggregate drift per block per player into standard signal output."""
    records: list[dict[str, Any]] = []
    for blk in blocks:
        bid = str(blk["block_id"].iloc[0])
        phase = int(bid.split("_")[0])
        start, end = int(blk["frame"].min()), int(blk["frame"].max())
        bmask = df["frame"].between(start, end, inclusive="left")
        if not bmask.any(): continue
        bdf = df.loc[bmask]
        for _, pr in (bdf[["player_id", "team_id_opta"]].drop_duplicates(subset="player_id").dropna(subset=["player_id"]).iterrows()):
            pid = int(pr["player_id"]); to = int(pr["team_id_opta"])
            dv = bdf.loc[bdf["player_id"] == pid, "drift_m"].dropna()
            nv = len(dv)
            if nv == 0 and valid_only: continue
            md = float(dv.mean()) if nv > 0 else np.nan
            p90 = float(dv.quantile(0.90)) if nv > 0 else np.nan
            mx = float(dv.max()) if nv > 0 else np.nan
            sd = float(dv.std()) if nv > 0 else np.nan
            fs = bdf.loc[bdf["player_id"] == pid, "fit_score"].dropna()
            mf = float(fs.mean()) if len(fs) > 0 else np.nan
            records.append({
                "game_id": game_id, "block_id": bid, "phase": phase,
                "player_id": pid, "team_id_opta": to,
                "signal_name": "positional_drift", "signal_value": md, "n_frames": nv,
                "drift_p90": p90, "drift_max": mx, "drift_std": sd, "mean_fit_score": mf,
            })
    if not records:
        return pd.DataFrame(columns=["game_id", "block_id", "phase", "player_id", "team_id_opta",
                                     "signal_name", "signal_value", "n_frames",
                                     "drift_p90", "drift_max", "drift_std", "mean_fit_score"])
    out = pd.DataFrame(records)
    for c in ["player_id", "team_id_opta", "phase", "n_frames"]: out[c] = out[c].astype(int)
    for c in ["game_id", "block_id", "signal_name"]: out[c] = out[c].astype(str)
    return out.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Registered Signal Class
# ═══════════════════════════════════════════════════════════════════════════

@register_signal
class PositionalDriftSignal(SignalBase):
    signal_name = "positional_drift"

    def __init__(self, signal_config=None, drift_config=None, logger=None):
        super().__init__(config=signal_config, logger=logger)
        self.drift_config = drift_config or DEFAULT_DRIFT_CONFIG

    def compute(self, match_df, blocks, *, game_id="", shape_path="",
                player_role_map=None, team_id_col="team_id_opta",
                team_in_possession_col="team_in_possession"):
        cfg = self.drift_config
        if player_role_map is None:
            if not shape_path:
                raise ValueError("shape_path required when player_role_map not provided")
            shapes = load_shape_file(shape_path)
            player_role_map = build_player_role_map(match_df, shapes, min_fit_score=cfg.min_fit_score)
        if not player_role_map:
            return pd.DataFrame(columns=["game_id", "block_id", "phase", "player_id",
                                         "team_id_opta", "signal_name", "signal_value", "n_frames"])
        drift_df = compute_drift(match_df, player_role_map, cfg, team_in_possession_col, team_id_col)
        return aggregate_drift_by_block(drift_df, blocks, cfg, game_id=game_id)

    def validate(self, output_df):
        super().validate(output_df)
        if len(output_df) == 0: return True
        sv = output_df["signal_value"]
        if sv.min() < 0:
            raise ValueError(f"signal_value (mean drift) contains negative values (min={sv.min():.3f})")
        if sv.max() > self.drift_config.max_plausible_drift_m:
            raise ValueError(f"signal_value exceeds max_plausible_drift_m={self.drift_config.max_plausible_drift_m}")
        return True
