"""Player-to-shape-role bridge for positional drift.

Connects tracking-data players (identified by ``team_id_opta`` +
``jersey_number``) to shape-json roles (identified by UUID continent
ID + jersey number) via ``team_mappings.csv``.

The bridge answers the question:

    "For player X at frame Y, where *should* they be according to
     the shape model?"

The shape.json file provides expected position centroids
(``averageRolePositionX/Y``) for each role on the pitch at ~1-minute
intervals. We match tracking players to these centroids by:

1.  Resolving the shape.json team UUID → tracking ``team_id_opta``
    using ``team_mappings.csv``.
2.  Matching the player's jersey number to a role within the shape
    entry for the relevant time window and possession state.
3.  Returning the centroid coordinates for that role.

Shape files can appear in two formats (auto-detected):

- **V1** (legacy, used by ``load_shapes.py``):
  ``data["liveData"]["shapes"][…]["contestant"]`` — shapes grouped
  into 5-minute windows with ``periodStart`` / ``periodEnd`` strings.

- **V2** (canonical, described in the task spec):
  ``data["periods"][…]["shapes"][…]`` — individual shape entries at
  1-minute resolution with ``atTime`` ISO timestamps.

See Also
--------
``src.loaders.load_shapes`` — An alternative parser for V1 shape files.
``src.loaders.team_names`` — UUID ↔ Opta ID and team name utilities.
"""

from __future__ import annotations

# ── Coordinate Conversion ──────────────────────────────────
# Shape.json uses 0-100 coords; tracking uses centred metres
_PITCH_LENGTH = 116.0  # -57 to +59
_PITCH_WIDTH = 78.0    # -39 to +39

def _shape_x(v): return v / 100.0 * _PITCH_LENGTH - 57.0
def _shape_y(v): return v / 100.0 * _PITCH_WIDTH - 39.0


import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

# ── Default Paths ───────────────────────────────────────────────────────────

_TEAM_MAPPINGS_PATH = Path(
    "./data/raw/team_mappings/team_mappings.csv"
)


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ShapeRole:
    """A single role entry from a shape file.

    Attributes
    ----------
    role_id : str
        Identifier for the role (e.g. ``"CB"``, ``"FB"``).
    role_name : str
        Human-readable role description.
    jersey_no : int
        Player shirt number at the time of the shape entry.
    player_uuid : str
        Stats Perform UUID for the player occupying this role.
    avg_x : float
        Expected x-coordinate (metres, DOP-normalised).
    avg_y : float
        Expected y-coordinate (metres, DOP-normalised).
    fit_score : float
        Quality-of-fit score for this role assignment (0–1).
    """

    role_id: str
    role_name: str
    jersey_no: int
    player_uuid: str
    avg_x: float
    avg_y: float
    fit_score: float


@dataclass
class ShapeWindow:
    """Shape data for a single time window.

    Attributes
    ----------
    minute_window : int
        Minute of the match this shape entry applies to (0-based).
    at_time : str
        ISO timestamp of the shape capture.
    phase : int
        Match period (1 = first half, 2 = second half).
    formation_out : str
        Formation label when out of possession (e.g. ``"4-4-2"``).
    formation_in : str
        Formation label when in possession.
    out_of_possession : list[ShapeRole]
        Roles for the out-of-possession team state.
    in_possession : list[ShapeRole]
        Roles for the in-possession team state.
    team_uuid : str
        UUID of the team this shape window belongs to.
    """

    minute_window: int
    at_time: str
    phase: int
    formation_out: str
    formation_in: str
    out_of_possession: list[ShapeRole]
    in_possession: list[ShapeRole]
    team_uuid: str = ""


@dataclass
class ShapeMatch:
    """Parsed shape file for a single match.

    Attributes
    ----------
    match_id : str
        Match identifier from the shape file.
    home_name : str
        Home team name.
    away_name : str
        Away team name.
    home_uuid : str
        Home team UUID.
    away_uuid : str
        Away team UUID.
    start_time : datetime | None
        Match epoch start time (UTC), if available.
    windows : list[ShapeWindow]
        All shape windows, sorted by minute_window.
    """

    match_id: str
    home_name: str
    away_name: str
    home_uuid: str
    away_uuid: str
    start_time: Optional[datetime]
    windows: list[ShapeWindow]


# ═══════════════════════════════════════════════════════════════════════════
# Team Mapping Helpers
# ═══════════════════════════════════════════════════════════════════════════


def load_team_mappings(
    path: Union[str, Path] = "",
) -> tuple[dict[str, int], dict[int, str]]:
    """Load UUID → Opta ID and Opta ID → UUID mappings from CSV.

    Parameters
    ----------
    path : str or Path
        Path to ``team_mappings.csv``. Falls back to the default path
        at ``_TEAM_MAPPINGS_PATH`` if empty or missing.

    Returns
    -------
    uuid_to_opta : dict[str, int]
        Map of UUID string → Opta team ID integer.
    opta_to_uuid : dict[int, str]
        Reverse map of Opta team ID → UUID string.
    """
    path = Path(path) if path else Path(_TEAM_MAPPINGS_PATH)

    uuid_to_opta: dict[str, int] = {}
    opta_to_uuid: dict[int, str] = {}

    if not path.exists():
        return uuid_to_opta, opta_to_uuid

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uuid_val = row.get("uuid", "").strip()
            opta_str = row.get("opta_id", "").strip()
            if uuid_val and opta_str:
                opta_id = int(opta_str)
                uuid_to_opta[uuid_val] = opta_id
                opta_to_uuid[opta_id] = uuid_val

    return uuid_to_opta, opta_to_uuid


# ═══════════════════════════════════════════════════════════════════════════
# Shape File Parser
# ═══════════════════════════════════════════════════════════════════════════


def _parse_iso_timestamp(ts: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp string to a UTC datetime.

    Returns ``None`` if parsing fails.
    """
    for fmt in [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
    ]:
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _compute_minute_window(
    at_time: str,
    match_start: Optional[datetime],
    phase_num: int,
) -> int:
    """Compute the match minute window from an ISO timestamp.

    Parameters
    ----------
    at_time : str
        ISO timestamp of the shape entry.
    match_start : datetime or None
        Match start time (UTC). If None, returns 0 for first period
        entries and 45 for second period entries (fallback).
    phase_num : int
        Match period (1 or 2).

    Returns
    -------
    int
        Minute window into the match (0-based).
    """
    ts = _parse_iso_timestamp(at_time)
    if ts is not None and match_start is not None:
        delta = ts - match_start
        return max(0, int(delta.total_seconds() // 60))
    # Fallback: approximate by phase
    return 0 if phase_num == 1 else 45


def _parse_v2_shapes(raw: dict) -> list[ShapeWindow]:
    """Parse the V2 shape format (``periods[].shapes[].atTime``).

    Parameters
    ----------
    raw : dict
        Raw parsed JSON from the shape file.

    Returns
    -------
    list[ShapeWindow]
        Parsed shape windows sorted by minute.
    """
    windows: list[ShapeWindow] = []
    periods = raw.get("periods", [])

    if not periods:
        return windows

    # Determine match start time from first period
    match_start: Optional[datetime] = None
    first_period = periods[0]
    start_raw = first_period.get("startTime", "")
    if start_raw:
        match_start = _parse_iso_timestamp(start_raw)

    for period in periods:
        phase_num = period.get("periodNumber", 1)
        shapes = period.get("shapes", [])

        for shape_entry in shapes:
            at_time = shape_entry.get("atTime", "")
            minute_window = _compute_minute_window(
                at_time, match_start, phase_num
            )

            # --- Out of possession ---
            oop = shape_entry.get("outOfPossession", {})
            oop_roles_raw = oop.get("roles", [])
            oop_roles = [
                ShapeRole(
                    role_id=r.get("roleId", ""),
                    role_name=r.get("roleDisplayName", r.get("roleId", "")),
                    jersey_no=int(r.get("jerseyNo", 0)),
                    player_uuid=r.get("playerUuid", ""),
                    avg_x=_shape_x(float(r.get("averageRolePositionX", 0.0))),
                    avg_y=_shape_y(float(r.get("averageRolePositionY", 0.0))),
                    fit_score=float(r.get("fitScore", 0.0)),
                )
                for r in oop_roles_raw
            ]

            # --- In possession ---
            ip = shape_entry.get("inPossession", {})
            ip_roles_raw = ip.get("roles", [])
            ip_roles = [
                ShapeRole(
                    role_id=r.get("roleId", ""),
                    role_name=r.get("roleDisplayName", r.get("roleId", "")),
                    jersey_no=int(r.get("jerseyNo", 0)),
                    player_uuid=r.get("playerUuid", ""),
                    avg_x=_shape_x(float(r.get("averageRolePositionX", 0.0))),
                    avg_y=_shape_y(float(r.get("averageRolePositionY", 0.0))),
                    fit_score=float(r.get("fitScore", 0.0)),
                )
                for r in ip_roles_raw
            ]

            windows.append(
                ShapeWindow(
                    minute_window=minute_window,
                    at_time=at_time,
                    phase=phase_num,
                    formation_out=oop.get("formation", ""),
                    formation_in=ip.get("formation", ""),
                    out_of_possession=oop_roles,
                    in_possession=ip_roles,
                )
            )

    # Sort by minute window
    windows.sort(key=lambda w: w.minute_window)
    return windows


def _parse_v1_contestant_shapes(
    contestant: dict,
    period_id: int,
    team_uuid: str,
) -> list[ShapeWindow]:
    """Parse V1 shape entries for a single contestant with their team UUID.

    Parameters
    ----------
    contestant : dict
        A single contestant entry from ``liveData.shapes[].contestant[]``.
    period_id : int
        Match period (1 or 2).
    team_uuid : str
        UUID of this contestant's team.

    Returns
    -------
    list[ShapeWindow]
        Parsed shape windows for this contestant with team_uuid set.
    """
    windows: list[ShapeWindow] = []
    for state_key in ("inPossession", "outOfPossession"):
        state_data = contestant.get(state_key, {})
        shape_list = state_data.get("shape", [])

        for shape_entry in shape_list:
            period_start = shape_entry.get("periodStart", "00:00")
            # Parse minutes from MM:SS
            parts_s = period_start.split(":")
            minute_start = int(parts_s[0]) + (period_id - 1) * 45

            roles_raw = (
                shape_entry.get("shapeRole", {}).get("role", [])
            )

            roles = [
                ShapeRole(
                    role_id=r.get("id", ""),
                    role_name=r.get("roleDescription", ""),
                    jersey_no=int(r.get("shirtNumber", 0)),
                    player_uuid=r.get("playerId", ""),
                    avg_x=_shape_x(float(r.get("averageRolePositionX", 0.0))),
                    avg_y=_shape_y(float(r.get("averageRolePositionY", 0.0))),
                    fit_score=float(r.get("fitScore", 0.0)),
                )
                for r in roles_raw
            ]

            if state_key == "outOfPossession":
                oop_roles, ip_roles = roles, []
            else:
                oop_roles, ip_roles = [], roles

            windows.append(
                ShapeWindow(
                    minute_window=minute_start,
                    at_time="",
                    phase=period_id,
                    formation_out=(
                        shape_entry.get("formation", "")
                        if state_key == "outOfPossession" else ""
                    ),
                    formation_in=(
                        shape_entry.get("formation", "")
                        if state_key == "inPossession" else ""
                    ),
                    out_of_possession=oop_roles,
                    in_possession=ip_roles,
                    team_uuid=team_uuid,
                )
            )
    return windows


def _parse_v1_shapes(raw: dict, match_start: Optional[datetime]) -> list[ShapeWindow]:
    """Parse the V1 shape format (``liveData.shapes[].contestant[]``).

    This format groups shapes into 5-minute windows and uses
    ``periodStart`` / ``periodEnd`` strings.

    Parameters
    ----------
    raw : dict
        Raw parsed JSON from the shape file.
    match_start : datetime or None
        Match start time (UTC), if available.

    Returns
    -------
    list[ShapeWindow]
        Parsed shape windows sorted by minute, with team_uuid populated.
    """
    windows: list[ShapeWindow] = []
    shapes = raw.get("liveData", {}).get("shapes", [])

    if not shapes:
        return windows

    for entry in shapes:
        period_id = int(entry.get("periodId", 1))
        contestant_list = entry.get("contestant", [])

        if not contestant_list:
            continue

        for contestant in contestant_list:
            cid = contestant.get("id", "")
            # Map contestant id to team UUID
            # In V1 format, the contestant id in liveData.shapes IS the team UUID
            team_uuid = cid
            windows.extend(
                _parse_v1_contestant_shapes(contestant, period_id, team_uuid)
            )

    windows.sort(key=lambda w: w.minute_window)
    return windows


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def load_shape_file(shape_path: Union[str, Path]) -> dict[str, Any]:
    """Load and parse a shape.json file.

    Parameters
    ----------
    shape_path : str or Path
        Path to the shape.json file.

    Returns
    -------
    dict
        A structured dictionary with the following keys:

        - **match_id** (*str*) — Match identifier from the shape file.
        - **date** (*str*) — Match date.
        - **home_name** (*str*) — Home team name.
        - **away_name** (*str*) — Away team name.
        - **home_uuid** (*str*) — Home team UUID.
        - **away_uuid** (*str*) — Away team UUID.
        - **start_time** (*datetime | None*) — Match start time (UTC).
        - **team_uuid_map** (*dict[str, str]*) — UUID → team name.
        - **shapes_by_time** (*dict[int, list[ShapeWindow]]*) — Shape
          entries keyed by minute window.
        - **windows** (*list[ShapeWindow]*) — All shape windows sorted
          by minute.
    """
    shape_path = Path(shape_path)

    with open(shape_path) as f:
        raw = json.load(f)

    mi = raw.get("matchInfo", {})

    match_id = mi.get("id", shape_path.stem)
    date = mi.get("date", "")
    competition = mi.get("competition", {}).get("name", "")

    # Extract team info
    contestants = mi.get("contestant", [])
    home = next((c for c in contestants if c.get("position") == "home"), {})
    away = next((c for c in contestants if c.get("position") == "away"), {})

    home_name = home.get("name", "")
    away_name = away.get("name", "")
    home_uuid = home.get("id", "")
    away_uuid = away.get("id", "")

    team_uuid_map: dict[str, str] = {}
    if home_uuid:
        team_uuid_map[home_uuid] = home_name
    if away_uuid:
        team_uuid_map[away_uuid] = away_name

    # Determine the match start time
    match_start: Optional[datetime] = None
    periods = raw.get("periods", [])
    if periods:
        start_raw = periods[0].get("startTime", "")
        if start_raw:
            match_start = _parse_iso_timestamp(start_raw)

    # Parse shape windows — auto-detect format
    v2_windows = _parse_v2_shapes(raw)
    if v2_windows:
        windows = v2_windows
    else:
        windows = _parse_v1_shapes(raw, match_start)

    # Build shapes_by_time lookup: minute_window → list[ShapeWindow]
    shapes_by_time: dict[int, list[ShapeWindow]] = {}
    for win in windows:
        shapes_by_time.setdefault(win.minute_window, []).append(win)

    return {
        "match_id": match_id,
        "date": date,
        "competition": competition,
        "home_name": home_name,
        "away_name": away_name,
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
    """Build a mapping from tracking player_id → shape role per minute.

    The bridge works by:

    1. Extracting the set of unique players from the tracking DataFrame,
       along with their ``team_id_opta`` and ``jersey_number``.
    2. Mapping ``team_id_opta`` → team UUID via ``team_mappings.csv``.
    3. Finding the matching team's shape entry for each minute window.
    4. Matching the player to a role via jersey number.
    5. Recording the expected position and fit score for each (player,
       minute_window) pair.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking DataFrame. Must contain columns:
        ``player_id``, ``team_id_opta``, ``jersey_number`` (or
        ``jersey_no``).
    shapes : dict
        The parsed shape dict from :func:`load_shape_file`.
    team_mappings_path : str or Path
        Path to ``team_mappings.csv``. If empty, uses the default.
    min_fit_score : float
        Minimum fit score to include a role. Roles below this are
        excluded (treated as unreliable).

    Returns
    -------
    dict[int, dict[int, dict[str, Any]]]
        A nested dict:

        .. code-block:: python

            {
                player_id: {
                    minute_window: {
                        "role": str,         # e.g. "CB", "RB"
                        "role_name": str,    # e.g. "Centre Back"
                        "expected_x": float, # DOP-normalised
                        "expected_y": float, # DOP-normalised
                        "fit_score": float,
                        "formation": str,    # e.g. "4-4-2"
                        "phase": int,
                    }
                }
            }
    """
    # ── Resolve team IDs ───────────────────────────────────────────────
    uuid_to_opta, opta_to_uuid = load_team_mappings(team_mappings_path)

    # ── Extract unique players ─────────────────────────────────────────
    jersey_col = (
        "jersey_number" if "jersey_number" in tracking_df.columns
        else "jersey_no" if "jersey_no" in tracking_df.columns
        else None
    )

    if jersey_col is None:
        raise ValueError(
            "tracking_df must contain 'jersey_number' or 'jersey_no' column"
        )

    players = (
        tracking_df[["player_id", "team_id_opta", jersey_col]]
        .drop_duplicates(subset=["player_id"])
        .dropna(subset=["player_id", "team_id_opta", jersey_col])
    )

    unique_players = players.astype(
        {"player_id": int, "team_id_opta": int, jersey_col: int}
    )

    # ── Build shape windows lookup by (team_uuid, minute) ──────────────
    # Group shape windows by team UUID + minute_window
    # Each window contains both out_of_possession and in_possession roles
    windows = shapes.get("windows", [])
    team_uuid_map = shapes.get("team_uuid_map", {})

    # Index windows by minute_window for quick lookup
    # {minute_window: [ShapeWindow, ...]}
    windows_by_minute: dict[int, list[ShapeWindow]] = {}
    for win in windows:
        windows_by_minute.setdefault(win.minute_window, []).append(win)

    # ── Build the role map ─────────────────────────────────────────────
    role_map: dict[int, dict[int, dict[str, Any]]] = {}

    for _, player_row in unique_players.iterrows():
        pid = int(player_row["player_id"])
        team_opta = int(player_row["team_id_opta"])
        jersey = int(player_row[jersey_col])

        # Look up team UUID
        team_uuid = opta_to_uuid.get(team_opta)

        if team_uuid is None:
            # Unknown team — can't bridge
            continue

        # Initialise player entry
        role_map[pid] = {}

        # For each minute window with shape data
        for minute_win, win_list in windows_by_minute.items():
            # Filter to shape windows belonging to this player's team.
            # Fallback for V2 shapes (team_uuid=""): try ALL windows for
            # this minute — jersey-number matching selects the right team.
            team_wins = [w for w in win_list if w.team_uuid == team_uuid]
            candidate_wins = team_wins if team_wins else win_list
            matched_role: Optional[ShapeRole] = None
            formation = ""
            phase = 0

            for win in candidate_wins:
                # Try out-of-possession roles first (primary use case)
                for role in win.out_of_possession:
                    if role.jersey_no == jersey and role.fit_score >= min_fit_score:
                        matched_role = role
                        formation = win.formation_out
                        phase = win.phase
                        break

                if matched_role is not None:
                    break

                # Fall back to in-possession roles
                for role in win.in_possession:
                    if role.jersey_no == jersey and role.fit_score >= min_fit_score:
                        matched_role = role
                        formation = win.formation_in
                        phase = win.phase
                        break

            if matched_role is not None:
                role_map[pid][minute_win] = {
                    "role": matched_role.role_id,
                    "role_name": matched_role.role_name,
                    "expected_x": matched_role.avg_x,
                    "expected_y": matched_role.avg_y,
                    "fit_score": matched_role.fit_score,
                    "formation": formation,
                    "phase": phase,
                }

    return role_map


def get_expected_position(
    player_id: int,
    frame_number: int,
    player_role_map: dict[int, dict[int, dict[str, Any]]],
    frames_per_minute: int = 1500,
    allow_forward_fill: bool = True,
) -> tuple[float, float, float, str]:
    """Get the expected position for a player at a given frame.

    Parameters
    ----------
    player_id : int
        The tracking player identifier.
    frame_number : int
        Frame number (0-based, at 25 fps).
    player_role_map : dict
        Mapping built by :func:`build_player_role_map`.
    frames_per_minute : int
        Number of frames per minute (default 1500 = 25 fps × 60 s).
    allow_forward_fill : bool
        If ``True``, when no exact shape entry exists for the current
        minute, uses the most recent previous entry. If ``False``,
        returns NaN for missing windows.

    Returns
    -------
    expected_x : float
        Expected x-coordinate, or NaN if unavailable.
    expected_y : float
        Expected y-coordinate, or NaN if unavailable.
    fit_score : float
        Fit score for the matched shape entry, or NaN.
    role : str
        Role description (e.g. ``"CB"``), or ``""`` if unavailable.
    """
    DEFAULT_RETURN: tuple[float, float, float, str] = (
        np.nan, np.nan, np.nan, ""
    )

    if player_id not in player_role_map:
        return DEFAULT_RETURN

    player_windows = player_role_map[player_id]
    if not player_windows:
        return DEFAULT_RETURN

    minute_window = frame_number // frames_per_minute

    # Exact match
    if minute_window in player_windows:
        entry = player_windows[minute_window]
        return (
            entry["expected_x"],
            entry["expected_y"],
            entry["fit_score"],
            entry["role"],
        )

    if not allow_forward_fill:
        return DEFAULT_RETURN

    # Forward-fill: find most recent entry before this minute
    sorted_minutes = sorted(player_windows.keys())
    if not sorted_minutes:
        return DEFAULT_RETURN

    # If we're before the first shape entry, can't forward-fill
    if minute_window < sorted_minutes[0]:
        return DEFAULT_RETURN

    # Find the closest minute <= current minute_window
    for m in reversed(sorted_minutes):
        if m <= minute_window:
            entry = player_windows[m]
            return (
                entry["expected_x"],
                entry["expected_y"],
                entry["fit_score"],
                entry["role"],
            )

    return DEFAULT_RETURN


# ═══════════════════════════════════════════════════════════════════════════
# Vectorised Bridge Helper
# ═══════════════════════════════════════════════════════════════════════════


def vectorise_expected_positions(
    df: pd.DataFrame,
    player_role_map: dict[int, dict[int, dict[str, Any]]],
    frames_per_minute: int = 1500,
    allow_forward_fill: bool = True,
) -> pd.DataFrame:
    """Apply :func:`get_expected_position` to every row in a DataFrame.

    Adds columns: ``expected_x``, ``expected_y``, ``fit_score``,
    ``shape_role``.

    This is a vectorised wrapper around the scalar lookup. For large
    DataFrames, pre-group by player_id for efficiency.

    Parameters
    ----------
    df : pd.DataFrame
        Tracking DataFrame with ``player_id`` and ``frame`` columns.
    player_role_map : dict
        Mapping from :func:`build_player_role_map`.
    frames_per_minute : int
        Frames per minute (default 1500).
    allow_forward_fill : bool
        Whether to forward-fill expected positions when no exact shape
        entry exists for the current minute.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns added in-place.
    """
    if "player_id" not in df.columns or "frame" not in df.columns:
        raise ValueError("DataFrame must contain 'player_id' and 'frame' columns")

    out_x = np.full(len(df), np.nan, dtype=np.float64)
    out_y = np.full(len(df), np.nan, dtype=np.float64)
    out_fit = np.full(len(df), np.nan, dtype=np.float64)
    out_role = np.full(len(df), "", dtype=object)

    # Process group-wise for caching — all frames for one player at once
    for pid, group_indices in df.groupby("player_id", sort=False).indices.items():
        idxs = group_indices
        frames = df["frame"].iloc[idxs].values

        if pid not in player_role_map:
            continue

        p_windows = player_role_map[pid]
        if not p_windows:
            continue

        sorted_minutes = sorted(p_windows.keys())
        min_ = sorted_minutes[0]

        # Build minute → entry lookup
        # Compute minute_window for each frame
        minute_windows = frames // frames_per_minute

        for local_i, mw in enumerate(minute_windows):
            global_i = idxs[local_i]

            if mw in p_windows:
                entry = p_windows[mw]
                out_x[global_i] = entry["expected_x"]
                out_y[global_i] = entry["expected_y"]
                out_fit[global_i] = entry["fit_score"]
                out_role[global_i] = entry["role"]
            elif allow_forward_fill and mw >= min_:
                # Forward-fill
                for m in reversed(sorted_minutes):
                    if m <= mw:
                        entry = p_windows[m]
                        out_x[global_i] = entry["expected_x"]
                        out_y[global_i] = entry["expected_y"]
                        out_fit[global_i] = entry["fit_score"]
                        out_role[global_i] = entry["role"]
                        break

    df = df.copy()
    df["expected_x"] = out_x
    df["expected_y"] = out_y
    df["fit_score"] = out_fit
    df["shape_role"] = out_role
    return df
