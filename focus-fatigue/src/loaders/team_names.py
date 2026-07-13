"""Team name lookup bridge.

Stats Perform data uses three different ID systems:
1. team_id_opta (tracking.parquet) — integer IDs, e.g. 142, 145
2. UUIDs (shape.json contestant IDs) — string IDs
3. Team names — in matchInfo of shape.json

This module provides a cache that maps team_id_opta → team name
for all matches, built from the shape files.
"""

from pathlib import Path
from typing import Optional, Union
import json

import pandas as pd

# Default team mapping path (relative to project root)
_DEFAULT_TEAM_MAP_PATH = Path("./data/raw/team_mappings/team_mappings.csv")


def _load_team_uuid_to_opta(path: Path | None = None) -> dict[str, int]:
    """Load the UUID → Opta ID mapping from CSV.

    Parameters
    ----------
    path : Path or None
        Path to the team_mappings.csv file. Falls back to
        ``_DEFAULT_TEAM_MAP_PATH`` if None.
    """
    import csv

    resolve_path = path or _DEFAULT_TEAM_MAP_PATH

    mapping = {}
    if resolve_path.exists():
        with open(resolve_path) as f:
            for row in csv.DictReader(f):
                mapping[row["uuid"]] = int(row["opta_id"])
    return mapping


def _load_opta_to_uuid(path: Path | None = None) -> dict[int, str]:
    """Reverse mapping: Opta ID → UUID.

    Parameters
    ----------
    path : Path or None
        Path to the team_mappings.csv file. Falls back to
        ``_DEFAULT_TEAM_MAP_PATH`` if None.
    """
    return {v: k for k, v in _load_team_uuid_to_opta(path=path).items()}


def build_team_name_cache(
    shape_dir: Union[str, Path],
    team_mappings_path: Optional[Union[str, Path]] = None,
) -> dict[int, str]:
    """Build a cache of team_id_opta → team name from all shape files.

    Parameters
    ----------
    shape_dir : str or Path
        Directory containing all shape.json files.
    team_mappings_path : str or Path, optional
        Path to team_mappings.csv. Falls back to ``_DEFAULT_TEAM_MAP_PATH``.

    Returns
    -------
    dict[int, str]
        {team_id_opta: "Team Name"}
    """
    shape_dir = Path(shape_dir)
    uuid_to_opta = _load_team_uuid_to_opta(
        path=Path(team_mappings_path) if team_mappings_path else None
    )
    result = {}

    for sf in sorted(shape_dir.glob("*.json")):
        try:
            with open(sf) as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue

        mi = data.get("matchInfo", {})
        for contestant in mi.get("contestant", []):
            cid = contestant["id"]
            opta_id = uuid_to_opta.get(cid)
            if opta_id is not None:
                result[opta_id] = contestant["name"]

    return result


def get_team_name(
    team_id_opta: int,
    cache: Optional[dict[int, str]] = None,
) -> str:
    """Get team name for an Opta team ID.

    Parameters
    ----------
    team_id_opta : int
        The team ID from tracking.parquet.
    cache : dict, optional
        Pre-built cache from build_team_name_cache(). If None, builds it.

    Returns
    -------
    str
        Team name, or f"Team-{team_id_opta}" if not found.
    """
    if cache is None:
        shape_dir = Path("./data/raw/shapes")
        cache = build_team_name_cache(shape_dir)

    return cache.get(team_id_opta, f"Team-{team_id_opta}")


def identify_opposing_team(team_id_opta: int, match_shape_file: Union[str, Path]) -> int:
    """Given one team's Opta ID, find the other team's Opta ID in a match.

    Useful for identifying which team is the opponent.
    """
    import json

    uuid_to_opta = _load_team_uuid_to_opta()

    with open(match_shape_file) as f:
        data = json.load(f)

    for contestant in data["matchInfo"]["contestant"]:
        cid = contestant["id"]
        opta_id = uuid_to_opta.get(cid)
        if opta_id is not None and opta_id != team_id_opta:
            return opta_id

    return -1
