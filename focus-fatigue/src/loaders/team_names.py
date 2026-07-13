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

# Load team UUID → Opta ID mapping
_TEAM_MAP_PATH = Path(
    "/home/conormalone/conor_downloads/team_mappings/team_mappings.csv"
)


def _load_team_uuid_to_opta() -> dict[str, int]:
    """Load the UUID → Opta ID mapping from CSV."""
    import csv

    mapping = {}
    if _TEAM_MAP_PATH.exists():
        with open(_TEAM_MAP_PATH) as f:
            for row in csv.DictReader(f):
                mapping[row["uuid"]] = int(row["opta_id"])
    return mapping


def _load_opta_to_uuid() -> dict[int, str]:
    """Reverse mapping: Opta ID → UUID."""
    return {v: k for k, v in _load_team_uuid_to_opta().items()}


def build_team_name_cache(
    shape_dir: Union[str, Path],
) -> dict[int, str]:
    """Build a cache of team_id_opta → team name from all shape files.

    Parameters
    ----------
    shape_dir : str or Path
        Directory containing all shape.json files.

    Returns
    -------
    dict[int, str]
        {team_id_opta: "Team Name"}
    """
    shape_dir = Path(shape_dir)
    uuid_to_opta = _load_team_uuid_to_opta()
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
        shape_dir = Path(
            "/home/conormalone/conor_downloads/team_mappings/shape_outputs"
        )
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
