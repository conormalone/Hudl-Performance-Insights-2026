"""Parser for Stats Perform shape.json files.

Shape files contain pre-computed formation templates every 60 seconds,
for both in-possession and out-of-possession phases, for both teams.

Key value for the project:
    - Provides expected position baselines for each player's role
    - Eliminates need for EFPI clustering (formation identification)
    - Links player jersey numbers to role descriptions per time window
    - Provides team name + competition metadata

Shape entry structure:
    {
        "increment": "60 Seconds",
        "periodId": "1",            # 1 or 2
        "contestant": [
            {
                "id": "uuid",
                "inPossession": {
                    "shape": [
                        {
                            "labelId": "1_0",
                            "label": "3-4-3",
                            "formation": "3-4-3",
                            "fitScore": 5.84,
                            "periodStart": "00:00",
                            "periodEnd": "05:00",
                            "shapeRole": {
                                "role": [
                                    {
                                        "id": "3-4-3 0",
                                        "roleDescription": "CB.R",
                                        "playerId": "uuid",
                                        "shirtNumber": "2",
                                        "fitScore": 0.28,
                                        "averageRolePositionX": "37.62",
                                        "averageRolePositionY": "31.06"
                                    },
                                    ...
                                ]
                            }
                        }
                    ]
                },
                "outOfPossession": { ... same structure ... }
            }
        ]
    }

Note: playerId values in shape.json are Stats Perform string UUIDs,
NOT the same as integer player_ids in tracking.parquet.
Bridge via team_id_opta + jersey_no + time window.
"""

from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd


def parse_shape_file(filepath: Union[str, Path]) -> dict:
    """Parse a raw shape.json file into a Python dict.

    Parameters
    ----------
    filepath : str or Path
        Path to the shape.json file.

    Returns
    -------
    dict
        Parsed JSON with top-level keys: matchInfo, liveData
    """
    import json

    filepath = Path(filepath)
    with open(filepath) as f:
        return json.load(f)


def get_match_info(filepath: Union[str, Path]) -> dict:
    """Extract match metadata from a shape file.

    Returns
    -------
    dict with keys: id, description, date, competition, home_team, away_team,
                     home_id, away_id, home_opta_id, away_opta_id
    """
    data = parse_shape_file(filepath)
    mi = data["matchInfo"]

    home = [c for c in mi["contestant"] if c["position"] == "home"][0]
    away = [c for c in mi["contestant"] if c["position"] == "away"][0]

    return {
        "match_id": mi["id"],
        "description": mi.get("description", ""),
        "date": mi.get("date", ""),
        "competition": mi["competition"]["name"],
        "home_name": home["name"],
        "away_name": away["name"],
        "home_id": home["id"],
        "away_id": away["id"],
    }


def load_shape_roles(
    filepath: Union[str, Path],
    team_opta_id: int,
    match_id: Optional[str] = None,
) -> pd.DataFrame:
    """Parse shape entries into a flat DataFrame of player roles per time window.

    Parameters
    ----------
    filepath : str or Path
        Path to shape.json.
    team_opta_id : int
        The Opta team ID for the team we want roles for (from tracking.parquet
        team_id_opta column). The shape entries use UUID contestant IDs, which
        need bridging.
    match_id : str, optional
        Match identifier.

    Returns
    -------
    pd.DataFrame
        Columns: game_id, phase, minute_start, minute_end, possession_state,
                 formation, role_description, player_uuid, jersey_no,
                 avg_x, avg_y, fit_score
        One row per role per shape entry (one entry per 5-minute window).
    """
    data = parse_shape_file(filepath)
    shapes = data["liveData"]["shapes"]
    mi = data["matchInfo"]

    if match_id is None:
        match_id = Path(filepath).stem

    # Map UUID contestant IDs to Opta IDs
    # We need the team_mappings.csv for this. But we can also infer from shape
    # positions (home = first contestant when entries are in order).
    # Actually, let's just accept team_opta_id and look it up from a provided mapping.

    # Load team mappings
    team_map_path = Path(filepath).parent.parent / "team_mappings.csv"
    team_uuid_to_opta = {}
    if team_map_path.exists():
        import csv
        with open(team_map_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                team_uuid_to_opta[row["uuid"]] = int(row["opta_id"])

    records = []

    for entry in shapes:
        increment_str = entry.get("increment", "60 Seconds")
        increment_min = int(increment_str.split()[0])
        period_id = int(entry["periodId"])

        for contestant in entry["contestant"]:
            contestant_uuid = contestant["id"]
            c_opta_id = team_uuid_to_opta.get(contestant_uuid)

            # Skip if this isn't the requested team
            if c_opta_id is None or c_opta_id != team_opta_id:
                continue

            for possession_state in ["inPossession", "outOfPossession"]:
                state_data = contestant.get(possession_state, {})
                shapes_list = state_data.get("shape", [])
                for shape_entry in shapes_list:
                    # Parse time window
                    period_start = shape_entry.get("periodStart", "00:00")
                    period_end = shape_entry.get("periodEnd", "05:00")

                    parts_s = period_start.split(":")
                    parts_e = period_end.split(":")
                    minute_start = int(parts_s[0]) + (period_id - 1) * 45
                    minute_end = int(parts_e[0]) + (period_id - 1) * 45

                    formation = shape_entry.get("formation", "?")
                    label = shape_entry.get("label", "?")

                    roles = (
                        shape_entry.get("shapeRole", {})
                        .get("role", [])
                    )

                    for role in roles:
                        records.append(
                            {
                                "game_id": match_id,
                                "phase": period_id,
                                "minute_start": minute_start,
                                "minute_end": minute_end,
                                "possession_state": possession_state,
                                "formation": formation,
                                "label": label,
                                "role_description": role.get(
                                    "roleDescription", ""
                                ),
                                "player_uuid": role.get("playerId", ""),
                                "jersey_no": int(
                                    role.get("shirtNumber", 0)
                                ),
                                "avg_x": float(
                                    role.get("averageRolePositionX", 0)
                                ),
                                "avg_y": float(
                                    role.get("averageRolePositionY", 0)
                                ),
                                "fit_score": float(
                                    role.get("fitScore", 0)
                                ),
                            }
                        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # DOP-normalise role positions: shape.json positions are always
    # in attacking-leftward coordinates (home team attacks left in first half).
    # We normalise so all attacks go rightward (consistent with tracking).
    # For the home team (first contestant), DOP is L in first half.
    # For the away team (second contestant), DOP is R in first half.
    # After DOP normalisation, flip x for home team entries.
    home_uuid = mi["contestant"][0]["id"]
    home_opta_id = team_uuid_to_opta.get(home_uuid)

    if team_opta_id == home_opta_id:
        # Home team: flip x in first half (DOP=L becomes DOP=R)
        df.loc[df["phase"] == 1, "avg_x"] = -df.loc[
            df["phase"] == 1, "avg_x"
        ]

    return df.reset_index(drop=True)


def get_team_formation_summary(filepath: Union[str, Path]) -> list[dict]:
    """Get a simple per-team summary of formations used in a match.

    Returns a list of dicts with team_name, possession_state,
    formation, and count of windows where it appears.
    """
    import json
    from collections import Counter

    data = parse_shape_file(filepath)
    mi = data["matchInfo"]
    shapes = data["liveData"]["shapes"]

    teams = {c["id"]: c["name"] for c in mi["contestant"]}
    results = []

    for entry in shapes:
        for contestant in entry["contestant"]:
            cid = contestant["id"]
            tname = teams.get(cid, cid)
            for state in ["inPossession", "outOfPossession"]:
                for s in contestant[state].get("shape", []):
                    results.append(
                        {
                            "team": tname,
                            "state": state,
                            "formation": s.get("formation", "?"),
                        }
                    )

    summary = []
    for (team, state, formation), count in Counter(
        (r["team"], r["state"], r["formation"]) for r in results
    ).most_common():
        summary.append(
            {
                "team": team,
                "possession_state": state,
                "formation": formation,
                "windows": count,
            }
        )

    return summary
