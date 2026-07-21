"""Signal: Team Centroid Distance.

Measures each player's distance from their team's centre of mass
while out of possession. Larger distances indicate players stretching
or being pulled out of the team shape (e.g., tracking a dropping
attacker); smaller distances indicate compactness.

Per-frame computation (out-of-possession frames only):
  1. Compute team centroid (mean x, mean y) per team
  2. For each outfield player, compute Euclidean distance to their
     team centroid

Aggregation: mean distance per 5-minute block per player.

Output schema: standard per-player signal output with columns:
  game_id, block_id, phase, player_id, team_id_opta,
  signal_name, signal_value, n_frames
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.signals.registry import register_signal

logger = logging.getLogger(__name__)


@register_signal
class TeamCentroidDistanceSignal:
    """Per-player distance from team centre of mass."""

    signal_name = "team_centroid_distance"

    # Player types to exclude (goalkeepers)
    GK_PREFIXES = ("goalkeeper", "gk", "keeper", "goalkeeper_")

    def compute(
        self,
        blocks: list[pd.DataFrame],
        match_id: int,
        events: Optional[pd.DataFrame] = None,
        phase: str = "out_of_possession",
        team_col: str = "team",
        player_id_col: str = "player_id",
        x_col: str = "x",
        y_col: str = "y",
    ) -> pd.DataFrame:
        """Compute per-player centroid distance per block.

        Parameters
        ----------
        blocks : list[pd.DataFrame]
            Model 1 blocks with columns: frame, x, y, player_id, team.
        match_id : int
            Game/match identifier.
        events : pd.DataFrame, optional
            Not used.
        phase : str
            Label for the `phase` output column.
        team_col : str
            Name of the team identifier column.
        player_id_col : str
            Name of the player ID column.
        x_col : str
            X coordinate column (metres).
        y_col : str
            Y coordinate column (metres).

        Returns
        -------
        pd.DataFrame
            Stacked output with columns:
            game_id, block_id, phase, player_id, team_id_opta,
            signal_name, signal_value, n_frames
        """
        if not blocks:
            return self._empty_result(match_id)

        all_records = []
        for block_df in blocks:
            block_id = block_df["block_id"].iloc[0]
            block_records = self._process_block(
                block_df, match_id, block_id, phase,
                team_col, player_id_col, x_col, y_col,
            )
            all_records.extend(block_records)

        result = pd.DataFrame(all_records)
        logger.info(
            "Match %d: %d blocks → %d centroid-distance records",
            match_id, len(blocks), len(result),
        )
        return result

    def _process_block(
        self,
        block_df: pd.DataFrame,
        match_id: int,
        block_id: int,
        phase: str,
        team_col: str,
        player_id_col: str,
        x_col: str,
        y_col: str,
    ) -> list[dict]:
        """Compute centroid distance for a single block.

        Yields one record per player per team.
        """
        if team_col not in block_df.columns:
            logger.warning("Block %d: no '%s' column, skipping", block_id, team_col)
            return []

        # Filter out goalkeepers
        df = block_df.copy()
        if player_id_col in df.columns:
            gk_mask = df[player_id_col].astype(str).str.lower().str.startswith(
                self.__class__.GK_PREFIXES, na=False
            )
            df = df[~gk_mask]

        teams = df[team_col].unique()
        records = []

        for team in teams:
            team_df = df[df[team_col] == team]
            team_records = self._process_team(
                team_df, match_id, block_id, phase,
                team, player_id_col, x_col, y_col,
            )
            records.extend(team_records)

        return records

    def _process_team(
        self,
        team_df: pd.DataFrame,
        match_id: int,
        block_id: int,
        phase: str,
        team_id: str,
        player_id_col: str,
        x_col: str,
        y_col: str,
    ) -> list[dict]:
        """Compute per-player centroid distance for one team."""
        players = team_df[player_id_col].unique()
        records = []

        for player in players:
            player_df = team_df[team_df[player_id_col] == player]
            distances = []

            for frame_id, frame_group in team_df.groupby("frame"):
                # Compute team centroid from ALL outfield players in this frame
                frame_players = frame_group
                cx = frame_players[x_col].mean()
                cy = frame_players[y_col].mean()

                # Get this player's position in this frame
                player_frame = frame_group[frame_group[player_id_col] == player]
                if player_frame.empty:
                    continue

                px = player_frame.iloc[0][x_col]
                py = player_frame.iloc[0][y_col]
                dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                distances.append(dist)

            n_frames = len(distances)
            mean_dist = float(np.mean(distances)) if distances else float("nan")

            records.append({
                "game_id": match_id,
                "block_id": block_id,
                "phase": phase,
                "player_id": str(player),
                "team_id_opta": str(team_id),
                "signal_name": self.signal_name,
                "signal_value": mean_dist,
                "n_frames": n_frames,
            })

        return records

    def _empty_result(self, match_id: int) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "game_id", "block_id", "phase", "player_id",
            "team_id_opta", "signal_name", "signal_value", "n_frames",
        ])
