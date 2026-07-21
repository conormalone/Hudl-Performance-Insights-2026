"""Signal: Team Polarisation (movement vector alignment).

Measures how aligned a team's outfield players are in their movement
direction while out of possession. Low polarisation indicates players
moving in diverse directions (low coordination); high polarisation
indicates synchronised movement.

Per-frame computation (out-of-possession frames only):
  1. Get unit velocity vectors for each outfield player
  2. Compute mean resultant vector length:
       R = ||sum(unit_vectors)|| / n
  3. R ∈ [0, 1] where:
       1 = all players moving in identical direction
       0 = perfect directional dispersion

Aggregation: mean R per 5-minute block per team.

Output schema:
  game_id, block_id, phase, player_id (0 = team placeholder),
  team_id_opta, signal_name, signal_value, n_frames
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.signals.registry import register_signal

logger = logging.getLogger(__name__)


@register_signal
class PolarisationSignal:
    """Team movement polarisation — how synchronised a team's movement is."""

    signal_name = "team_polarisation"

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
        vx_col: str = "vx",
        vy_col: str = "vy",
    ) -> pd.DataFrame:
        """Compute team polarisation per block.

        Parameters
        ----------
        blocks : list[pd.DataFrame]
            Model 1 blocks with columns: frame, x, y, player_id, team,
            and optionally vx, vy.
        match_id : int
            Game/match identifier.
        events : pd.DataFrame, optional
            Not used — possession is estimated or passed as-is.
        phase : str
            Label for the `phase` output column.
        team_col : str
            Name of the team identifier column.
        player_id_col : str
            Name of the player ID column.
        vx_col : str
            Velocity X component column.
        vy_col : str
            Velocity Y component column.

        Returns
        -------
        pd.DataFrame
            Stacked output with columns:
            game_id, block_id, phase, player_id, team_id_opta,
            signal_name, signal_value, n_frames
        """
        if not blocks:
            return self._empty_result(match_id)

        records = []
        for block_df in blocks:
            block_id = block_df["block_id"].iloc[0]
            block_result = self._process_block(
                block_df, match_id, block_id, phase,
                team_col, player_id_col, vx_col, vy_col,
            )
            records.extend(block_result)

        result = pd.DataFrame(records)
        logger.info(
            "Match %d: %d blocks → %d team-polarisation records",
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
        vx_col: str,
        vy_col: str,
    ) -> list[dict]:
        """Compute polarisation for a single block.

        Yields one record per team.
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
            n_frames = team_df["frame"].nunique()

            if n_frames < 2:
                records.append(self._make_record(
                    match_id, block_id, phase, 0, str(team),
                    float("nan"), n_frames,
                ))
                continue

            # Compute per-frame polarisation
            r_values = []
            for _, frame_group in team_df.groupby("frame"):
                r = self._frame_polarisation(
                    frame_group, vx_col, vy_col,
                )
                if r is not None:
                    r_values.append(r)

            mean_r = float(np.mean(r_values)) if r_values else float("nan")
            records.append(self._make_record(
                match_id, block_id, phase, 0, str(team),
                mean_r, n_frames,
            ))

        return records

    def _frame_polarisation(
        self,
        frame_df: pd.DataFrame,
        vx_col: str,
        vy_col: str,
    ) -> Optional[float]:
        """Compute mean resultant vector length R for one frame.

        R = ||sum(unit_vectors)|| / n

        Returns None if velocity data is unavailable.
        """
        if vx_col not in frame_df.columns or vy_col not in frame_df.columns:
            return None

        vx = frame_df[vx_col].values
        vy = frame_df[vy_col].values

        # Compute speed
        speeds = np.sqrt(vx**2 + vy**2)
        n = len(vx)

        if n == 0 or np.all(speeds < 1e-6):
            return None

        # Unit vectors
        ux = np.divide(vx, speeds, out=np.zeros_like(vx), where=speeds > 1e-6)
        uy = np.divide(vy, speeds, out=np.zeros_like(vy), where=speeds > 1e-6)

        # Resultant vector
        rx = np.sum(ux)
        ry = np.sum(uy)
        r = np.sqrt(rx**2 + ry**2) / n

        return float(r)

    def _make_record(
        self,
        match_id: int,
        block_id: int,
        phase: str,
        player_id: int,
        team_id_opta: str,
        signal_value: float,
        n_frames: int,
    ) -> dict:
        return {
            "game_id": match_id,
            "block_id": block_id,
            "phase": phase,
            "player_id": player_id,
            "team_id_opta": team_id_opta,
            "signal_name": self.signal_name,
            "signal_value": signal_value,
            "n_frames": n_frames,
        }

    def _empty_result(self, match_id: int) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "game_id", "block_id", "phase", "player_id",
            "team_id_opta", "signal_name", "signal_value", "n_frames",
        ])
