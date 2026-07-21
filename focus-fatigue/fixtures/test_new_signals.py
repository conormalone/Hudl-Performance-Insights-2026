#!/usr/bin/env python3
"""Synthetic data tests for the two new signals.

Tests use purely synthetic tracking data generated in-memory — no real
data files required. This ensures tests pass on the Raspberry Pi with
no external dependencies beyond the project's standard package imports.

Each test verifies:
- Correct output schema (OUTPUT_COLUMNS)
- Signal-specific value ranges and invariants
- Edge cases: empty blocks, missing columns, extreme values

Usage:
    python -m pytest fixtures/test_new_signals.py -v
    python fixtures/test_new_signals.py   (runs unittest directly)
"""

import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math
import unittest
import warnings

import numpy as np
import pandas as pd

# Import signal modules to trigger @register_signal decorators
import src.signals.polarisation           # noqa: F401
import src.signals.team_centroid_distance  # noqa: F401

from src.signals.registry import SIGNAL_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Data Helpers
# ═══════════════════════════════════════════════════════════════════════════

RNG = np.random.default_rng(42)
BALL_PLAYER_ID = -1
PITCH_X_MAX = 116.0
PITCH_Y_MAX = 78.0


def make_synthetic_match(
    n_frames: int = 100,
    n_players_per_team: int = 10,
    seed: int = 42,
    team_a_id: int = 1,
    team_b_id: int = 2,
) -> pd.DataFrame:
    """Generate synthetic tracking data for testing.

    Returns
    -------
    pd.DataFrame
        Columns: frame_count, player_id, team_id_opta, x, y, vx, vy,
        v_mag, heading, team_in_possession, phase.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    # Two teams alternating possession every 25 frames
    possession_team = team_a_id

    for frame in range(n_frames):
        # Alternate possession every 25 frames
        if frame > 0 and frame % 25 == 0:
            possession_team = team_b_id if possession_team == team_a_id else team_a_id

        # Ball player (id = -1) — record ball position and possession
        rows.append({
            "frame_count": frame,
            "player_id": BALL_PLAYER_ID,
            "team_id_opta": 0,
            "x": PITCH_X_MAX / 2 + rng.uniform(-10, 10),
            "y": PITCH_Y_MAX / 2 + rng.uniform(-10, 10),
            "vx": rng.uniform(-5, 5),
            "vy": rng.uniform(-5, 5),
            "v_mag": rng.uniform(1.0, 10.0),
            "heading": rng.uniform(-np.pi, np.pi),
            "team_in_possession": possession_team,
            "phase": 1 if frame < n_frames // 2 else 2,
        })

        # Team A outfield players
        for p in range(n_players_per_team):
            pid = 100 + p  # player IDs 100..109
            rows.append({
                "frame_count": frame,
                "player_id": pid,
                "team_id_opta": team_a_id,
                "x": rng.uniform(0, PITCH_X_MAX),
                "y": rng.uniform(0, PITCH_Y_MAX),
                "vx": rng.uniform(-10, 10),
                "vy": rng.uniform(-10, 10),
                "v_mag": rng.uniform(0.5, 8.0),
                "heading": rng.uniform(-np.pi, np.pi),
                "team_in_possession": possession_team,
                "phase": 1 if frame < n_frames // 2 else 2,
            })

        # Team B outfield players
        for p in range(n_players_per_team):
            pid = 200 + p  # player IDs 200..209
            rows.append({
                "frame_count": frame,
                "player_id": pid,
                "team_id_opta": team_b_id,
                "x": rng.uniform(0, PITCH_X_MAX),
                "y": rng.uniform(0, PITCH_Y_MAX),
                "vx": rng.uniform(-10, 10),
                "vy": rng.uniform(-10, 10),
                "v_mag": rng.uniform(0.5, 8.0),
                "heading": rng.uniform(-np.pi, np.pi),
                "team_in_possession": possession_team,
                "phase": 1 if frame < n_frames // 2 else 2,
            })

    return pd.DataFrame(rows)


def make_aligned_match(n_frames: int = 50) -> pd.DataFrame:
    """Generate synthetic data where both teams move in near-perfect unison.

    All players on each team have nearly identical velocity vectors,
    producing R ≈ 1 for polarisation.
    """
    rng = np.random.default_rng(43)
    rows: list[dict] = []

    team_a_dir = np.array([8.0, 2.0])   # team A moves roughly right-up
    team_b_dir = np.array([-7.0, -3.0])  # team B moves roughly left-down

    for frame in range(n_frames):
        possession = 1 if frame < 25 else 2

        # Ball
        rows.append({
            "frame_count": frame, "player_id": BALL_PLAYER_ID,
            "team_id_opta": 0,
            "x": 58.0, "y": 39.0,
            "vx": 0.0, "vy": 0.0, "v_mag": 0.0, "heading": 0.0,
            "team_in_possession": possession, "phase": 1,
        })

        # Team A — mostly aligned (small normal noise)
        for p in range(10):
            jitter_x = rng.normal(0, 0.3)   # small jitter
            jitter_y = rng.normal(0, 0.3)
            vx = team_a_dir[0] + jitter_x
            vy = team_a_dir[1] + jitter_y
            heading = math.atan2(vy, vx)
            rows.append({
                "frame_count": frame, "player_id": 100 + p,
                "team_id_opta": 1,
                "x": rng.uniform(0, 116), "y": rng.uniform(0, 78),
                "vx": vx, "vy": vy,
                "v_mag": math.sqrt(vx**2 + vy**2), "heading": heading,
                "team_in_possession": possession, "phase": 1,
            })

        # Team B — mostly aligned
        for p in range(10):
            jitter_x = rng.normal(0, 0.3)
            jitter_y = rng.normal(0, 0.3)
            vx = team_b_dir[0] + jitter_x
            vy = team_b_dir[1] + jitter_y
            heading = math.atan2(vy, vx)
            rows.append({
                "frame_count": frame, "player_id": 200 + p,
                "team_id_opta": 2,
                "x": rng.uniform(0, 116), "y": rng.uniform(0, 78),
                "vx": vx, "vy": vy,
                "v_mag": math.sqrt(vx**2 + vy**2), "heading": heading,
                "team_in_possession": possession, "phase": 1,
            })

    return pd.DataFrame(rows)


def make_blocks(match_df: pd.DataFrame, window_frames: int = 50) -> list[pd.DataFrame]:
    """Split synthetic match into blocks (mimics split_into_blocks behaviour).

    Returns list of DataFrames with 'block_id' column attached.
    """
    blocks: list[pd.DataFrame] = []
    frames = sorted(match_df["frame_count"].unique())
    phase_frames: dict[int, list[int]] = {}
    for f in frames:
        ph = match_df[match_df["frame_count"] == f]["phase"].iloc[0]
        phase_frames.setdefault(ph, []).append(f)

    for phase, pf in phase_frames.items():
        pf_sorted = sorted(pf)
        blk_num = 0
        start = pf_sorted[0]
        while start <= pf_sorted[-1]:
            end = start + window_frames
            block_df = match_df[
                (match_df["frame_count"] >= start)
                & (match_df["frame_count"] < end)
            ].copy()
            if len(block_df) > 0:
                block_df["block_id"] = f"{phase}_{blk_num}"
                blocks.append(block_df)
            start = end
            blk_num += 1

    return blocks


# ═══════════════════════════════════════════════════════════════════════════
# Test Suite
# ═══════════════════════════════════════════════════════════════════════════

class TestNewSignals(unittest.TestCase):
    """Test suite for team_polarisation and team_centroid_distance signals."""

    @classmethod
    def setUpClass(cls):
        # Build synthetic match data once
        cls.match_df = make_synthetic_match(n_frames=100, n_players_per_team=5)
        cls.blocks = make_blocks(cls.match_df, window_frames=50)

        cls.aligned_df = make_aligned_match(n_frames=50)
        cls.aligned_blocks = make_blocks(cls.aligned_df, window_frames=50)

        # Get signal instances
        pol_cls = SIGNAL_REGISTRY.get("team_polarisation")
        cent_cls = SIGNAL_REGISTRY.get("team_centroid_distance")
        if pol_cls is None or cent_cls is None:
            raise RuntimeError(
                "Signals not registered! Did you import the modules?"
            )
        cls.polarisation_signal = pol_cls()
        cls.centroid_signal = cent_cls()

    # ── Schema Tests ──────────────────────────────────────────────────

    def test_polarisation_output_schema(self):
        """Polarisation output has all required columns."""
        result = self.polarisation_signal.compute(
            self.match_df, self.blocks, game_id="test_match"
        )
        expected_cols = [
            "game_id", "block_id", "phase", "player_id",
            "team_id_opta", "signal_name", "signal_value", "n_frames",
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")
        self.assertEqual(
            result["signal_name"].unique().tolist(),
            ["team_polarisation"],
        )

    def test_centroid_distance_output_schema(self):
        """Centroid distance output has all required columns."""
        result = self.centroid_signal.compute(
            self.match_df, self.blocks, game_id="test_match"
        )
        expected_cols = [
            "game_id", "block_id", "phase", "player_id",
            "team_id_opta", "signal_name", "signal_value", "n_frames",
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")
        self.assertEqual(
            result["signal_name"].unique().tolist(),
            ["team_centroid_distance"],
        )

    # ── Polarisation Value Range Tests ────────────────────────────────

    def test_polarisation_values_in_0_1(self):
        """Polarisation R values are always in [0, 1]."""
        result = self.polarisation_signal.compute(
            self.match_df, self.blocks, game_id="test_match"
        )
        if len(result) > 0:
            sv = result["signal_value"]
            self.assertGreaterEqual(sv.min(), 0.0)
            self.assertLessEqual(sv.max(), 1.0)

    def test_polarisation_aligned_team_near_1(self):
        """Aligned team (all players moving same direction) has R close to 1."""
        result = self.polarisation_signal.compute(
            self.aligned_df, self.aligned_blocks, game_id="aligned",
        )
        if len(result) > 0:
            sv = result["signal_value"]
            # With low jitter (std=0.3), R should be very close to 1
            mean_r = sv.mean()
            self.assertGreater(
                mean_r, 0.95,
                f"Expected R ≈ 1 for aligned motion, got mean R={mean_r:.4f}",
            )

    def test_polarisation_random_team_near_0(self):
        """Random-direction team (uniform velocities) has R near 0."""
        # Our synthetic match_df has random velocities → R should be low
        result = self.polarisation_signal.compute(
            self.match_df, self.blocks, game_id="random",
        )
        if len(result) > 0:
            sv = result["signal_value"]
            mean_r = sv.mean()
            self.assertLess(
                mean_r, 0.5,
                f"Expected R ≈ 0 for random motion, got mean R={mean_r:.4f}",
            )

    def test_polarisation_no_blocks_returns_empty(self):
        """Empty blocks yield empty DataFrame with correct schema."""
        result = self.polarisation_signal.compute(
            self.match_df, [], game_id="test_empty",
        )
        self.assertEqual(len(result), 0)

    # ── Centroid Distance Value Range Tests ───────────────────────────

    def test_centroid_distance_non_negative(self):
        """All centroid distances are non-negative."""
        result = self.centroid_signal.compute(
            self.match_df, self.blocks, game_id="test_match"
        )
        if len(result) > 0:
            sv = result["signal_value"]
            self.assertGreaterEqual(sv.min(), 0.0)

    def test_centroid_distance_no_ball_player(self):
        """Ball player (id=-1) never appears in centroid distance output."""
        result = self.centroid_signal.compute(
            self.match_df, self.blocks, game_id="test_match"
        )
        if len(result) > 0:
            self.assertNotIn(BALL_PLAYER_ID, result["player_id"].values)

    def test_centroid_distance_plausible_magnitude(self):
        """Centroid distances are within pitch dimensions."""
        result = self.centroid_signal.compute(
            self.match_df, self.blocks, game_id="test_match"
        )
        if len(result) > 0:
            sv = result["signal_value"]
            # Points can't be > pitch diagonal (~140m)
            self.assertLessEqual(sv.max(), 150.0)

    def test_centroid_distance_empty_blocks(self):
        """Empty blocks yield empty DataFrame."""
        result = self.centroid_signal.compute(
            self.match_df, [], game_id="test_empty",
        )
        self.assertEqual(len(result), 0)

    # ── Edge Case: Missing Columns ────────────────────────────────────

    def test_polarisation_missing_velocity_columns(self):
        """Polarisation handles missing vx/vy columns gracefully (zero velocity fallback)."""
        df_no_vx = self.match_df.drop(columns=["vx", "vy", "v_mag", "heading"],
                                       errors="ignore")
        # Should not crash — will log a warning and use zero velocity
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.polarisation_signal.compute(
                df_no_vx, self.blocks, game_id="test_novx",
            )
        self.assertIsInstance(result, pd.DataFrame)

    def test_centroid_distance_missing_x_y(self):
        """Centroid distance raises error if x/y columns are missing."""
        df_no_xy = self.match_df.drop(columns=["x", "y"], errors="ignore")
        with self.assertRaises(KeyError):
            self.centroid_signal.compute(
                df_no_xy, self.blocks, game_id="test_noxy"
            )

    # ── Validate Method Tests ─────────────────────────────────────────

    def test_polarisation_validate_passes(self):
        """Polarisation.validate() returns True for valid output."""
        result = self.polarisation_signal.compute(
            self.match_df, self.blocks, game_id="test_val"
        )
        self.assertTrue(self.polarisation_signal.validate(result))

    def test_centroid_distance_validate_passes(self):
        """CentroidDistance.validate() returns True for valid output."""
        result = self.centroid_signal.compute(
            self.match_df, self.blocks, game_id="test_val"
        )
        self.assertTrue(self.centroid_signal.validate(result))

    def test_polarisation_validate_empty(self):
        """Polarisation.validate() handles empty DataFrames."""
        empty = pd.DataFrame(columns=[
            "game_id", "block_id", "phase", "player_id", "team_id_opta",
            "signal_name", "signal_value", "n_frames",
        ])
        self.assertTrue(self.polarisation_signal.validate(empty))

    def test_centroid_distance_validate_empty(self):
        """CentroidDistance.validate() handles empty DataFrames."""
        empty = pd.DataFrame(columns=[
            "game_id", "block_id", "phase", "player_id", "team_id_opta",
            "signal_name", "signal_value", "n_frames",
        ])
        self.assertTrue(self.centroid_signal.validate(empty))

    # ── Graceful NaN Handling ─────────────────────────────────────────

    def test_polarisation_handles_missing_possession(self):
        """Polarisation doesn't crash when team_in_possession is all NaN."""
        df = self.match_df.copy()
        df["team_in_possession"] = np.nan
        result = self.polarisation_signal.compute(
            df, self.blocks, game_id="test_noposs"
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_centroid_distance_handles_missing_possession(self):
        """Centroid distance doesn't crash when team_in_possession is all NaN."""
        df = self.match_df.copy()
        df["team_in_possession"] = np.nan
        result = self.centroid_signal.compute(
            df, self.blocks, game_id="test_noposs"
        )
        # Both teams are treated as out-of-possession, so both get distances
        self.assertIsInstance(result, pd.DataFrame)

    # ── Output Integrity ──────────────────────────────────────────────

    def test_polarisation_team_level_output(self):
        """Polarisation is a team-level signal: player_id should be 0."""
        result = self.polarisation_signal.compute(
            self.aligned_df, self.aligned_blocks, game_id="test_team"
        )
        if len(result) > 0:
            unique_pids = result["player_id"].unique()
            self.assertEqual(len(unique_pids), 1)
            self.assertEqual(unique_pids[0], 0)

    def test_centroid_distance_per_player_output(self):
        """Centroid distance is per-player: unique player_ids > 0."""
        result = self.centroid_signal.compute(
            self.match_df, self.blocks, game_id="test_player"
        )
        if len(result) > 0:
            pids = result["player_id"].unique()
            self.assertGreater(len(pids), 1)
            self.assertTrue(all(pid >= 0 for pid in pids))


if __name__ == "__main__":
    unittest.main(verbosity=2)
