#!/usr/bin/env python3
"""Lightweight test of the two new signals using synthetic data.

This test uses small DataFrames (5 frames, 6 players, 2 teams) to
verify that the compute() methods run without errors and produce the
expected output schemas.

No real data is loaded — safe for the Raspberry Pi.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Import signals (triggers registration)
import src.signals.polarisation  # noqa: F401
import src.signals.team_centroid_distance  # noqa: F401

from src.signals.registry import discover_signals


def make_synthetic_blocks(n_frames: int = 10, n_players: int = 6):
    """Create synthetic block DataFrames for testing.

    Two teams: home (players 1-3) and away (players 4-6).
    Players move in roughly the same direction (high polarisation)
    for team home, and in random directions (low polarisation) for
    team away.
    """
    np.random.seed(42)

    rows = []
    for frame in range(n_frames):
        # Team home: all move roughly rightward → high polarisation
        for pid in [1, 2, 3]:
            vx = 2.0 + np.random.normal(0, 0.1)  # mostly right
            vy = 0.0 + np.random.normal(0, 0.1)
            rows.append({
                "frame": frame,
                "player_id": f"home_{pid}",
                "team": "home",
                "x": 10.0 + frame * 0.5 + np.random.normal(0, 0.2),
                "y": 0.0 + pid * 5 + np.random.normal(0, 0.2),
                "vx": vx,
                "vy": vy,
                "speed": np.sqrt(vx**2 + vy**2),
            })

        # Team away: random directions → low polarisation
        for pid in [4, 5, 6]:
            vx = np.random.uniform(-3, 3)
            vy = np.random.uniform(-3, 3)
            rows.append({
                "frame": frame,
                "player_id": f"away_{pid}",
                "team": "away",
                "x": 50.0 + np.random.normal(0, 3),
                "y": np.random.uniform(-20, 20),
                "vx": vx,
                "vy": vy,
                "speed": np.sqrt(vx**2 + vy**2),
            })

    df = pd.DataFrame(rows)

    # Create two blocks from this data
    # (just split frames in half)
    half = n_frames // 2
    block1 = df[df["frame"] < half].copy()
    block1["block_id"] = 1
    block2 = df[df["frame"] >= half].copy()
    block2["block_id"] = 2

    return [block1, block2]


def test_polarisation():
    """Test the team_polarisation signal with synthetic data."""
    print("\n" + "=" * 60)
    print("TEST: Team Polarisation Signal")
    print("=" * 60)

    blocks = make_synthetic_blocks(n_frames=10, n_players=6)
    signals = discover_signals()
    cls = signals["team_polarisation"]
    sig = cls()

    result = sig.compute(blocks, match_id=1)

    # Check output
    expected_cols = [
        "game_id", "block_id", "phase", "player_id",
        "team_id_opta", "signal_name", "signal_value", "n_frames",
    ]

    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"

    assert result["signal_name"].iloc[0] == "team_polarisation"
    assert result["game_id"].iloc[0] == 1
    assert len(result) == 4  # 2 teams x 2 blocks

    # Team home should have HIGHER polarisation (all moving right)
    # Team away should have LOWER polarisation (random directions)
    home_vals = result[result["team_id_opta"] == "home"]["signal_value"].values
    away_vals = result[result["team_id_opta"] == "away"]["signal_value"].values

    print(f"Home polarisation values: {home_vals}")
    print(f"Away polarisation values: {away_vals}")

    for hv in home_vals:
        assert hv > 0.8, f"Home polarisation {hv} should be > 0.8"
    for av in away_vals:
        assert av < 0.9, f"Away polarisation {av} should be < 0.9"

    # Home should be more polarised than away (all moving in same direction)
    assert home_vals.mean() > away_vals.mean(), (
        f"Home polarisation ({home_vals.mean():.3f}) should be > "
        f"away ({away_vals.mean():.3f})"
    )

    print("✅ Polarisation test PASSED")
    return result


def test_centroid_distance():
    """Test the team_centroid_distance signal with synthetic data."""
    print("\n" + "=" * 60)
    print("TEST: Team Centroid Distance Signal")
    print("=" * 60)

    blocks = make_synthetic_blocks(n_frames=10, n_players=6)
    signals = discover_signals()
    cls = signals["team_centroid_distance"]
    sig = cls()

    result = sig.compute(blocks, match_id=1)

    expected_cols = [
        "game_id", "block_id", "phase", "player_id",
        "team_id_opta", "signal_name", "signal_value", "n_frames",
    ]

    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"

    assert result["signal_name"].iloc[0] == "team_centroid_distance"
    assert result["game_id"].iloc[0] == 1

    # Should have 6 players x 2 blocks = 12 rows (3 home + 3 away)
    n_home = len(result[result["team_id_opta"] == "home"])
    n_away = len(result[result["team_id_opta"] == "away"])
    print(f"Home players: {n_home}, Away players: {n_away}")
    print(f"Total rows: {len(result)}")
    assert n_home == 6, f"Expected 6 home records, got {n_home}"  # 3 players x 2 blocks
    assert n_away == 6, f"Expected 6 away records, got {n_away}"  # 3 players x 2 blocks

    # All distances should be positive
    assert (result["signal_value"] >= 0).all(), "Distances should be non-negative"

    # Synthetic data: away team has wider spread → larger distances on avg
    home_mean = result[result["team_id_opta"] == "home"]["signal_value"].mean()
    away_mean = result[result["team_id_opta"] == "away"]["signal_value"].mean()
    print(f"Home mean distance: {home_mean:.3f}")
    print(f"Away mean distance: {away_mean:.3f}")

    print("✅ Centroid distance test PASSED")
    return result


def test_empty_blocks():
    """Test signals with empty blocks (edge case)."""
    print("\n" + "=" * 60)
    print("TEST: Empty blocks edge case")
    print("=" * 60)

    signals = discover_signals()

    for name in signals:
        sig = signals[name]()
        result = sig.compute([], match_id=1)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        print(f"✅ {name}: empty blocks handled correctly")


def test_no_velocity():
    """Test polarisation handles missing velocity data gracefully."""
    print("\n" + "=" * 60)
    print("TEST: Missing velocity data")
    print("=" * 60)

    rows = []
    for frame in range(5):
        for team, pid in [("home", 1), ("home", 2), ("away", 3), ("away", 4)]:
            rows.append({
                "frame": frame,
                "player_id": f"{team}_{pid}",
                "team": team,
                "x": float(frame) * 0.5 + np.random.normal(0, 0.1),
                "y": float(pid) + np.random.normal(0, 0.1),
                # No vx/vy columns present
            })

    df = pd.DataFrame(rows)
    block = df.copy()
    block["block_id"] = 1
    blocks = [block]

    signals = discover_signals()

    # Polarisation without vx/vy should return records with NaN
    sig_pol = signals["team_polarisation"]()
    result = sig_pol.compute(blocks, match_id=1)
    assert not result.empty
    assert result["signal_value"].isna().all()
    print(f"✅ Polarisation: no-velocity handled — all NaN values (n={len(result)})")

    # Centroid distance should still work without vx/vy
    sig_cd = signals["team_centroid_distance"]()
    result = sig_cd.compute(blocks, match_id=1)
    assert not result.empty
    assert not result["signal_value"].isna().all()
    print(f"✅ Centroid distance: works without velocity data (n={len(result)})")


if __name__ == "__main__":
    print("=" * 60)
    print("New Signals Test Suite")
    print("=" * 60)

    try:
        test_polarisation()
        test_centroid_distance()
        test_empty_blocks()
        test_no_velocity()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
