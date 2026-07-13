"""Configuration parameters for Signal 3 — Pressing Accuracy (Bekkers TTI).

All tuning parameters are collected here for easy adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PressingConfig:
    """Configuration for pressing accuracy using Bekkers Time-To-Intercept.

    Controls the TTI formula parameters, pressing detection criteria,
    and accuracy classification thresholds.
    """

    # ── TTI Parameters ─────────────────────────────────────────────────

    reaction_time_s: float = 0.2
    """τᵣ — Perceptual / reaction delay (seconds). Default 0.2 s."""

    tti_threshold_s: float = 1.5
    """T below this value means the defender can intercept before the
    attacker receives the pass (seconds)."""

    beta_scaling: float = 1.0
    """τ_β multiplier controlling how heavily angle misalignment penalises
    the defender's estimated intercept time."""

    # ── Pressing Detection ─────────────────────────────────────────────

    press_speed_threshold: float = 2.0
    """Minimum defender speed (m/s) to be considered actively pressing."""

    press_angle_threshold: float = 45.0
    """Maximum angle (degrees) between a defender's velocity vector and
    the direction to the attacker for the movement to count as
    'pressing toward' the opponent."""

    # ── Accuracy Classification ────────────────────────────────────────

    correct_press_threshold: float = 0.18
    """Intercept probability above which a press is classified as
    'correct' (value in [0, 1]).
    
    Calibrated from 0.3 → 0.18 based on validation: 0.3 gave 26.4%
    accuracy vs 30-60% expected range. 0.18 should bring it closer
    to ~40-50% (FIX: threshold calibration).
    """

    # ── Frame Rate ─────────────────────────────────────────────────────

    frames_per_second: int = 25
    """Frame rate of the tracking data (Hz)."""

    # ── Pairwise Distance Parameters ───────────────────────────────────

    max_pair_distance: float = 30.0
    """Only consider defender-attacker pairs within this Euclidean
    distance (metres). Reduces noise from irrelevant pairings."""

    # ── Speed Guard ────────────────────────────────────────────────────

    speed_guard: float = 0.1
    """Minimum speed (m/s) used as a floor when computing τ_dist to
    avoid division-by-zero or extreme values from stationary players."""

    # ── TTI Steepness ──────────────────────────────────────────────────

    tti_steepness_k: float = 3.0
    """Steepness parameter k for the logistic function used to convert
    TTI value into an intercept probability:
        P = 1 / (1 + exp(-k * (T_threshold - T)))
    """


# Default global config instance
DEFAULT_PRESSING_CONFIG = PressingConfig()
