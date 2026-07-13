"""Core Bekkers Time-To-Intercept (TTI) computation for Signal 3.

Implements the formula from Bekkers et al. (EFPI paper, arXiv:2506.23843):

    T = τᵣ + τ_dist + τ_β

where:
    τᵣ     = perceptual reaction time (configurable, default 0.2 s)
    τ_dist = d_opponent / v_defender
    τ_β    = angle adjustment factor penalising off-angle movement

The intercept probability is then obtained via a logistic transform:

    P(intercept) = 1 / (1 + exp(-k * (T_threshold - T)))
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PressingConfig

# ── Module-level constants ──────────────────────────────────────────────────

_DEFAULT_TTA_THRESHOLD: float = 1.33
"""Default TTA (Time-To-Arrive) threshold for a 20 m pass at 15 m/s."""

# ── Public Functions ───────────────────────────────────────────────────────


def compute_tta_threshold(
    pass_speed: float = 15.0,
    pass_distance: float = 20.0,
) -> float:
    """Compute the TTA (Time-To-Arrive) threshold based on pass speed.

    Represents how long the ball takes to travel from passer to receiver.
    A defender must reach the attacker within this window to intercept.

    Parameters
    ----------
    pass_speed : float, default 15.0
        Typical pass speed (m/s). Professional range ~10–25 m/s.
    pass_distance : float, default 20.0
        Typical pass distance (m). Average ~15–25 m in open play.

    Returns
    -------
    float
        TTA threshold in seconds.
    """
    if pass_speed <= 0:
        raise ValueError(f"pass_speed must be positive, got {pass_speed}")
    if pass_distance <= 0:
        raise ValueError(f"pass_distance must be positive, got {pass_distance}")

    return pass_distance / pass_speed


def compute_tti(
    df: pd.DataFrame,
    config: PressingConfig,
    own_team_id: int,
    opponent_team_id: int,
) -> pd.DataFrame:
    """Compute Bekkers Time-To-Intercept for all defender-attacker pairs.

    For each frame:
    1. Identify defenders (``own_team_id``) and attackers
       (``opponent_team_id``), excluding goalkeepers.
    2. For every defender-attacker pair within ``max_pair_distance``:
       a. Euclidean distance ``d_opponent``.
       b. ``τ_dist = d_opponent / v_defender`` (with speed guard).
       c. ``τ_β`` based on the angle between the defender's velocity
          vector and the direction to the attacker.
       d. ``T = τᵣ + τ_dist + τ_β``.
       e. ``intercept_prob = 1 / (1 + exp(-k * (T_threshold - T)))``.
    3. For each defender per frame, keep only the *minimum* T (i.e. the
       closest attacker threat).

    Parameters
    ----------
    df : pd.DataFrame
        Tracking data. Must include columns:
        ``frame_count``, ``player_id``, ``team_id_opta``,
        ``x``, ``y``, ``vx_smooth``, ``vy_smooth``.
        Optionally ``goalkeeper`` (bool) or ``jersey_number`` (int)
        for goal-keeper filtering.
    config : PressingConfig
        Configuration with TTI parameters.
    own_team_id : int
        Team ID of the defending / pressing team.
    opponent_team_id : int
        Team ID of the attacking team.

    Returns
    -------
    pd.DataFrame
        Per-frame, per-defender TTI results with columns:

        - ``frame_count``: Frame identifier.
        - ``player_id``: Defender's player ID.
        - ``closest_attacker_id``: ID of nearest (in TTI terms) attacker.
        - ``closest_attacker_distance``: Euclidean distance (m) to that attacker.
        - ``tti_value``: T = τᵣ + τ_dist + τ_β (seconds).
        - ``intercept_probability``: Logistic-transformed probability in [0, 1].
    """
    # ── Validate inputs ────────────────────────────────────────────────
    required_cols = [
        "frame_count", "player_id", "team_id_opta",
        "x", "y", "vx_smooth", "vy_smooth",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame missing required columns: {missing}"
        )

    # ── Subset and split teams ─────────────────────────────────────────
    df = df[required_cols + _optional_cols(df)].copy()

    defenders = df[df["team_id_opta"] == own_team_id]
    attackers = df[df["team_id_opta"] == opponent_team_id]

    # Filter goalkeepers from both sets
    defenders = _filter_goalkeepers(defenders)
    attackers = _filter_goalkeepers(attackers)

    # ── Rename for merge ───────────────────────────────────────────────
    def_for_merge = defenders[["frame_count", "player_id", "x", "y",
                                "vx_smooth", "vy_smooth"]].rename(
        columns={
            "player_id": "defender_id",
            "x": "def_x",
            "y": "def_y",
            "vx_smooth": "def_vx",
            "vy_smooth": "def_vy",
        }
    )

    att_for_merge = attackers[["frame_count", "player_id", "x", "y"]].rename(
        columns={
            "player_id": "attacker_id",
            "x": "att_x",
            "y": "att_y",
        }
    )

    # ── Cross-join on frame_count ──────────────────────────────────────
    pairs = def_for_merge.merge(att_for_merge, on="frame_count", how="inner")

    if len(pairs) == 0:
        # No valid defender-attacker pairs — return empty result
        return pd.DataFrame(
            columns=[
                "frame_count", "player_id", "closest_attacker_id",
                "closest_attacker_distance", "tti_value",
                "intercept_probability",
            ]
        )

    # ── Compute distances ──────────────────────────────────────────────
    dx = pairs["att_x"] - pairs["def_x"]
    dy = pairs["att_y"] - pairs["def_y"]
    distance = np.sqrt(dx ** 2 + dy ** 2)

    # Filter by max_pair_distance before further computation
    mask_dist = distance <= config.max_pair_distance
    pairs = pairs[mask_dist].copy()
    dx = dx[mask_dist]
    dy = dy[mask_dist]
    distance = distance[mask_dist]

    if len(pairs) == 0:
        return pd.DataFrame(
            columns=[
                "frame_count", "player_id", "closest_attacker_id",
                "closest_attacker_distance", "tti_value",
                "intercept_probability",
            ]
        )

    # ── Defender speed ─────────────────────────────────────────────────
    def_speed = np.sqrt(pairs["def_vx"] ** 2 + pairs["def_vy"] ** 2)
    v_clamped = np.maximum(def_speed, config.speed_guard)

    # ── τ_dist = d_opponent / v_defender ───────────────────────────────
    tau_dist = distance / v_clamped

    # ── τ_β — angle adjustment factor ───────────────────────────────────
    # cos(θ) where θ = angle between defender velocity vector and the
    # vector from defender to attacker.
    #   cos(θ) = (v · dir_to_att) / (|v| * |dir_to_att|)
    #
    # When cos(θ) = 1  (θ = 0°)   → defender moves directly at attacker
    # When cos(θ) = 0  (θ = 90°)  → defender moves perpendicular
    # When cos(θ) = -1 (θ = 180°) → defender moves away from attacker
    dot = pairs["def_vx"] * dx + pairs["def_vy"] * dy
    cos_theta = dot / (v_clamped * distance)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # τ_β penalises off-angle movement proportional to τ_dist
    # When running directly at attacker (cos=1): τ_β = 0
    # When perpendicular (cos=0): τ_β = β * τ_dist
    # When running away (cos=-1): τ_β = 2 * β * τ_dist
    tau_beta = config.beta_scaling * (1.0 - cos_theta) * tau_dist

    # ── T = τᵣ + τ_dist + τ_β ─────────────────────────────────────────
    tti_value = config.reaction_time_s + tau_dist + tau_beta

    # ── Intercept probability ──────────────────────────────────────────
    tta_threshold = compute_tta_threshold()
    intercept_prob = 1.0 / (
        1.0 + np.exp(-config.tti_steepness_k * (tta_threshold - tti_value))
    )

    pairs["distance"] = distance
    pairs["tti_value"] = tti_value
    pairs["intercept_probability"] = intercept_prob

    # ── Keep minimum T per defender per frame ──────────────────────────
    idx_min = pairs.groupby(["frame_count", "defender_id"])["tti_value"].idxmin()
    result = pairs.loc[idx_min].copy()

    result = result.rename(
        columns={
            "defender_id": "player_id",
            "attacker_id": "closest_attacker_id",
            "distance": "closest_attacker_distance",
        }
    )

    output_cols = [
        "frame_count", "player_id", "closest_attacker_id",
        "closest_attacker_distance", "tti_value", "intercept_probability",
    ]

    return result[output_cols].reset_index(drop=True)


def compute_angle_to_attacker(
    def_vx: np.ndarray,
    def_vy: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    speed_guard: float = 0.1,
) -> np.ndarray:
    """Compute the angle (in radians) between a defender's velocity vector
    and the direction to the attacker.

    Parameters
    ----------
    def_vx : np.ndarray
        Defender x-velocity components.
    def_vy : np.ndarray
        Defender y-velocity components.
    dx : np.ndarray
        x-displacement from defender to attacker.
    dy : np.ndarray
        y-displacement from defender to attacker.
    speed_guard : float, default 0.1
        Minimum speed floor to avoid division by zero.

    Returns
    -------
    np.ndarray
        Angles in radians, ranging [0, π].
    """
    v_mag = np.maximum(np.sqrt(def_vx ** 2 + def_vy ** 2), speed_guard)
    dist = np.maximum(np.sqrt(dx ** 2 + dy ** 2), 1e-6)
    cos_theta = (def_vx * dx + def_vy * dy) / (v_mag * dist)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


# ── Internal Helpers ────────────────────────────────────────────────────────


def _optional_cols(df: pd.DataFrame) -> list[str]:
    """Return names of optional columns that exist in the DataFrame."""
    known = {"goalkeeper", "jersey_number"}
    return [c for c in known if c in df.columns]


def _filter_goalkeepers(team_df: pd.DataFrame) -> pd.DataFrame:
    """Remove goalkeepers from a team DataFrame.

    Uses the ``goalkeeper`` boolean column if available; otherwise falls
    back to filtering by ``jersey_number != 1``.

    Parameters
    ----------
    team_df : pd.DataFrame
        Subset of tracking data for one team.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with goalkeeper rows removed.
    """
    if "goalkeeper" in team_df.columns:
        return team_df[~team_df["goalkeeper"]].copy()
    if "jersey_number" in team_df.columns:
        return team_df[team_df["jersey_number"] != 1].copy()
    # No GK marker available — return as-is (risk of including GK)
    return team_df.copy()
