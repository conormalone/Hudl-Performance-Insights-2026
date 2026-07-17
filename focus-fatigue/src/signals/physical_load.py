"""Signal 5 — Physical Load.

Measures physical exertion for each player per block using tracking data.
Derives distance covered, high-speed running metrics, and speed statistics
from per-frame (x,y) positions and v_mag velocity data.

The core hypothesis is that physical load acts as a confounding control
variable when modelling cognitive fatigue effects — players who run more
may exhibit worse positional drift / pressing accuracy simply due to
metabolic fatigue rather than mental fatigue.

One file contains: config, per-frame displacement computation, speed
band classification, HSR effort detection, block aggregation, and the
registered signal class.
"""

from __future__ import annotations
from dataclasses import dataclass

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import SignalBase
from .registry import register_signal


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicalLoadConfig:
    """Configuration for physical load computation.

    Attributes
    ----------
    hsr_threshold : float
        Minimum v_mag (m/s) to count as high-speed running (default 7.0).
    sprint_threshold : float
        Minimum v_mag (m/s) to count as sprinting (default 8.0).
    hsr_effort_min_frames : int
        Minimum continuous frames at HSR to count as one effort (default 25;
        equivalent to 1 second at 25 fps).
    frames_per_second : int
        Frame rate of the tracking data (default 25).
    max_acceleration_m_s2 : float
        Maximum plausible acceleration for sanity-clipping frame-to-frame
        displacement (default 15.0). Displacements implying > this value
        are treated as noise and set to 0.
    """
    hsr_threshold: float = 7.0
    sprint_threshold: float = 8.0
    hsr_effort_min_frames: int = 25
    frames_per_second: int = 25
    max_acceleration_m_s2: float = 15.0


DEFAULT_PHYSICAL_LOAD_CONFIG = PhysicalLoadConfig()


# ═══════════════════════════════════════════════════════════════════════════
# Per-frame Physical Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_frame_displacements(
    player_df: pd.DataFrame,
    config: PhysicalLoadConfig,
) -> pd.DataFrame:
    """Compute per-frame displacement from (x,y) positions for a single player.

    Parameters
    ----------
    player_df : pd.DataFrame
        Tracking data for one player, sorted by ``frame_count``.
        Must contain ``x``, ``y`` columns.
    config : PhysicalLoadConfig

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        - ``displacement_m``: frame-to-frame Euclidean distance in metres
        - ``hsr``: boolean, v_mag >= hsr_threshold
        - ``sprinting``: boolean, v_mag >= sprint_threshold
    """
    df = player_df.copy()

    x = df["x"].values.astype(np.float64)
    y = df["y"].values.astype(np.float64)

    # Frame-to-frame displacement
    dx = np.diff(x, prepend=np.nan)
    dy = np.diff(y, prepend=np.nan)
    displacement = np.sqrt(dx**2 + dy**2)

    # Sanity clip: max plausible displacement per frame at 25fps
    # 15 m/s² acceleration → ~0.15 m/frame at 25fps with dt=0.04s
    # More generously cap at v_mag * dt
    dt = 1.0 / config.frames_per_second
    max_plausible = config.max_acceleration_m_s2 * dt

    df["displacement_m"] = np.where(
        np.isfinite(displacement) & (displacement <= max_plausible * 10),
        displacement,
        0.0,
    )

    # Speed band classification from v_mag
    if "v_mag" in df.columns:
        v = df["v_mag"].values.astype(np.float64)
        v_safe = np.where(np.isfinite(v), v, 0.0)
    elif "speed" in df.columns:
        v = df["speed"].values.astype(np.float64)
        v_safe = np.where(np.isfinite(v), v, 0.0)
    else:
        # Compute from vx_smooth / vy_smooth if available
        if "vx_smooth" in df.columns and "vy_smooth" in df.columns:
            v_safe = np.sqrt(
                df["vx_smooth"].fillna(0.0).values.astype(np.float64)**2
                + df["vy_smooth"].fillna(0.0).values.astype(np.float64)**2
            )
        else:
            v_safe = np.zeros(len(df), dtype=np.float64)

    df["hsr"] = v_safe >= config.hsr_threshold
    df["sprinting"] = v_safe >= config.sprint_threshold

    return df


def detect_hsr_efforts(
    df: pd.DataFrame,
    config: PhysicalLoadConfig,
) -> pd.DataFrame:
    """Detect continuous high-speed running efforts from boolean hsr column.

    An HSR effort is a contiguous segment of ``hsr == True`` spanning at
    least ``hsr_effort_min_frames`` frames.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``hsr`` boolean column (sorted by frame_count).
    config : PhysicalLoadConfig

    Returns
    -------
    pd.DataFrame
        Original DataFrame with an added ``hsr_effort_id`` column:
        - ``0`` for non-HSR frames
        - Positive integer id for each distinct HSR effort
    """
    df = df.copy()
    hsr = df["hsr"].values.astype(np.int8)

    # Find edges: leading with 0 means non-HSR → HSR transition
    edges = np.diff(hsr, prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    effort_id = np.zeros(len(df), dtype=np.int64)
    effort_counter = 0

    for s, e in zip(starts, ends):
        duration = e - s
        if duration >= config.hsr_effort_min_frames:
            effort_counter += 1
            effort_id[s:e] = effort_counter

    df["hsr_effort_id"] = effort_id
    return df


def compute_physical_metrics_frame(
    match_df: pd.DataFrame,
    config: PhysicalLoadConfig,
) -> pd.DataFrame:
    """Compute all per-frame physical metrics for every player in a match.

    Parameters
    ----------
    match_df : pd.DataFrame
        Full tracking DataFrame for one match. Must contain ``player_id``,
        ``team_id_opta``, ``frame_count``, ``x``, ``y``, and either
        ``v_mag`` or ``speed`` or ``vx_smooth``/``vy_smooth`` columns.
    config : PhysicalLoadConfig

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with physical metrics columns:
        ``displacement_m``, ``hsr``, ``sprinting``, ``hsr_effort_id``.
    """
    # Process per-player to get correct displacements (frame-to-frame)
    players = match_df["player_id"].unique()
    frames = []

    for pid in players:
        mask = match_df["player_id"] == pid
        p_df = match_df[mask].sort_values("frame_count").copy()
        p_df = compute_frame_displacements(p_df, config)
        p_df = detect_hsr_efforts(p_df, config)
        frames.append(p_df)

    if not frames:
        result = match_df.copy()
        for col in ["displacement_m", "hsr", "sprinting"]:
            result[col] = 0
        result["hsr_effort_id"] = 0
        return result

    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# Block Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_physical_load_by_block(
    df: pd.DataFrame,
    blocks: list[dict[str, Any] | pd.DataFrame],
    config: PhysicalLoadConfig,
    game_id: str = "",
) -> pd.DataFrame:
    """Aggregate per-frame physical metrics to per-block, per-player summaries.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with per-frame physical metrics (output of
        ``compute_physical_metrics_frame``).
    blocks : list[dict | pd.DataFrame]
        List of block definitions. Each element can be:
        - DataFrame with columns ``block_id``, ``frame_count``/``frame``
        - dict with keys ``block_id``, ``start_frame``, ``end_frame``, ``phase``
    config : PhysicalLoadConfig
    game_id : str
        Match identifier.

    Returns
    -------
    pd.DataFrame
        Standardised signal output with columns:
        - Required: game_id, block_id, phase, player_id, team_id_opta,
          signal_name="physical_load", signal_value=total_distance, n_frames
        - Extra: hsr_distance, sprint_distance, hsr_efforts, avg_speed, max_speed
    """
    # Normalise blocks to a list of dicts with consistent keys
    block_defs: list[dict[str, Any]] = []
    for blk in blocks:
        if isinstance(blk, dict):
            bid = str(blk["block_id"])
            ph = int(blk.get("phase", bid.split("_")[0]))
            sf = int(blk.get("start_frame", 0))
            ef = int(blk.get("end_frame", 0))
        else:
            # DataFrame
            bid = str(blk["block_id"].iloc[0])
            ph = int(bid.split("_")[0])
            fc_col = "frame_count" if "frame_count" in blk.columns else "frame"
            sf = int(blk[fc_col].min())
            ef = int(blk[fc_col].max())
        block_defs.append({"block_id": bid, "phase": ph,
                           "start_frame": sf, "end_frame": ef})

    records: list[dict[str, Any]] = []
    players_in_df = df[["player_id", "team_id_opta"]].drop_duplicates("player_id")

    for bd in block_defs:
        bmask = (
            df["frame_count"].between(bd["start_frame"], bd["end_frame"], inclusive="left")
            & (df["frame_count"] >= bd["start_frame"])
            & (df["frame_count"] < bd["end_frame"])
        )
        if not bmask.any():
            continue

        bdf = df[bmask]
        block_nframes = bdf["frame_count"].nunique()

        for _, prow in players_in_df.iterrows():
            pid = int(prow["player_id"])
            to = int(prow["team_id_opta"])

            pbmask = bdf["player_id"] == pid
            if not pbmask.any():
                continue

            pdf = bdf[pbmask]

            # Total distance
            total_dist = float(pdf["displacement_m"].sum())

            # HSR / sprint distances (use v_mag-weighted displacement)
            # Displacement during HSR-only frames
            hsr_mask = pdf["hsr"] == True  # noqa: E712
            hsr_dist = float(pdf.loc[hsr_mask, "displacement_m"].sum()) if hsr_mask.any() else 0.0

            sprint_mask = pdf["sprinting"] == True  # noqa: E712
            sprint_dist = float(pdf.loc[sprint_mask, "displacement_m"].sum()) if sprint_mask.any() else 0.0

            # HSR efforts (unique effort ids in this player-block)
            hsr_effort_ids = pdf.loc[hsr_mask, "hsr_effort_id"].unique()
            n_hsr_efforts = int((hsr_effort_ids > 0).sum())

            # Speed statistics
            vmag_col = None
            for c in ["v_mag", "speed"]:
                if c in pdf.columns:
                    vmag_col = c
                    break
            if vmag_col is None and "vx_smooth" in pdf.columns and "vy_smooth" in pdf.columns:
                vx = pdf["vx_smooth"].fillna(0.0).astype(np.float64)
                vy = pdf["vy_smooth"].fillna(0.0).astype(np.float64)
                pdf = pdf.copy()
                pdf["_vmag"] = np.sqrt(vx**2 + vy**2)
                vmag_col = "_vmag"

            if vmag_col is not None:
                v = pdf[vmag_col].fillna(0.0).astype(np.float64)
                avg_speed = float(v.mean())
                max_speed = float(v.max())
            else:
                avg_speed = 0.0
                max_speed = 0.0

            records.append({
                "game_id": game_id,
                "block_id": bd["block_id"],
                "phase": bd["phase"],
                "player_id": pid,
                "team_id_opta": to,
                "signal_name": "physical_load",
                "signal_value": round(total_dist, 3),
                "n_frames": block_nframes,
                "total_distance": round(total_dist, 3),
                "hsr_distance": round(hsr_dist, 3),
                "sprint_distance": round(sprint_dist, 3),
                "hsr_efforts": n_hsr_efforts,
                "avg_speed": round(avg_speed, 3),
                "max_speed": round(max_speed, 3),
            })

    if not records:
        cols = [
            "game_id", "block_id", "phase", "player_id", "team_id_opta",
            "signal_name", "signal_value", "n_frames",
            "total_distance", "hsr_distance", "sprint_distance",
            "hsr_efforts", "avg_speed", "max_speed",
        ]
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(records)
    # Ensure integer types
    for c in ["player_id", "team_id_opta", "phase", "n_frames", "hsr_efforts"]:
        out[c] = out[c].astype(int)
    for c in ["game_id", "block_id", "signal_name"]:
        out[c] = out[c].astype(str)
    return out.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Registered Signal Class
# ═══════════════════════════════════════════════════════════════════════════

@register_signal
class PhysicalLoadSignal(SignalBase):
    """Physical load signal — distance, HSR, and speed per block per player.

    Computes per-player physical metrics from tracking data (x, y, v_mag)
    and aggregates to block-level summaries. Intended as a confounding
    control variable for cognitive fatigue models.
    """

    signal_name = "physical_load"

    def __init__(
        self,
        signal_config=None,
        physical_config: PhysicalLoadConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(config=signal_config, logger=logger)
        self.physical_config = physical_config or DEFAULT_PHYSICAL_LOAD_CONFIG

    def compute(
        self,
        match_df: pd.DataFrame,
        blocks: list[dict[str, Any]] | list[pd.DataFrame],
        *,
        game_id: str = "",
    ) -> pd.DataFrame:
        """Compute physical load metrics per block per player.

        Parameters
        ----------
        match_df : pd.DataFrame
            Full tracking data for a single match. Must contain columns:
            ``player_id``, ``team_id_opta``, ``frame_count``, ``x``, ``y``,
            and either ``v_mag`` or ``speed`` or ``vx_smooth``/``vy_smooth``.
        blocks : list[dict | pd.DataFrame]
            Block definitions. See ``aggregate_physical_load_by_block``.
        game_id : str
            Match identifier.

        Returns
        -------
        pd.DataFrame
            Standard signal output — see ``aggregate_physical_load_by_block``.
        """
        if len(match_df) == 0:
            cols = [
                "game_id", "block_id", "phase", "player_id", "team_id_opta",
                "signal_name", "signal_value", "n_frames",
                "total_distance", "hsr_distance", "sprint_distance",
                "hsr_efforts", "avg_speed", "max_speed",
            ]
            return pd.DataFrame(columns=cols)

        cfg = self.physical_config

        # Compute per-frame physical metrics
        physical_df = compute_physical_metrics_frame(match_df, cfg)
        self.logger.info(
            "Computed physical metrics for %d players across %d frames",
            match_df["player_id"].nunique(),
            len(physical_df),
        )

        # Aggregate per block
        result = aggregate_physical_load_by_block(
            physical_df, blocks, cfg, game_id=game_id,
        )
        self.logger.info("Aggregated %d player-block rows", len(result))
        return result

    def validate(self, output_df: pd.DataFrame) -> bool:
        """Validate physical load output.

        Checks:
        - Standard schema compliance (via base class)
        - No negative distances or speeds
        - total_distance >= hsr_distance >= sprint_distance
        - n_frames within plausible range
        """
        super().validate(output_df)
        if len(output_df) == 0:
            return True

        # Non-negative metrics
        for col in ["signal_value", "total_distance", "hsr_distance",
                     "sprint_distance", "avg_speed", "max_speed"]:
            if col in output_df.columns and output_df[col].min() < 0:
                raise ValueError(
                    f"Column '{col}' contains negative values "
                    f"(min={output_df[col].min():.3f})"
                )

        # Distance hierarchy: total >= hsr >= sprint
        if "total_distance" in output_df.columns and "hsr_distance" in output_df.columns:
            violations = (output_df["hsr_distance"] > output_df["total_distance"] + 0.01).sum()
            if violations > 0:
                raise ValueError(
                    f"{violations} row(s) have hsr_distance > total_distance"
                )

        if "hsr_distance" in output_df.columns and "sprint_distance" in output_df.columns:
            violations = (output_df["sprint_distance"] > output_df["hsr_distance"] + 0.01).sum()
            if violations > 0:
                raise ValueError(
                    f"{violations} row(s) have sprint_distance > hsr_distance"
                )

        # n_frames bounds
        if "n_frames" in output_df.columns:
            nf_min = output_df["n_frames"].min()
            nf_max = output_df["n_frames"].max()
            if nf_max > 100000:
                raise ValueError(
                    f"n_frames seems unreasonably large: max={nf_max}"
                )
            if nf_min < 0:
                raise ValueError(f"n_frames has negative values (min={nf_min})")

        # HSR efforts
        if "hsr_efforts" in output_df.columns:
            if output_df["hsr_efforts"].min() < 0:
                raise ValueError(
                    f"hsr_efforts has negative values "
                    f"(min={output_df['hsr_efforts'].min()})"
                )

        # Max speed sanity
        if "max_speed" in output_df.columns:
            max_s = output_df["max_speed"].max()
            if max_s > 15.0:
                self.logger.warning(
                    "max_speed=%.1f m/s is unusually high; verify tracking data",
                    max_s,
                )

        return True
