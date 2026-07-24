#!/usr/bin/env python3
"""Capture baseline signal outputs for a single match (pre-optimisation).

Saves polarisation and centroid_distance outputs for verification.
"""
import sys, time, pickle, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders import load_tracking_statsperform
from src.smoothing import smooth_trajectory, compute_velocity_features
from src.segments import split_into_blocks
import src.signals.polarisation
import src.signals.team_centroid_distance
from src.signals.registry import SIGNAL_REGISTRY

DATA_ROOT = Path("/mnt/usb/conor_downloads/team_mappings")

def capture(match_id="2215790", nrows=500000):
    tracking_path = DATA_ROOT / "sample" / match_id / "tracking.parquet"
    if not tracking_path.exists():
        tracking_path = DATA_ROOT / "tracking" / match_id / "tracking.parquet"
    
    df = load_tracking_statsperform(str(tracking_path), match_id=match_id, 
                                     normalise_dop=True, include_ball=True)
    if nrows: df = df.head(nrows)
    df["frame"] = df["frame_count"]
    df = smooth_trajectory(df, inplace=False)
    compute_velocity_features(df, inplace=True)
    blocks = split_into_blocks(df, window_minutes=5, min_frames=100)
    
    print(f"Match: {match_id}, {len(df):,} rows, {len(blocks)} blocks")
    
    baseline = {}
    for sig_name in ["team_polarisation", "team_centroid_distance"]:
        sig_cls = SIGNAL_REGISTRY[sig_name]
        signal = sig_cls()
        output = signal.compute(match_df=df, blocks=blocks, game_id=match_id)
        baseline[sig_name] = {
            "rows": len(output),
            "cols": list(output.columns),
            "dtypes": {c: str(output[c].dtype) for c in output.columns},
            "values_hash": hash(tuple(output["signal_value"].round(6).values)),
            "sample": output.head(3).to_dict("records"),
        }
        print(f"  {sig_name}: {len(output)} rows, hash={baseline[sig_name]['values_hash']}")
    
    # Save
    out_path = PROJECT_ROOT / "outputs" / "baseline" / f"baseline_{match_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2, default=str)
    print(f"Saved baseline to {out_path}")
    
    # Also save full outputs as CSVs for exact comparison
    csv_dir = PROJECT_ROOT / "outputs" / "baseline" / match_id
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    df2 = load_tracking_statsperform(str(tracking_path), match_id=match_id, 
                                      normalise_dop=True, include_ball=True)
    if nrows: df2 = df2.head(nrows)
    df2["frame"] = df2["frame_count"]
    df2 = smooth_trajectory(df2, inplace=False)
    compute_velocity_features(df2, inplace=True)
    blocks2 = split_into_blocks(df2, window_minutes=5, min_frames=100)
    
    for sig_name in ["team_polarisation", "team_centroid_distance"]:
        sig_cls = SIGNAL_REGISTRY[sig_name]
        signal = sig_cls()
        output = signal.compute(match_df=df2, blocks=blocks2, game_id=match_id)
        output.to_csv(csv_dir / f"{sig_name}.csv", index=False)
        print(f"  Saved {csv_dir / f'{sig_name}.csv'} ({len(output)} rows)")

if __name__ == "__main__":
    capture()
