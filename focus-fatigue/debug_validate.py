#!/usr/bin/env python3
"""Debug validation — checks imports and runs basic tests."""

import sys
sys.path.insert(0, '/mnt/usb/project/focus-fatigue')

print("=== Step 1: Basic imports ===", flush=True)

import numpy as np
import pandas as pd
print("numpy, pandas: OK", flush=True)

from src.loaders.load_tracking import load_tracking_statsperform
print("load_tracking_statsperform: OK", flush=True)

from src.smoothing import smooth_trajectory, compute_velocity_features
print("smoothing: OK", flush=True)

from src.segments import split_into_blocks
print("split_into_blocks: OK", flush=True)

print("\n=== Step 2: Model 1 imports ===", flush=True)
from src.pressure.composite import build_pressure_dataset, compute_block_baselines
print("composite: OK", flush=True)
from src.pressure.config import DEFAULT_CONFIG
print("config: OK", flush=True)
from src.pressure.gk_utils import flag_goalkeepers
print("gk_utils: OK", flush=True)
from src.pressure.opponent_proximity import calculate_opponent_proximity_corrected
from src.pressure.defensive_depth import compute_defensive_depth
from src.pressure.reorientations import detect_reorientations
from src.pressure.transitions import detect_transition_frames
print("pressure modules: OK", flush=True)

print("\n=== Step 3: Signal imports ===", flush=True)
from src.signals.shift_latency import ShiftLatencySignal
print("shift_latency: OK", flush=True)
from src.signals.transition import TransitionRecognitionSignal
print("transition: OK", flush=True)
from src.signals.pressing import PressingAccuracySignal
print("pressing: OK", flush=True)
from src.signals.positional_drift import PositionalDriftSignal
print("positional_drift: OK", flush=True)

from src.signals.registry import list_signals
print(f"Registered: {list_signals()}", flush=True)

from src.signals.output_schema import validate_output
print("output_schema: OK", flush=True)

print("\n=== Step 4: Data loading ===", flush=True)
MATCH_ID = "2215790"
SAMPLE_DIR = "/mnt/usb/conor_downloads/team_mappings/sample"
TRACKING_PATH = f"{SAMPLE_DIR}/{MATCH_ID}/tracking.parquet"

df = load_tracking_statsperform(TRACKING_PATH, match_id=MATCH_ID, normalise_dop=True, include_ball=True)
n_frames = df["frame_count"].nunique()
print(f"Loaded: {len(df):,} rows, {n_frames:,} frames", flush=True)

# Slice
frames = sorted(df["frame_count"].unique())
df = df[df["frame_count"].between(frames[0], frames[0] + 5000, inclusive="left")].copy()
print(f"Sliced: {len(df):,} rows, {df['frame_count'].nunique():,} frames", flush=True)

# Smooth
df = smooth_trajectory(df, inplace=False)
df = compute_velocity_features(df)
print("Smoothing: OK", flush=True)

# Blocks
blocks = split_into_blocks(df)
print(f"Blocks: {len(blocks)}", flush=True)

print("\n=== Step 5: Signal 5 test ===", flush=True)
s5 = TransitionRecognitionSignal()
out5 = s5.compute(df, blocks, game_id=MATCH_ID)
print(f"Signal 5 output: {len(out5)} rows", flush=True)
if len(out5) > 0:
    print(f"  Values: min={out5['signal_value'].min():.3f}, mean={out5['signal_value'].mean():.3f}, max={out5['signal_value'].max():.3f}", flush=True)
    validate_output(out5, signal_name="transition_latency")
    print("  Schema: OK", flush=True)

print("\n=== Step 6: Signal 3 test ===", flush=True)
from src.signals.pressing.config import DEFAULT_PRESSING_CONFIG
print(f"Threshold: {DEFAULT_PRESSING_CONFIG.correct_press_threshold}", flush=True)

teams = df[df["player_id"] != -1]["team_id_opta"].unique()
team_a, team_b = int(teams[0]), int(teams[1])
print(f"Teams: {team_a}, {team_b}", flush=True)

s3 = PressingAccuracySignal()
out3 = s3.compute(df, blocks, game_id=MATCH_ID, own_team_id=team_a, opponent_team_id=team_b)
print(f"Signal 3 (team {team_a}) output: {len(out3)} rows", flush=True)
if len(out3) > 0:
    print(f"  Mean accuracy: {out3['signal_value'].mean():.3f}", flush=True)
    if "n_presses" in out3.columns:
        print(f"  Total presses: {out3['n_presses'].sum()}", flush=True)
    validate_output(out3, signal_name="pressing_accuracy")
    print("  Schema: OK", flush=True)

print("\n=== Step 7: Signal 1 test ===", flush=True)
SHAPE_PATH = f"{SAMPLE_DIR}/{MATCH_ID}.json"
s1 = PositionalDriftSignal()
out1 = s1.compute(df, blocks, game_id=MATCH_ID, shape_path=SHAPE_PATH)
print(f"Signal 1 output: {len(out1)} rows", flush=True)
if len(out1) > 0:
    print(f"  Drift values: min={out1['signal_value'].min():.2f}, mean={out1['signal_value'].mean():.2f}, max={out1['signal_value'].max():.2f}", flush=True)
    validate_output(out1, signal_name="positional_drift")
    print("  Schema: OK", flush=True)
else:
    print("  (empty — checking bridge debug)", flush=True)
    from src.signals.positional_drift.bridge import load_shape_file, build_player_role_map
    shapes = load_shape_file(SHAPE_PATH)
    for mfs in [0.5, 0.3, 0.1, 0.0]:
        prm = build_player_role_map(df, shapes, min_fit_score=mfs)
        print(f"  min_fit={mfs}: {len(prm)} players, {sum(len(v) for v in prm.values())} entries", flush=True)

print("\n=== Step 8: Signal 2 test ===", flush=True)
s2 = ShiftLatencySignal()
out2 = s2.compute(df, blocks, game_id=MATCH_ID)
print(f"Signal 2 output: {len(out2)} rows", flush=True)
if len(out2) > 0:
    print(f"  Values: min={out2['signal_value'].min():.3f}, mean={out2['signal_value'].mean():.3f}, max={out2['signal_value'].max():.3f}", flush=True)
    validate_output(out2, signal_name="shift_latency")
    print("  Schema: OK", flush=True)

print("\n=== ALL DONE ===", flush=True)
