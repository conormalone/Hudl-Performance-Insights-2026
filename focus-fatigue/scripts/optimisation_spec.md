# Optimisation Specification

## Profile Results (match 2215790, 3 blocks / 500K rows)

| Signal | Time | Bottleneck |
|---|---|---|
| team_centroid_distance | 73.2s | Per-frame Python loop |
| team_polarisation | 52.0s | Per-frame Python loop |
| pressing_accuracy | 18.9s | Not our target |
| shift_latency | 1.8s | OK |
| physical_load | 0.8s | OK |
| transition_latency | 0.4s | OK |

## Strategy
Replace per-frame Python for-loops with vectorised pandas groupby operations.

---

## 1. Optimise `compute_polarisation_block` in `src/signals/polarisation.py`

### Current (slow):
```python
for frame_val in sorted(block_df["frame_count"].unique()):
    fdf = block_df[block_df["frame_count"] == frame_val]
    pol = compute_polarisation_frame(fdf, ...)
    for team_id, r_val in pol.items():
        if not np.isnan(r_val):
            frame_pol.setdefault(team_id, []).append(r_val)
    n_frames_total += 1
```

### Optimised (vectorised):
```python
# Step 1: Get ball possession per frame
ball_df = block_df[block_df["player_id"] == _BALL_PLAYER_ID]
in_poss_per_frame = ball_df.set_index("frame_count")[team_in_possession_col].to_dict()

# Step 2: Filter to outfield, out-of-possession teams
outfield = block_df[block_df["player_id"] != _BALL_PLAYER_ID].copy()
outfield["_in_poss"] = outfield["frame_count"].map(in_poss_per_frame)
oop_mask = outfield["_in_poss"].isna() | (outfield[team_id_col] != outfield["_in_poss"])
oop = outfield[oop_mask].copy()

if len(oop) == 0:
    return []

# Step 3: Compute unit vectors (filter stationary)
vx = oop[vx_col].values.astype(np.float64)
vy = oop[vy_col].values.astype(np.float64)
speed = np.sqrt(vx**2 + vy**2)
moving_mask = speed >= min_velocity

oop_moving = oop[moving_mask].copy()
if len(oop_moving) < 2:
    return []

sm = speed[moving_mask]
oop_moving["_ux"] = oop_moving[vx_col].values / sm
oop_moving["_uy"] = oop_moving[vy_col].values / sm

# Step 4: Per (frame, team) compute mean resultant length R
grouped = oop_moving.groupby(["frame_count", team_id_col])
sum_vx_g = grouped["_ux"].sum()
sum_vy_g = grouped["_uy"].sum()
counts = grouped["_ux"].count()

# R = sqrt(sum_x² + sum_y²) / n
r_values = np.sqrt(sum_vx_g**2 + sum_vy_g**2) / counts.values

# Step 5: Filter to groups with >= 2 moving players
valid = counts >= 2
r_series = r_values[valid]

# Step 6: Aggregate per team (mean R across frames for each team)
team_r_mean = r_series.groupby(level=team_id_col, sort=False).mean()

# Build records
records = []
for team_id, mean_r in team_r_mean.items():
    records.append({
        "block_id": block["block_id"],
        "phase": block["phase"],
        "player_id": 0,
        "team_id_opta": int(team_id),
        "signal_name": _SIGNAL_NAME,
        "signal_value": round(float(mean_r), 6),
        "n_frames": int(counts[counts.index.get_level_values(team_id_col) == team_id].sum()),
    })
```

### Key points:
- No Python for-loops over frames
- No repeated df filtering per frame
- Uses pandas groupby for all aggregation
- `compute_polarisation_frame` kept for backward compatibility but `compute_polarisation_block` uses vectorised logic internally

---

## 2. Optimise `compute_centroid_distance_block` in `src/signals/team_centroid_distance.py`

### Current (slow):
```python
for frame_val in sorted(block_df["frame_count"].unique()):
    fdf = block_df[block_df["frame_count"] == frame_val]
    team_map = compute_centroid_distance_frame(fdf, ...)
    for team_id, player_dists in team_map.items():
        for pid, dist in player_dists.items():
            player_distances.setdefault((pid, int(team_id)), []).append(dist)
```

### Optimised (vectorised):
```python
# Step 1: Get ball possession per frame
ball_df = block_df[block_df["player_id"] == _BALL_PLAYER_ID]
if len(ball_df) > 0:
    in_poss_per_frame = ball_df.set_index("frame_count")[team_in_possession_col].to_dict()
else:
    in_poss_per_frame = {}

# Step 2: Filter to outfield, OOP
outfield = block_df[block_df["player_id"] != _BALL_PLAYER_ID].copy()
if len(outfield) == 0:
    return []
outfield["_in_poss"] = outfield["frame_count"].map(in_poss_per_frame)
oop_mask = outfield["_in_poss"].isna() | (outfield[team_id_col] != outfield["_in_poss"])
oop = outfield[oop_mask].copy()

if len(oop) == 0:
    return []

# Step 3: Compute team centroids per frame (vectorised transform)
# Mean x, mean y within each (frame, team) group
centroids = oop.groupby(["frame_count", team_id_col])[["x", "y"]].transform("mean")
oop["_cx"] = centroids["x"].values
oop["_cy"] = centroids["y"].values

# Step 4: Euclidean distance to centroid
oop["_dist"] = np.sqrt(
    (oop["x"].values - oop["_cx"])**2 + 
    (oop["y"].values - oop["_cy"])**2
)

# Step 5: Aggregate per player per team (mean distance across frames)
agg = oop.groupby(["player_id", team_id_col])["_dist"].agg(mean="mean", count="count")

# Step 6: Build records
n_frames_in_block = oop["frame_count"].nunique()
records = []
for (pid, team_id), row in agg.iterrows():
    records.append({
        "block_id": block["block_id"],
        "phase": block["phase"],
        "player_id": int(pid),
        "team_id_opta": int(team_id),
        "signal_name": _SIGNAL_NAME,
        "signal_value": round(float(row["mean"]), 6),
        "n_frames": int(row["count"]),
    })
```

### Key points:
- Single `groupby.transform("mean")` replaces per-frame loop
- Single `groupby().agg()` replaces per-player accumulation
- No Python loops over frames or players

---

## Implementation Files

1. `src/signals/polarisation.py` — Replace `compute_polarisation_block` body (keep `compute_polarisation_frame` as-is for tests/backward compat)
2. `src/signals/team_centroid_distance.py` — Replace `compute_centroid_distance_block` body (keep `compute_centroid_distance_frame` as-is)

## Verification

After implementation, run:
```bash
cd /home/conormalone/.openclaw/workspace/project/focus-fatigue
python3 scripts/profile_signals.py --match 2215790 --nrows 500000
```

Compare:
- Old output rows must match new output rows (for identical input)
- Signal values must match within floating-point tolerance (1e-6)
- Speed should improve dramatically (10-50x expected)
