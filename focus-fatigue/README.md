# Focus Fatigue — Cognitive Fatigue Detection in Football Defence

Detect cognitive fatigue in football defenders using optical tracking data.
The project computes **Model 1 (Pressure Exposure)** and **5 defensive-quality
signals** on per-block, per-player basis, then merges everything into a
unified dataset for analysis.

## Table of Contents

1. [What It Does](#what-it-does)
2. [Data Requirements](#data-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [How to Run](#how-to-run)
6. [Output Structure](#output-structure)
7. [Module Tree](#module-tree)

---

## What It Does

### Model 1 — Pressure Exposure

Four load indicators computed from optical tracking data:

| Indicator | Description |
|-----------|-------------|
| **Opponent Proximity** | Number of opponents within a 7m radius of each defender |
| **Defensive Depth** | Distance from each defender to their own goal line |
| **Reorientations** | Sharp heading changes (≥45°) — a proxy for scanning |
| **Zone Transitions** | Movement between defensive zones during possession changes |

These are combined into a **Pressure Composite** that classifies each
player-block as **high pressure**, **normal**, or **low pressure**.

### Model 2 — Defensive Quality Signals

| Signal | Description |
|--------|-------------|
| **Positional Drift** | Mean distance (m) from expected role centroid during out-of-possession — measures spatial awareness degradation |
| **Shift Latency** | Mean reaction time (s) to sudden triggers (ball speed spikes, opponent runs) — measures perceptual-motor delay |
| **Pressing Accuracy** | Fraction of pressing actions classified as "correct" using Bekkers Time-To-Intercept — measures decision quality |
| **Transition Latency** | Mean reaction time (s) to possession turnovers — measures cognitive recognition delay |

### Hypothesis

As defenders experience cognitive fatigue:

- **Positional drift increases** (worse spatial awareness)
- **Reaction latencies increase** (slower cognitive processing)
- **Pressing accuracy decreases** (worse decision-making)
- **Pressure exposure changes** (altered defensive behaviour)

---

## Data Requirements

### Directory Structure

```
project-root/
├── data/
│   └── raw/
│       ├── tracking/
│       │   ├── {match_id}/
│       │   │   └── tracking.parquet    # Optical tracking data
│       │   ├── sample/                  # (optional) Sample subset
│       │   │   └── {match_id}/
│       │   │       └── tracking.parquet
│       ├── shapes/
│       │   └── {match_id}.json          # Shape model files (for positional drift)
│       ├── team_mappings/
│       │   └── team_mappings.csv        # UUID → Opta ID mapping
│       └── shape_outputs/               # Legacy V1 shape files
│           └── *.json
├── outputs/
│   ├── pressure_exposure/               # Model 1 results
│   └── signals/                         # Signal results
│       ├── positional_drift/
│       ├── shift_latency/
│       ├── pressing_accuracy/
│       └── transition_latency/
└── src/
    └── ...
```

### Data Format

**Tracking data** (`tracking.parquet`):

Stats Perform optical tracking at **25 fps** with DOP-normalised pitch
coordinates (x: -57 to +59 m, y: -39 to +39 m). Required columns:

- `current_phase` — match period (1, 2, …)
- `timeelapsed` — time in seconds
- `team_id_opta` — Stats Perform team ID
- `player_id` — player identifier (int)
- `jersey_no` — shirt number
- `pos_x`, `pos_y` — pitch coordinates
- `speed`, `speed_x`, `speed_y` — velocity
- `frame_count` — frame number (0-based, 25 fps)
- `dop` — direction of play ("L" / "R")
- `team_in_possession` — which team has the ball

**Shape files** (for positional drift):

Stats Perform shape.json files with `averageRolePositionX/Y` at ~1-minute
resolution. Supports both **V1** (legacy `liveData.shapes[]`) and **V2**
(canonical `periods[].shapes[].atTime`) formats.

**Team mappings** (`team_mappings.csv`):

Two-column CSV mapping `uuid` → `opta_id` to bridge between shape file
team UUIDs and tracking data Opta IDs.

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url> focus-fatigue
cd focus-fatigue

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e .        # installs from pyproject.toml
# OR use the lightweight requirements:
pip install -r requirements.txt

# 4. Smoke test imports
python3 -c "from src.run_signals import main; print('✅ OK')"
python3 -c "from src.run_pipeline import main; print('✅ OK')"
```

---

## Configuration

Edit `src/pressure/config.py` to set data paths, or use the CLI arguments:

```bash
# Override tracking directory
python3 src/run_signals.py --all --tracking-dir /path/to/tracking

# Override output directory (via env var or config)
```

Key configuration in `src/pressure/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `tracking_dir` | `./data/raw/tracking` | Directory containing match subdirectories |
| `sample_dir` | `./data/raw/tracking/sample` | Sample data directory |
| `output_dir` | `./outputs/pressure_exposure` | Model 1 output directory |
| `block_window_minutes` | 5 | Duration of each analysis block |
| `opponent_radius` | 7.0 | Radius (m) for opponent proximity |

Signal configuration in `src/signals/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `output_root` | `outputs/signals` | Root directory for signal outputs |

---

## How to Run

### Quick Start

```bash
# Run everything: Model 1 + all signals + merge
python3 src/run_pipeline.py --all

# Test on a single match with limited rows
python3 src/run_pipeline.py --match 2215790 --nrows 5000
```

### Individual Commands

```bash
# List available signals
python3 src/run_signals.py --list

# Run all signals on all matches
python3 src/run_signals.py --all

# Run a single signal on all matches
python3 src/run_signals.py --signal transition_latency --all

# Run signals on one match (for development)
python3 src/run_signals.py --match 2215790

# Run signals with limited rows (for testing)
python3 src/run_signals.py --match 2215790 --nrows 5000
```

```bash
# Model 1 only
python3 src/run_pressure.py --all
python3 src/run_pressure.py --match 2215790
```

```bash
# Merge outputs separately (after running Model 1 and signals)
python3 src/merge_outputs.py
```

### Order of Operations

1. **Model 1** creates `outputs/pressure_exposure/pressure_composite_{match_id}.csv`
2. **Signals** create `outputs/signals/{signal_name}/{match_id}.csv`
3. **Merge** joins everything into `outputs/unified_fatigue_dataset.parquet`

---

## Output Structure

```
outputs/
├── pressure_exposure/
│   ├── pressure_indicators_{match_id}.csv    # Per-block indicators
│   └── pressure_composite_{match_id}.csv     # Pressure composite + classification
├── signals/
│   ├── positional_drift/
│   │   └── {match_id}.csv                    # drift mean, p90, max per player-block
│   ├── shift_latency/
│   │   └── {match_id}.csv                    # mean reaction time per player-block
│   ├── pressing_accuracy/
│   │   └── {match_id}.csv                    # accuracy, n_presses per player-block
│   └── transition_latency/
│       └── {match_id}.csv                    # mean reaction time per player-block
└── unified_fatigue_dataset.parquet           # All signals + pressure merged
```

### Unified Dataset Columns

| Column | Type | Source |
|--------|------|--------|
| `game_id` | string | Match identifier |
| `block_id` | string | Block identifier (e.g. `"1_0"`) |
| `player_id` | int | Player identifier |
| `team_id_opta` | int | Team identifier |
| `opponent_proximity` | float | Mean opponents within 7m |
| `defensive_depth` | float | Distance to own goal line |
| `reorientations` | float | Reorientation count |
| `zone_transitions` | float | Zone transition count |
| `pressure_composite` | float | Weighted pressure index |
| `pressure_category` | str | `"high"`, `"normal"`, `"low"` |
| `positional_drift` | float | Mean drift distance (m) |
| `shift_latency` | float | Mean reaction time (s) to triggers |
| `pressing_accuracy` | float | Fraction of correct presses |
| `transition_latency` | float | Mean reaction time (s) to transitions |

---

## Module Tree

```
src/
├── __init__.py
│
├── run_pressure.py           # CLI entry point for Model 1
├── run_signals.py            # CLI entry point for all signals
├── run_pipeline.py           # Unified pipeline: Model 1 → signals → merge
├── merge_outputs.py          # Merge all outputs into unified parquet
│
├── pressure/                 # Model 1 — Pressure Exposure
│   ├── config.py             #   Configuration dataclass
│   ├── opponent_proximity.py #   Opponent proximity indicator
│   ├── defensive_depth.py    #   Defensive depth indicator
│   ├── reorientations.py     #   Reorientation detection
│   ├── transitions.py        #   Zone transition detection
│   ├── composite.py          #   Pressure composite + classification
│   └── gk_utils.py           #   Goalkeeper detection heuristic
│
├── signals/                  # Model 2 — Defensive Quality Signals
│   ├── base.py               #   Abstract signal base class
│   ├── config.py             #   Signal framework configuration
│   ├── output_schema.py      #   Standardised output schema + validation
│   ├── registry.py           #   Signal auto-discovery registry
│   │
│   ├── positional_drift/     #   Signal 1 — Positional Drift
│   │   ├── bridge.py         #     Player ↔ shape-role bridge
│   │   ├── drift.py          #     Drift computation + aggregation
│   │   └── config.py         #     Drift-specific configuration
│   │
│   ├── shift_latency/        #   Signal 2 — Shift Latency
│   │   ├── triggers.py       #     Ball spike + opponent run detection
│   │   ├── latency.py        #     Reaction time computation
│   │   └── config.py         #     Shift latency configuration
│   │
│   ├── pressing/             #   Signal 3 — Pressing Accuracy
│   │   ├── tti.py            #     Bekkers Time-To-Intercept
│   │   ├── detection.py      #     Pressing event detection
│   │   ├── accuracy.py       #     Classification + aggregation
│   │   └── config.py         #     Pressing configuration
│   │
│   └── transition/           #   Signal 5 — Transition Latency
│       ├── detector.py       #     Possession transition detection
│       ├── latency.py        #     Reaction time computation
│       └── config.py         #     Transition configuration
│
├── loaders/                  # Data Loading
│   ├── load_tracking.py      #   Stats Perform tracking.parquet loader
│   ├── load_shapes.py        #   Shape.json loader (V1 format)
│   ├── load_events.py        #   Event data loader
│   └── team_names.py         #   Team name ↔ UUID ↔ Opta ID bridge
│
├── smoothing.py              # Savitzky-Golay trajectory smoothing
├── segments.py               # Match segmentation into 5-minute blocks
│
├── synthetic/                # Synthetic data generator for testing
├── viz/                      # Plotting and visualisation
└── tests/                    # Unit tests (syntax checks)
```

---

## Development

```bash
# Interactive testing on a small slice
python3 src/run_pipeline.py --match 2215790 --nrows 5000

# After changes, verify imports
.venv/bin/python3 -c "from src.run_signals import main; print('✅ OK')"
.venv/bin/python3 -c "from src.run_pipeline import main; print('✅ OK')"
.venv/bin/python3 -c "from src.merge_outputs import main; print('✅ OK')"
```

## License

Proprietary — Stats Perform / Hudl data. Do not redistribute raw data.
