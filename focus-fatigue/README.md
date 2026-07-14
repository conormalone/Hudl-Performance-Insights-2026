# Focus Fatigue — Cognitive Fatigue Detection in Football Defence

Detect cognitive fatigue in football defenders using optical tracking data
(Stats Perform, 25fps). Computes **Model 1 (Pressure Exposure)** and
**4 defensive-quality signals** on per-block, per-player basis, then merges
everything into a unified dataset.

## Quick Start

```bash
pip install -r requirements.txt
python3 src/pipeline.py --sample  # runs on 3 sample matches
python3 src/pipeline.py --match 2215790 --nrows 5000  # test on one match
python3 src/pipeline.py --list-signals
```

Or use the notebook: `notebooks/pipeline.ipynb` — sets `DATA_ROOT`, calls the same pipeline functions.

## File-by-File Purpose

### `/src` — Library code

| File | Purpose |
|------|---------|
| `loaders.py` | Load tracking.parquet files and team_mappings.csv |
| `smoothing.py` | Savitzky-Golay trajectory smoothing + velocity/heading features |
| `segments.py` | Split match into 5-minute time blocks |
| `pipeline.py` | **Single entry point**: Model 1 → Signals → Merge |
| `merge_outputs.py` | Merge all CSVs into unified `unified_fatigue_dataset.parquet` |
| `pressure/config.py` | All Model 1 tuning parameters (dataclass) |
| `pressure/gk_utils.py` | Goalkeeper detection heuristic (shared) |
| `pressure/opponent_proximity.py` | Indicator 1: count opponents within 7m |
| `pressure/defensive_depth.py` | Indicator 2: distance from own goal line |
| `pressure/reorientations.py` | Indicator 3: sharp heading changes ≥45° |
| `pressure/transitions.py` | Indicator 4: possession flips in defender's zone |
| `pressure/composite.py` | Combine 4 indicators → pressure score → classification |
| `signals/__init__.py` | Re-exports for the signal framework |
| `signals/base.py` | `SignalBase` — abstract class all signals inherit from |
| `signals/config.py` | Signal framework config (output paths) |
| `signals/registry.py` | `@register_signal` decorator + lookup by name |
| `signals/output_schema.py` | Standardised output columns + validation |
| `signals/drift.py` | **Signal 1**: positional drift from shape-model centroids |
| `signals/shift.py` | **Signal 2**: reaction latency to ball spikes + opponent runs |
| `signals/pressing.py` | **Signal 3**: pressing accuracy via Bekkers Time-To-Intercept |
| `signals/transition.py` | **Signal 5**: reaction latency to possession turnovers |

### `/notebooks`

| File | Purpose |
|------|---------|
| `pipeline.ipynb` | Thin runner — sets paths, calls `src.pipeline` functions |

### `/outputs`

| Path | Contents |
|------|----------|
| `pressure_exposure/pressure_composite_{match_id}.csv` | Model 1 per-block results |
| `pressure_exposure/pressure_indicators_{match_id}.csv` | Raw indicator values |
| `signals/{signal_name}/{match_id}.csv` | Per-block signal values |
| `unified_fatigue_dataset.parquet` | All signals + pressure merged |

## Data Flow

```
tracking.parquet
    ├─→ Model 1 (pressure/) — 4 indicators → pressure composite + classification
    └─→ Signals (signals/) — positional_drift / shift_latency / pressing_accuracy / transition_latency
                ↓
         merge_outputs.py  →  unified_fatigue_dataset.parquet
```

## What Each Signal Measures

| Signal | Input | Hypothesis |
|--------|-------|-----------|
| **Positional Drift** | Shape-model centroids | Fatigued defenders drift from expected positions |
| **Shift Latency** | Ball speed spikes + opponent runs | Fatigued defenders react slower to sudden play shifts |
| **Pressing Accuracy** | Bekkers Time-To-Intercept | Fatigued defenders make worse pressing decisions |
| **Transition Latency** | Possession flip events | Fatigued defenders are slower to recognise turnovers |

## Adding a New Signal

1. Create a new file in `src/signals/` (e.g., `my_signal.py`)
2. Define a class inheriting from `SignalBase` with `@register_signal`
3. Import the module in `src/pipeline.py` (or `notebooks/pipeline.ipynb`)
4. Done — it's auto-discovered via `list_signals()`

## Known Issues

- **Signal 4 gap**: Signals are numbered 1, 2, 3, 5. Signal 4 not yet implemented.
- **Orphaned code removed**: Previous cleanup deleted `baselines.py`, `run_pressure.py`,
  `run_signals.py`, empty placeholders (`synthetic/`, `tests/`, `viz/`), unused loaders
  (`load_events.py`, `load_shapes.py`, `team_names.py`).

## License

Proprietary — Stats Perform / Hudl data. Do not redistribute raw data.
