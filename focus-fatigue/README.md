# Focus Fatigue — Cognitive Fatigue in Football Defence

## Project structure

```
focus-fatigue/
├── src/
│   ├── loaders/         # Data loading & canonicalisation
│   ├── pressure/        # Model 1: Pressure Exposure
│   ├── signals/         # Model 2: Defensive Quality Signals
│   │   ├── pressing/    #   Signal 3: Bekkers TTI methods
│   │   └── positional_drift/  # Signal 1: EFPI methods
│   ├── synthetic/       # Synthetic data generator
│   ├── viz/             # Plotting & visualisation
│   └── tests/           # Tests (syntax checks)
├── data/
│   ├── raw/             # Original Hudl data (gitignored)
│   └── processed/       # Cleaned/merged data (gitignored)
├── outputs/
│   ├── pressure_exposure/
│   └── signals/
├── notebooks/           # Exploration notebooks
├── pyproject.toml       # Project metadata & dependencies
└── README.md
```
