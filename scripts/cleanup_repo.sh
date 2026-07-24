#!/usr/bin/env bash
set -euo pipefail

cd /home/conormalone/.openclaw/workspace/project

echo "=== Step 1: Create docs/ directory ==="
mkdir -p docs

echo "=== Step 2: git mv tracked project docs to docs/ ==="
git mv data-audit-report.md docs/
git mv literature-review.md docs/ 2>/dev/null || echo "  literature-review.md not tracked, copying"
if [ -f literature-review.md ] && [ ! -f docs/literature-review.md ]; then
  cp literature-review.md docs/ && rm literature-review.md
fi
git mv methodology-defensive-quality.md docs/
git mv research-findings-24jun.md docs/
git mv task-plan.md docs/
git mv work-plan.md docs/
git mv work-plan-critique.md docs/
git mv workflow.md docs/

echo "=== Step 3: git rm duplicate parquet ==="
git rm unified_fatigue_dataset.parquet

echo "=== Step 4: Delete untracked role spec files ==="
rm -f pm-role-spec.md reviewer-role-spec.md storyteller-role-spec.md validator-role-spec.md

echo "=== Step 5: Move integration_audit.md ==="
mkdir -p focus-fatigue/outputs/analysis
if [ -f outputs/signals/integration_audit.md ]; then
  mv outputs/signals/integration_audit.md focus-fatigue/outputs/analysis/
fi

echo "=== Step 6: Remove empty vestigial root directories ==="
rm -rf analysis/ fixtures/ src/ outputs/signals/ outputs/
rm -f data/.gitkeep
rmdir data/ 2>/dev/null || true

echo "=== Step 7: Remove all __pycache__ directories ==="
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

echo "=== Step 8: Remove per-match CSV signal outputs ==="
find focus-fatigue/outputs/signals/ -name '*.csv' -delete 2>/dev/null || true
# Remove empty signal dirs but keep structure via .gitkeep in parent
find focus-fatigue/outputs/signals/ -type d -empty -delete 2>/dev/null || true

# Also clean up regenerable CSV outputs in other dirs
rm -f focus-fatigue/outputs/scoreline_summary.csv
rm -f focus-fatigue/outputs/pressure_exposure/pressure_*.csv
rm -f focus-fatigue/outputs/baseline/2215790/*.csv
rm -f focus-fatigue/outputs/baseline/*.csv
rm -f focus-fatigue/outputs/profile_pipeline/*.csv

echo "=== Step 9: Remove duplicate parquet from focus-fatigue/ root ==="
rm -f focus-fatigue/unified_fatigue_dataset.parquet
rm -f focus-fatigue/outputs/unified_fatigue_dataset.parquet

echo "=== Step 10: Write updated .gitignore ==="
cat > .gitignore << 'GITIGNORE'
# Outputs
outputs/
code/

# Python
__pycache__/
*.py[cod]
*.egg-info/
*.so
.venv/
venv/

# Data — regenerable
*.parquet
*.csv

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
*.orig
GITIGNORE

echo "=== Step 11: Update README.md ==="
cat > README.md << 'README'
# Hudl Performance Insights 2026

Soccer analytics research project exploring cognitive fatigue and performance in football defence.

## Structure

```
docs/                               — Project documentation (reports, plans, methodology)
├── data-audit-report.md
├── literature-review.md
├── methodology-defensive-quality.md
├── research-findings-24jun.md
├── task-plan.md
├── work-plan.md
├── work-plan-critique.md
└── workflow.md

focus-fatigue/                      — Core analysis project
├── notebooks/                      — Jupyter notebooks
├── scripts/                        — Analysis scripts
├── specs/                          — Methodology specs
├── src/                            — Python source
├── analysis/                       — Analysis code
├── fixtures/                       — Test fixtures
├── outputs/                        — Generated outputs
│   ├── analysis/                   — Reports, figures, summaries
│   ├── baseline/                   — Baseline measurements
│   ├── pressure_exposure/          — Pressure exposure outputs
│   └── profile_pipeline/           — Profile pipeline outputs
├── data/                           — Data files
├── README.md
├── pyproject.toml
└── requirements.txt

literature-review/                  — Literature review with references
├── fatigue-lit-review.md
└── references.bib

review/                             — Peer review documents
├── peer-review.md

assets/                             — Reference PDFs and downloads
├── download-report.md
└── *.pdf
```

## Cleanup

Repository was cleaned on July 2026:
- Project docs consolidated under `docs/`
- Duplicate data files removed
- Generated/output files gitignored
- `__pycache__` directories removed
README

echo "=== Done ==="
echo "Showing staged changes:"
git status
