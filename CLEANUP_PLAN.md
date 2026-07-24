# Repo Cleanup Plan — Hudl-Performance-Insights-2026

## Goals
1. Collapse duplication — everything lives under `focus-fatigue/` or is consolidated in `docs/`
2. Remove tracked binary artifacts (parquet), pycache, per-match CSVs
3. Organise project documentation into `docs/`
4. Update .gitignore to prevent re-introduction
5. Update README to reflect new structure

## Actions

### 1. Move tracked project docs → `docs/` (git mv)
- `data-audit-report.md` → `docs/data-audit-report.md`
- `literature-review.md` → `docs/literature-review.md`
- `methodology-defensive-quality.md` → `docs/methodology-defensive-quality.md`
- `research-findings-24jun.md` → `docs/research-findings-24jun.md`
- `task-plan.md` → `docs/task-plan.md`
- `work-plan.md` → `docs/work-plan.md`
- `work-plan-critique.md` → `docs/work-plan-critique.md`
- `workflow.md` → `docs/workflow.md`

### 2. Remove tracked files (git rm)
- `unified_fatigue_dataset.parquet` — duplicate of focus-fatigue/outputs/analysis/unified_fatigue_dataset.parquet

### 3. Delete untracked development artifacts
- `pm-role-spec.md`
- `reviewer-role-spec.md`
- `storyteller-role-spec.md`
- `validator-role-spec.md`

### 4. Delete empty/vestigial root directories (only pycache or .gitkeep)
- `analysis/` — only `__pycache__/`
- `fixtures/` — only `__pycache__/`
- `src/` — only `__pycache__/` (src/model1/, src/model2/)
- `outputs/signals/` — only `integration_audit.md`
- `data/` — only `.gitkeep`

### 5. Move integration_audit.md to focus-fatigue/outputs/analysis/
- `outputs/signals/integration_audit.md` → `focus-fatigue/outputs/analysis/integration_audit.md`

### 6. Remove all `__pycache__/` directories
- `analysis/__pycache__/`
- `fixtures/__pycache__/`
- `src/model1/__pycache__/`
- `src/model2/__pycache__/`
- `focus-fatigue/src/__pycache__/`
- `focus-fatigue/src/pressure/__pycache__/`
- `focus-fatigue/src/signals/__pycache__/`

### 7. Per-match CSV signal files (already gitignored, just delete)
- 373 CSV files across focus-fatigue/outputs/signals/*/ (press_accuracy, positional_drift, etc.)

### 8. Update `.gitignore`
- Add `*.parquet` (data is regenerable)
- Add `data/` root dir
- Keep existing patterns

### 9. Update `README.md`
- Reflect new directory structure
