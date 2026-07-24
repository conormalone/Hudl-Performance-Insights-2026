# Focus Fatigue — Soccer Analytics

Research project investigating cognitive and physical fatigue in elite soccer, using tracking data to quantify pressure events, defensive load, and performance decline.

---

## Repository Structure

```
docs/                          — Project documentation and reports
  data-audit-report.md         — Data quality audit
  literature-review.md         — Literature survey
  methodology-defensive-quality.md
  research-findings-24jun.md
  task-plan.md / work-plan.md  — Planning docs
  workflow.md                  — Development workflow

focus-fatigue/                 — Primary analysis package
  analysis/                    — Analysis scripts
  data/                        — Raw and processed datasets
  figures/                     — Generated figures
  notebooks/                   — Jupyter notebooks
  outputs/                     — All generated outputs
    analysis/                  — Markdown reports, figures, summaries
    signals/                   — Per-match signal CSVs (gitignored)
  review/                      — Peer review and validation docs
  scripts/                     — Pipeline scripts
  specs/                       — Spec docs
  src/                         — Python package source
    pressure/                  — Pressure event modules
    signals/                   — Signal computation modules

assets/                        — Reference papers (PDFs) and download reports
review/                        — Peer review documents
scripts/                       — Root-level utility scripts
literature-review/             — Literature review files (bibliography)
```

## Workflow

1. Specs and methodology documented in `focus-fatigue/specs/`
2. Analysis scripts in `focus-fatigue/analysis/` implement the models
3. Results output to `focus-fatigue/outputs/analysis/`
4. Peer review and validation tracked in `focus-fatigue/review/`

---

*Powered by Jervis 🧠*
