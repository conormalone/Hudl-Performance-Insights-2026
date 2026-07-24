# Data Audit Report — Stats Perform Dataset

**Date:** 10 Jul 2026  
**Source:** Stats Perform optical tracking, Ligue 1 2021-22  
**Total:** 100 matches | 16 GB

---

## Schema Check

| Check | Result |
|-------|--------|
| Column consistency | ✅ All 100 matches share identical 17 columns |
| shape.json format | ✅ All 100 files parse cleanly (0 errors) |
| Dtype variations | ⚠️ Minor: some matches store `player_id` as int64, others as float64 (due to NaN presence at one dropout frame). Easily normalised. |

**Verdict:** ✅ Consistent across all 100 matches — one pipeline fits all.

---

## Team Coverage

| Team | Matches |
|------|---------|
| Metz | 11 |
| Lens | 11 |
| PSG | 11 |
| Bordeaux | 11 |
| Strasbourg | 11 |
| Monaco | 11 |
| Brest | 10 |
| Lorient | 10 |
| Nice | 10 |
| Nantes | 10 |
| Lille | 10 |
| Olympique Lyonnais | 10 |
| Rennes | 10 |
| Saint-Étienne | 10 |
| Troyes | 10 |
| Angers SCO | 10 |
| Montpellier | 10 |
| Clermont Foot | 10 |
| Marseille | 10 |
| Stade de Reims | 10 |

**Total:** 20 unique teams | All Ligue 1 2021-22 | 10-11 appearances each

---

## Player ID Consistency

| Metric | Value |
|--------|-------|
| Total unique player IDs | 459 |
| IDs appearing in 2+ matches | 426 (92.8%) |
| IDs appearing in only 1 match | 33 (7.2% — fringe/subs) |
| Players changing teams | **0** — IDs are stable and team-specific |

**Verdict:** ✅ Player IDs are persistent across the season. No cross-team ID reuse.

---

## Data Quality

| Issue | Count | Severity |
|-------|-------|----------|
| NaN `player_id` rows | ~46 per match (single frame dropout) | 🟢 Negligible |
| NaN `team_in_possession` | ~42% of rows (ball not possessed) | 🟢 Expected |
| Missing tracking frames | None detected | 🟢 Full coverage |

---

## Team ID Bridge

Three ID systems in play:

1. **`team_id_opta`** (tracking.parquet) — e.g., 142, 145
2. **Stats Perform UUID** (shape.json) — e.g., `2khen2a38l2hkx33s73pehl6o`
3. **Team name** (shape.json `matchInfo.contestant[].name`)

A `team_mappings.csv` maps UUID → Opta ID. The shape.json provides team names per match. These can be joined to translate between systems.

---

## Sample Data

Three matches staged in `/home/conormalone/conor_downloads/team_mappings/sample/`:
- 2215790, 2215791, 2215792 — each with `tracking.parquet` + flat `shape.json`

---

## Recommendations

1. **Normalise dtypes on load** — cast `player_id` to int, fill NaNs with -1 (ball marker)
2. **Build team name lookup** from shape.json per match (not just the CSV — shape.json has the definitive per-match team name)
3. **Proceed with 100-match bulk processing** — no schema surprises expected
