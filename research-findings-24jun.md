# Research Findings — 24 Jun 2026

## 1. Exact Equations Extracted

### Bekkers' Time-to-Intercept (Signal 3)

**Full formula** (eq 1-4 from the paper):

```
T_i,j(t) = τ_r + τ_i,j(t) + τ_β(t)

Where:
  τ_r = reaction time (fixed, ~0.5-1.0s)
  τ_i,j(t) = ||d|| / v_max    [distance to intercept ÷ max speed]
    d_j = r_j(t) + v_j(t)          [attacker's future position]
    d_i = r_i(t) + v_i(t) · τ_r   [defender's position after reaction]
    d = d_j - d_i                  [interception distance vector]

  τ_β(t) = ||u|| · β / π   [direction penalty]
    u = (r_i(t) + v_i(t)) - r_i(t)
    v = d_j - r_i(t)
    β = arccos(u·v / (||u|| · ||v|| + ε))
```

Then convert to probability:
```
p_i,j(T_i,j, T | σ) = [1 + exp(-π/(√3·σ) · (T - T_i,j))]⁻¹
```
where σ = 0.45 and T = 1.5 seconds (defaults).

Total pressure on an attacker:
```
P_j = 1 - ∏_i (1 - p_i,j)
```

**Active pressing threshold:** speed < 2 m/s → pressure = 0 (filters noise).

**Key insight for us:** This is already implemented in `unravelsports` as `PressingIntensity` — we may not need to code it from scratch.

---

### EFPI (Signal 1 — Expected Positions)

**Core algorithm:**
1. Take player positions for a frame or segment
2. Scale positions to match formation template dimensions (width × length)
3. For each of 65 templates, run Hungarian assignment to minimise `Σ||player - template_position||`
4. Pick the template with the lowest total cost
5. The assigned template positions become the "expected positions"

**Stability parameter:** Only change formation if `(C_prev - C_current) / C_current > ε` (default ε = 0.1)

**Works differently** for attacking vs. defending phases — you partition segments by game state first.

**Key insight:** This is already in `unravelsports` as `EFPI()` and uses kloppy under the hood. It works per-frame or per-segment.

---

### Groom's CDHMM Ghosting (Signal 4 Inspiration)

This is a covariate-dependent HMM that infers latent states (man-marking vs. zonal) for each defender on corner kicks. It requires training per team per delivery type — much heavier than what we need.

What's useful for us is the **concept**: compare actual defensive positioning to a "ghost" (average player in the same role). We can implement this without a CDHMM by:
- Using EFPI to determine expected position per defender
- Using that as our "ghost" — simpler, no ML required

---

## 2. Open-Source Tools Survey

### kloppy (already installed)
**What it does:** Standardises tracking + event data from any major provider into a common format.
**Providers supported:** 9 tracking + several event:
- Tracking: HawkEye, Metrica, PFF, SecondSpectrum, Signality, SkillCorner, Sportec, StatsPerform, Tracab
- Events: Opta, StatsBomb, Wyscout, Sportec, Metrica, SkillCorner

**What this means for Hudl data:** If Hudl's tracking is camera-based (likely SecondSpectrum or similar), kloppy can probably load it. If it's a custom format, we might need a one-time adapter.

**Key output format when loaded:**
```python
tracking_dataset = kloppy.load_tracking(…)  # returns TrackingDataset
# Columns: game_id, period_id, frame_id, timestamp
# Per player: player_id, x, y, team, position_name
# Also computes: vx, vy, velocity, acceleration (!!)
```

Kloppy already computes **velocity and acceleration** from raw positions — we might not need our own smoothing filter.

### unravelsports (not yet installed)
**What it does:** Sports analytics toolkit built on top of kloppy. Uses Polars DataFrames (faster than pandas).

**Features relevant to us:**
| Feature | What it does | Our use |
|---------|-------------|---------|
| `KloppyPolarsDataset` | Converts kloppy data to Polars | Data pipeline |
| `PressingIntensity()` | Loads Bekkers' TTI per frame | **Replaces Signal 3** implementation entirely |
| `EFPI()` | Formation + position assignment | **Replaces Signal 1** implementation entirely |
| `SoccerGraphConverter` | Converts games to graphs for GNN | Not needed for now |

**This is huge — two of our five signals already exist as production code.**

### OpenSTARLab (for reference)
- Focused on deep learning + RL (event prediction, reinforcement learning)
- Their preprocessing package (`UIED` format) might help with data standardisation
- Their SL (Supervised Learning) event prediction models are overkill for our needs
- Not worth installing unless we need their data formats

---

## 3. Key Takeaways

| Signal | Our approach | Available off-the-shelf? |
|--------|-------------|--------------------------|
| **Signal 5** — Transition recognition | Our own (novel) | Never been done |
| **Signal 3** — Pressing accuracy | Bekkers TTI | **Yes** — `unravelsports.PressingIntensity` |
| **Signal 1** — Positional drift | EFPI + deviation | **Yes** — `unravelsports.EFPI` |
| **Signal 2** — Shift latency | Our own (novel) | Never been done |
| **Signal 4** — Spatial awareness | Simplified ghosting | Partial (pitch control formula) |

**What I'd recommend when data arrives:**
1. Try loading Hudl data with **kloppy** — if it works, we skip the data format problem
2. Install **unravelsports** — saves us rewriting Signals 1 and 3
3. Build Signals 2 and 5 from scratch (novel contributions)
4. Pitch control formula can be coded in ~50 lines using Spearman's method

---

Now have a think about that. I'll be here when you're ready at 9pm 👍
