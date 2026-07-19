# Scoreline State Integration Plan

## Overview

Add scoreline context to every block in the unified fatigue dataset (45,634 player-blocks across 100 Ligue 1 2021-22 matches). This enables testing whether cognitive fatigue varies with match scoreline.

---

## a) Match Mapping Approach

### Goal
Link each tracking `game_id` (e.g. `2215771`) to its row in the fixtures CSV.

### Data Sources
1. **Tracking match metadata** — `/mnt/usb/conor_downloads/team_mappings/shape_outputs/{game_id}.json`
   - `matchInfo.contestant[]` array includes `name` and `position: "home"|"away"` for both teams
   - `matchInfo.localDate` provides the actual match date
2. **Fixtures CSV** — `/mnt/usb/project/focus-fatigue/data/ligue1_2021_22_complete_fixtures.csv`
   - Columns: Date, HomeTeam, AwayTeam, HomeGoals, AwayGoals, HomeGoalMinutes, AwayGoalMinutes
   - Each row is a unique `(HomeTeam, AwayTeam)` pair (380 rows = one full Ligue 1 season)

### Team Name Normalization
The shape JSON uses full/formal names; the CSV uses shorter names. A lookup dict handles the 4 mismatches:

| Shape JSON name       | CSV name       |
|-----------------------|----------------|
| Angers SCO            | Angers         |
| Clermont              | Clermont Foot  |
| Olympique Lyonnais    | Lyon           |
| Olympique Marseille   | Marseille      |
| (all others unchanged)| (same)         |

The remaining 16 names match directly (Bordeaux, Brest, Lens, Lille, Lorient, Metz, Monaco, Montpellier, Nantes, Nice, PSG, Reims, Rennes, Saint-Étienne, Strasbourg, Troyes).

### Matching Algorithm (per game_id)
```
1. Load shape JSON for {game_id}.json
2. Read home_name = contestant["home"], away_name = contestant["away"]
3. Apply team name normalization → csv_home, csv_away
4. Find row in fixtures where HomeTeam == csv_home AND AwayTeam == csv_away
5. Assert: exactly 1 row matched (validated: 100/100 tracking matches map uniquely)
```

### Validation
All 100 tracking matches were verified to map to exactly one fixture row. Zero failures.

---

## b) Block-to-Minute Mapping

Each block represents a 5-minute window of match time. The tracking data has per-block frame counts of 7,500 frames (at 25 fps tracking rate = 300 seconds = 5 minutes).

### Minute Range Per Block

| `block_num` | Minutes range | Match timing          |
|-------------|---------------|-----------------------|
| 0           | [0, 5)        | Kick-off → 5'         |
| 1           | [5, 10)       |                       |
| 2           | [10, 15)      |                       |
| 3           | [15, 20)      |                       |
| 4           | [20, 25)      |                       |
| 5           | [25, 30)      |                       |
| 6           | [30, 35)      |                       |
| 7           | [35, 40)      |                       |
| 8           | [40, 45)      | End of first half     |
| 9           | [45, 50)      | Start of second half  |
| 10          | [50, 55)      |                       |
| 11          | [55, 60)      |                       |
| 12          | [60, 65)      |                       |
| 13          | [65, 70)      |                       |
| 14          | [70, 75)      |                       |
| 15          | [75, 80)      |                       |
| 16          | [80, 85)      |                       |

### Block Availability

| Blocks present | Number of games | Coverage    |
|----------------|-----------------|-------------|
| 0–9            | 67              | 50 minutes  |
| 0–10           | 31              | 55 minutes  |
| 0–12           | 1               | 65 minutes  |
| 0–16           | 1               | 85 minutes  |

### Key Rule
For each block, the **scoreline state is determined at the start of the block** (minute `block_num * 5`). This means the state reflects conditions *before* any goals that may occur during that block's 5-minute window — avoiding look-ahead bias.

---

## c) Scoreline State Per Block

### Cumulative Goals Calculation

For block starting at minute `M`:
```
home_goals_at_M = count of HomeGoalMinutes where minute < M
away_goals_at_M = count of AwayGoalMinutes where minute < M
goal_diff = home_goals_at_M - away_goals_at_M
```

### Five Scoreline States

| Condition                          | State                  |
|------------------------------------|------------------------|
| goal_diff == 0                     | draw                   |
| goal_diff == 1                     | winning_by_1           |
| goal_diff == -1                    | losing_by_1            |
| goal_diff >= 2                     | winning_by_more_than_1 |
| goal_diff <= -2                    | losing_by_more_than_1  |

### Boundary Rules
- Goal minute is compared with **strict less-than** (`<`), not `<=`
- A goal scored at minute 30 is counted for block 6 (starting at min 30) *during* block 6, not at its start
- Goal minutes are stored as string-encoded lists in the CSV (e.g. `"['4', '15', '82']"`). Parse as integers.

---

## d) New Columns

The following columns will be added to each row in the unified dataset:

| Column name      | Type    | Description                                          |
|------------------|---------|------------------------------------------------------|
| `home_goals`     | int     | Cumulative home goals scored before this block starts |
| `away_goals`     | int     | Cumulative away goals scored before this block starts |
| `goal_diff`      | int     | `home_goals - away_goals` at block start             |
| `scoreline_state`| str     | One of the 5 categories above                        |
| `home_goals_in_block`   | int | Home goals scored *during* this block's 5-min window |
| `away_goals_in_block`   | int | Away goals scored *during* this block's 5-min window |
| `conceded_in_block`     | bool | True if opponent scored during this block           |
| `scored_in_block`       | bool | True if this team scored during this block          |

The last three allow testing immediate conceding/scoring effects *on the next block*.

---

## e) Example: Marseille 0–5 Lyon (game_id: 2215840)

**Match data:** Home: Marseille, Away: Lyon  
**Goal minutes:** Away = [13, 30, 36, 49, 59]  
**Blocks available:** 0–9 (50 minutes of tracking data)

| Block | Start Min | Home Goals | Away Goals | Goal Diff | Scoreline State       |
|-------|-----------|------------|------------|-----------|-----------------------|
| 0     | 0         | 0          | 0          | 0         | draw                  |
| 1     | 5         | 0          | 0          | 0         | draw                  |
| 2     | 10        | 0          | 0          | 0         | draw                  |
| 3     | 15        | 0          | 1          | −1        | losing_by_1           |
| 4     | 20        | 0          | 1          | −1        | losing_by_1           |
| 5     | 25        | 0          | 1          | −1        | losing_by_1           |
| 6     | 30        | 0          | 1          | −1        | losing_by_1           |
| 7     | 35        | 0          | 2          | −2        | losing_by_more_than_1 |
| 8     | 40        | 0          | 3          | −3        | losing_by_more_than_1 |
| 9     | 45        | 0          | 3          | −3        | losing_by_more_than_1 |

**Transition analysis:**
- Block 2 → Block 3: Lyon scores at 13'. State goes from `draw` to `losing_by_1`.
- Block 6 → Block 7: Lyon scores at 30'. State goes from `losing_by_1` to `losing_by_more_than_1`.
- Block 7 → Block 8: Lyon scores at 36'. Remains `losing_by_more_than_1` (goal_diff deepens to −3).
- Block 9: Lyon scores at 49' *during* this block, but state at start reflects 0–3.

### Example: Metz 1–1 Lens (game_id: 2215771)

**Goal minutes:** Home = [31], Away = [23]

| Block | Start Min | Home Goals | Away Goals | Goal Diff | Scoreline State |
|-------|-----------|------------|------------|-----------|-----------------|
| 0     | 0         | 0          | 0          | 0         | draw            |
| 1     | 5         | 0          | 0          | 0         | draw            |
| 2     | 10        | 0          | 0          | 0         | draw            |
| 3     | 15        | 0          | 0          | 0         | draw            |
| 4     | 20        | 0          | 1          | −1        | losing_by_1     |
| 5     | 25        | 0          | 1          | −1        | losing_by_1     |
| 6     | 30        | 1          | 1          | 0         | draw            |
| 7     | 35        | 1          | 1          | 0         | draw            |
| 8     | 40        | 1          | 1          | 0         | draw            |
| 9     | 45        | 1          | 1          | 0         | draw            |

---

## f) Analyses Enabled

With scoreline context per block, the following research questions become testable:

### 1. Does cognitive fatigue differ by scoreline state?
- Group blocks by `scoreline_state` and compare `reorientation_rate`, `shift_latency`, `transition_latency`, `pressure_composite`
- Controls: home/away, match timing (first vs second half — proxied by `block_num`), opponent strength

### 2. Conceding effect: does conceding degrade next-block cognition?
- `conceded_in_block == True` → compare cognitive metrics in `block_num + 1` vs blocks where no goal was conceded
- Hypothesis: conceding a goal causes attentional disruption → higher shift latency in the next block

### 3. Scoring effect: does scoring improve next-block metrics?
- `scored_in_block == True` → compare next-block metrics
- Hypothesis: scoring boosts arousal → faster reorientation

### 4. Marginal vs blowout effects
- Compare `winning_by_1` vs `winning_by_more_than_1` — is there a "protecting the lead" effect?
- Compare `losing_by_1` vs `losing_by_more_than_1` — does blowout cause disengagement?

### 5. Scoreline × match timing interaction
- Does scoreline affect cognition differently early vs late in the match?
- Block 0–8 (first half) vs block 9+ (second half), controlling for scoreline

### 6. Home/away asymmetry controlling for scoreline
- Are home teams' cognitive metrics less affected by losing than away teams'?

### 7. Goal difference as a continuous predictor
- Use `goal_diff` as a numeric covariate (not just categorical) in linear mixed models with `player_id` as random intercept

### 8. Within-match scoreline trajectory
- Identify blocks where scoreline *just changed* (e.g. goal in previous block) vs blocks where it's stable — test whether recent change has acute cognitive effects beyond the static state

---

## Implementation Summary

```
For each of 100 tracking game_ids:
  → Read shape JSON, extract [home_team, away_team, localDate]
  → Normalize team names, look up fixture row
  → Parse HomeGoalMinutes, AwayGoalMinutes into int lists
  → For each tracked block_num [0..max_for_game]:
      start_min = block_num * 5
      home_goals = count(gm < start_min for gm in home_goal_mins)
      away_goals = count(gm < start_min for gm in away_goal_mins)
      goal_diff = home_goals - away_goals
      scoreline_state = classify(goal_diff)
  → Join scoreline_state, home_goals, away_goals, goal_diff onto all
      rows matching this game_id and block_num
```

**Output:** Augmented unified dataset with 8 new columns added to the existing 39.

---

*Plan produced 2026-07-19. All 100 tracking-to-fixture mappings validated. No code written.*
