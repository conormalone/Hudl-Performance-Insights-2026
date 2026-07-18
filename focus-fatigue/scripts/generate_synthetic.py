#!/usr/bin/env python3
"""Generate synthetic unified fatigue dataset for analysis development."""
import numpy as np
import pandas as pd

np.random.seed(42)

games = ['2215790', '2215791']
teams_per_game = {'2215790': [1001, 1002], '2215791': [1003, 1004]}
players_per_team = 11
n_blocks = 18
phases = [1]*9 + [2]*9

records = []
for game_id in games:
    teams = teams_per_game[game_id]
    for team_id in teams:
        for pid_offset in range(1, players_per_team+1):
            player_id = team_id * 100 + pid_offset
            p_base_drift = 1.2 + np.random.exponential(0.3)
            p_base_shift = 0.35 + np.random.exponential(0.08)
            p_base_press = 0.72 + np.random.normal(0, 0.06)
            p_base_trans = 0.45 + np.random.exponential(0.1)
            is_defender = pid_offset <= 4

            for block_idx in range(n_blocks):
                phase = phases[block_idx]
                block_id = f'{phase}_{block_idx}'
                time_fatigue = 0 if block_idx < 3 else (block_idx - 2) * 0.06
                if phase == 2:
                    time_fatigue *= 1.5

                is_critical = block_idx in [4,5,6,7,8,13,14,15,16]
                pressure_score = 1.0
                pressure_score += 0.3 if is_critical else 0
                pressure_score += 0.4 if phase == 2 else 0
                pressure_score += np.random.uniform(0, 0.5)
                pressure_score += time_fatigue * 0.8
                if is_defender:
                    pressure_score += 0.3
                pressure_score = round(pressure_score, 3)

                fatigue_factor = time_fatigue * (1.0 + 0.5 * (pressure_score / 2.0))
                drift = max(0.1, min(8.0, p_base_drift + fatigue_factor*1.8 + np.random.normal(0,0.1)))
                shift_lat = max(0.1, min(3.0, p_base_shift + fatigue_factor*0.5 + np.random.normal(0,0.03)))
                press_acc = max(0.1, min(1.0, p_base_press - fatigue_factor*0.3 + np.random.normal(0,0.02)))
                trans_lat = max(0.1, min(3.0, p_base_trans + fatigue_factor*0.6 + np.random.normal(0,0.04)))

                if pressure_score > 1.8:
                    pressure_cat = 'high'
                elif pressure_score < 1.2:
                    pressure_cat = 'low'
                else:
                    pressure_cat = 'medium'

                opp_nearby = np.random.poisson(3 + pressure_score*1.5)
                depth = 25 + pressure_score*5 + np.random.normal(0, 3)
                reo_count = int(np.random.poisson(2 + pressure_score*1.2))
                trans_count = int(np.random.poisson(1 + pressure_score*0.8))
                n_frames = int(7500 + np.random.normal(0, 200))

                records.append({
                    'game_id': game_id, 'block_id': block_id, 'phase': phase,
                    'player_id': player_id, 'team_id_opta': team_id,
                    'pressure_composite': pressure_score,
                    'pressure_category': pressure_cat,
                    'pressure_quartile': 1 if pressure_cat=='low' else (3 if pressure_cat=='high' else 2),
                    'n_frames': n_frames,
                    'opponents_nearby_mean': round(opp_nearby, 1),
                    'depth_mean': round(depth, 1),
                    'reorientation_count': reo_count,
                    'transition_count': trans_count,
                    'reorientation_rate': round(reo_count/5.0, 2),
                    'transition_rate': round(trans_count/5.0, 2),
                    'positional_drift': round(drift, 3),
                    'shift_latency': round(shift_lat, 3),
                    'pressing_accuracy': round(press_acc, 3),
                    'transition_latency': round(trans_lat, 3),
                })

df = pd.DataFrame(records)
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Pressure categories: {df["pressure_category"].value_counts().to_dict()}')
print(f'Num players: {df["player_id"].nunique()}')
print(f'Num games: {df["game_id"].nunique()}')

output_dir = '/home/conormalone/.openclaw/workspace/speed-check/focus-fatigue/outputs'
df.to_parquet(f'{output_dir}/unified_fatigue_dataset.parquet', index=False)
print(f'Saved to {output_dir}/unified_fatigue_dataset.parquet')

for signal in ['positional_drift', 'shift_latency', 'pressing_accuracy', 'transition_latency']:
    print(f'\n{signal}:')
    for cat in ['low', 'medium', 'high']:
        vals = df[df['pressure_category']==cat][signal]
        print(f'  {cat}: mean={vals.mean():.3f}, std={vals.std():.3f}')
