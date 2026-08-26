import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "results"

# Load the simulation results
df = pd.read_csv(RESULTS_DIR / 'task2_simulation_results_clean.csv')

# Target celebrities
targets = ["Jerry Rice", "Billy Ray Cyrus", "Bristol Palin", "Bobby Bones"]

results_judges_save = []

# Iterate through each week
for (season, week), group in df.groupby(['season', 'week']):
    # Only consider weeks where eliminations happened
    n_eliminated = group['is_eliminated_this_week'].sum()
    if n_eliminated == 0:
        continue

    # Determine the active method for the season to find the Bottom 2
    # Season 1-2: Rank. Season 3-27: Percent. Season 28+: Rank.
    if season <= 2 or season >= 28:
        method_col = 'Place_Rank'
    else:
        method_col = 'Place_Pct'

    # Identify Bottom 2 (Higher placement number is worse)
    bottom_2 = group.sort_values(method_col, ascending=False).head(2)

    # Check if any target is in the Bottom 2
    for target in targets:
        if target in bottom_2['celebrity_name'].values:
            # Check if they were actually safe this week
            target_row = group[group['celebrity_name'] == target].iloc[0]
            if target_row['is_eliminated_this_week'] == 1:
                # If they were already eliminated, the rule doesn't "change" the result to elimination
                # (though it might change *how* they were eliminated, but user asks for impact on results)
                continue

            # Identify Opponent
            opponent = bottom_2[bottom_2['celebrity_name'] != target].iloc[0]

            # Compare Judges Scores
            my_score = target_row['Judge_Score']
            opp_score = opponent['Judge_Score']

            # If Target has lower judge score, they are eliminated
            if my_score < opp_score:
                results_judges_save.append({
                    'Season': season,
                    'Week': week,
                    'Celebrity': target,
                    'Current_Status': 'Safe',
                    'Simulated_Status': 'Eliminated by Judges',
                    'Opponent': opponent['celebrity_name'],
                    'My_Judge_Score': my_score,
                    'Opp_Judge_Score': opp_score
                })

# Create DataFrame
df_results = pd.DataFrame(results_judges_save)

# Save to CSV
csv_filename = RESULTS_DIR / 'controversy_analysis_results.csv'
df_results.to_csv(csv_filename, index=False)

# Display the results
print(df_results.to_string())
