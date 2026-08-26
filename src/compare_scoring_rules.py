import pandas as pd
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data" / "processed" / "mcm_panel_by_week.csv"
VOTES_PATH = ROOT / "data" / "results" / "final_estimated_votes_panel_2.csv"
RESULTS_DIR = ROOT / "data" / "results"

# 1. 数据加载与合并

df_panel = pd.read_csv(PANEL_PATH)
df_theta = pd.read_csv(VOTES_PATH)

# 标准化列名
df_theta = df_theta.rename(columns={
    'Season': 'season',
    'Week': 'week',
    'Contestant': 'celebrity_name',
    'Est_Fan_Vote': 'theta'
})

# 清理人名
df_panel['celebrity_name'] = df_panel['celebrity_name'].astype(str).str.strip()
df_theta['celebrity_name'] = df_theta['celebrity_name'].astype(str).str.strip()

# 合并
df = pd.merge(df_panel, df_theta, on=['season', 'week', 'celebrity_name'], how='inner')


# 2. 模拟逻辑
def simulate_outcome(group):
    g = group.copy()

    # --- Rank Method ---
    # Rank J: Higher score -> Lower rank (1 is best)
    g['Rank_J'] = g['judge_total_model'].rank(ascending=False, method='min')
    # Rank F: Higher theta -> Lower rank (1 is best)
    g['Rank_F'] = g['theta'].rank(ascending=False, method='min')
    # Sum
    g['Score_Rank'] = g['Rank_J'] + g['Rank_F']
    # Final Place: Sort by Score (asc), then Theta (desc) for tie-breaker
    g = g.sort_values(by=['Score_Rank', 'theta'], ascending=[True, False])
    g['Place_Rank'] = range(1, len(g) + 1)

    # --- Percent Method ---
    total_j = g['judge_total_model'].sum()
    if total_j == 0: total_j = 1
    g['Pct_J'] = g['judge_total_model'] / total_j

    total_f = g['theta'].sum()
    if total_f == 0: total_f = 1
    g['Pct_F'] = g['theta'] / total_f

    g['Score_Pct'] = g['Pct_J'] + g['Pct_F']
    # Final Place: Sort by Score (desc)
    g = g.sort_values(by='Score_Pct', ascending=False)
    g['Place_Pct'] = range(1, len(g) + 1)

    # Calculate Diff
    g['Diff_Rank_Minus_Pct'] = g['Place_Rank'] - g['Place_Pct']

    return g


df_sim = pd.concat(
    [simulate_outcome(group) for _, group in df.groupby(['season', 'week'])],
    ignore_index=True,
)


# 3. 输出文件 1: 详细结果 (Clean Results)
columns_to_keep = [
    'season', 'week', 'celebrity_name',
    'judge_total_model', 'theta',
    'Place_Rank', 'Place_Pct', 'Diff_Rank_Minus_Pct',
    'is_eliminated_this_week'
]
df_clean = df_sim[columns_to_keep].copy()
df_clean = df_clean.rename(columns={
    'judge_total_model': 'Judge_Score',
    'theta': 'Est_Fan_Vote',
    'Diff_Rank_Minus_Pct': 'Placement_Diff'
})
df_clean.to_csv(RESULTS_DIR / 'task2_simulation_results_clean.csv', index=False)
print("File 1 Saved: task2_simulation_results_clean.csv")
# 4. 输出文件 2: 验证摘要 (Validation Summary)
# 仅筛选出发生了淘汰的周次
elim_weeks = df_sim[df_sim['is_eliminated_this_week'] == 1].groupby(['season', 'week']).size().reset_index()
val_rows = []

for _, row in elim_weeks.iterrows():
    s, w = row['season'], row['week']
    week_data = df_sim[(df_sim['season'] == s) & (df_sim['week'] == w)]

    # 获取实际淘汰者名单
    actual = set(week_data[week_data['is_eliminated_this_week'] == 1]['celebrity_name'])
    k = len(actual)  # 本周淘汰了 k 个人

    # 模拟预测: 取排名最后 k 名的选手
    # Rank Method: 数值最大的前 k 个
    pred_rank = set(week_data.sort_values('Place_Rank', ascending=False).head(k)['celebrity_name'])
    # Percent Method: 数值最大的前 k 个
    pred_pct = set(week_data.sort_values('Place_Pct', ascending=False).head(k)['celebrity_name'])

    val_rows.append({
        'Season': s,
        'Week': w,
        'Num_Eliminated': k,
        'Actual_Eliminated': ", ".join(actual),
        'Pred_Rank_Eliminated': ", ".join(pred_rank),
        'Pred_Pct_Eliminated': ", ".join(pred_pct),
        # 只要预测名单和实际名单有交集，就算命中 (Hit)
        'Hit_Rank': 1 if actual.intersection(pred_rank) else 0,
        'Hit_Pct': 1 if actual.intersection(pred_pct) else 0
    })

df_val = pd.DataFrame(val_rows)
df_val.to_csv(RESULTS_DIR / 'task2_validation_summary.csv', index=False)
print("File 2 Saved: task2_validation_summary.csv")

# 输出验证统计
print("\n=== Model Validation Stats ===")
print(f"Total Elimination Weeks: {len(df_val)}")
print(f"Rank Method Accuracy:   {df_val['Hit_Rank'].mean():.2%}")
print(f"Percent Method Accuracy: {df_val['Hit_Pct'].mean():.2%}")
