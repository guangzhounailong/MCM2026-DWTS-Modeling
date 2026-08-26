import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "2026_MCM_Problem_C_Data.csv"
VOTES_PATH = ROOT / "data" / "results" / "final_estimated_votes_panel_2.csv"
OUTPUT_PATH = ROOT / "data" / "results" / "task4_driver_analysis_results.csv"

# 忽略不必要的警告
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


def solve_task4():
    print("Step 1: Loading and Preprocessing Data...")

    # 1.1 读取数据
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH)
        df_task1 = pd.read_csv(VOTES_PATH)
    except FileNotFoundError:
        print(
            "Error: Files not found. Please make sure '2026_MCM_Problem_C_Data.csv' and 'final_estimated_votes_panel_2.csv' are in the working directory.")
        return

    # 1.2 处理原始宽表 (Raw Data Processing)
    score_cols = [c for c in df_raw.columns if 'judge' in c]
    score_records = []

    for idx, row in df_raw.iterrows():
        season = row['season']
        contestant = str(row['celebrity_name']).strip()

        for w in range(1, 16):
            week_cols = [c for c in score_cols if f'week{w}_' in c]
            if not week_cols: continue

            scores = pd.to_numeric(row[week_cols], errors='coerce')
            if scores.isna().all(): continue

            avg_score = scores.mean()
            if np.isnan(avg_score) or avg_score == 0: continue

            score_records.append({
                'Season': season,
                'Contestant': contestant,
                'Week': w,
                'Raw_Judge_Score': avg_score
            })

    df_scores = pd.DataFrame(score_records)

    # 1.3 数据合并 (Merge)
    df_task1['Contestant'] = df_task1['Contestant'].str.strip()
    df_merged = pd.merge(df_task1, df_scores, on=['Season', 'Week', 'Contestant'], how='inner')

    # 1.4 合并元数据
    meta_cols = ['season', 'celebrity_name', 'celebrity_age_during_season', 'celebrity_industry', 'ballroom_partner']
    df_meta = df_raw[meta_cols].drop_duplicates()
    df_meta.columns = ['Season', 'Contestant', 'Age', 'Industry', 'Pro_Partner']
    df_meta['Contestant'] = df_meta['Contestant'].str.strip()
    df_final = pd.merge(df_merged, df_meta, on=['Season', 'Contestant'], how='inner')

    # 1.5 行业清洗
    top_industries = df_final['Industry'].value_counts().nlargest(5).index.tolist()
    df_final['Industry_Simple'] = df_final['Industry'].apply(lambda x: x if x in top_industries else 'Other')

    print(f"Data merged. Total samples: {len(df_final)}")

    # ==========================================
    # Step 2: 变量变换 (Transformations)
    # ==========================================
    epsilon = 1e-6
    df_final['Log_Fan_Vote'] = np.log(df_final['Est_Fan_Vote'] + epsilon)
    scaler_fan = StandardScaler()
    df_final['Y_Fan_Std'] = scaler_fan.fit_transform(df_final[['Log_Fan_Vote']])

    scaler_judge = StandardScaler()
    df_final['Y_Judge_Std'] = scaler_judge.fit_transform(df_final[['Raw_Judge_Score']])

    df_final['Age'] = pd.to_numeric(df_final['Age'], errors='coerce')
    df_final = df_final.dropna(subset=['Age'])

    # ==========================================
    # Step 3: 模型拟合 (Mixed Effects Models)
    # ==========================================
    print("Step 3: Fitting Models...")
    formula = " ~ Age + C(Industry_Simple)"

    model_j = smf.mixedlm("Y_Judge_Std" + formula, df_final,
                          groups=df_final["Season"],
                          vc_formula={"Pro": "0 + C(Pro_Partner)"})
    res_j = model_j.fit(reml=True)

    model_f = smf.mixedlm("Y_Fan_Std" + formula, df_final,
                          groups=df_final["Season"],
                          vc_formula={"Pro": "0 + C(Pro_Partner)"})
    res_f = model_f.fit(reml=True)

    # ==========================================
    # Step 4: 生成 Wald Test 表格并保存 CSV
    # ==========================================
    print("Step 4: Generating Results and Saving CSV...")

    def clean_summary(res):
        summ = res.summary().tables[1]
        df = pd.DataFrame(summ)
        df['Feature'] = df.index
        df['Coef'] = pd.to_numeric(df['Coef.'])
        df['StdErr'] = pd.to_numeric(df['Std.Err.'])
        return df[~df['Feature'].str.contains('Group|Scale|Var')]

    df_res_j = clean_summary(res_j)
    df_res_f = clean_summary(res_f)

    comparison = pd.merge(df_res_j[['Feature', 'Coef', 'StdErr']],
                          df_res_f[['Feature', 'Coef', 'StdErr']],
                          on='Feature', suffixes=('_Judge', '_Fan'))

    comparison['Diff'] = comparison['Coef_Judge'] - comparison['Coef_Fan']
    comparison['SE_Diff'] = np.sqrt(comparison['StdErr_Judge'] ** 2 + comparison['StdErr_Fan'] ** 2)
    comparison['Z_Score'] = comparison['Diff'] / comparison['SE_Diff']
    comparison['P_Value'] = 2 * (1 - norm.cdf(np.abs(comparison['Z_Score'])))

    comparison['Significant_Diff'] = comparison['P_Value'] < 0.05

    output_table = comparison[['Feature', 'Coef_Judge', 'Coef_Fan', 'P_Value', 'Significant_Diff']].copy()
    output_table.columns = ['Feature', 'Beta_Judge', 'Beta_Fan', 'P_Value', 'Significant_Diff']

    print("\n=== 驱动因素对比结果 (Wald Test) ===")
    print(output_table)

    # --- 保存结果为 CSV ---
    csv_filename = OUTPUT_PATH
    output_table.to_csv(csv_filename, index=False)
    print(f"\n[Success] Analysis results saved to: {csv_filename}")


if __name__ == "__main__":
    solve_task4()
