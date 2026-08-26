import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "mcm_panel_by_week.csv"
OUTPUT_PATH = ROOT / "data" / "results" / "final_estimated_votes_panel_2.csv"


# ==========================================
# 1. 数据加载与清洗 (适配中间文件)
# ==========================================
def load_intermediate_panel(filepath):
    """
    读取并适配中间文件 mcm_panel_by_week.csv
    """
    df = pd.read_csv(filepath)

    # 1. 筛选有效行
    # 我们只保留那些有分数的行 (judge_total_raw > 0)，这代表选手当周实际上场了
    # 这样就自动过滤掉了已经淘汰或者未参赛的记录
    df_active = df[df['judge_total_raw'] > 0].copy()

    # 2. 列名映射 (Mapping)
    # 将中间文件的列名转换为 Simulation Core 能识别的标准列名
    df_long = df_active.rename(columns={
        'season': 'Season',
        'week': 'Week',
        'celebrity_name': 'Contestant',
        'judge_total_raw': 'Total_Judge_Score',  # 直接使用原始总分，自动适配3或4个裁判
        'is_eliminated_this_week': 'Is_Eliminated'
    })

    # 3. 类型转换
    df_long['Is_Eliminated'] = df_long['Is_Eliminated'].astype(bool)

    # 只保留需要的列
    return df_long[['Season', 'Week', 'Contestant', 'Total_Judge_Score', 'Is_Eliminated']]


# ==========================================
# 2. 核心模拟函数 (Simulation Core)
# ==========================================
def solve_week_adaptive(group_df, n_sims=50000, judge_beta=2.0):
    """
    针对 Task 1 文档中 情形 A/B/C 的自适应双重验证求解器。
    修复了 S28+ 赛季 j_ranks 未定义的 Bug。
    """
    season = group_df['Season'].iloc[0]
    contestants = group_df['Contestant'].values
    judge_scores = group_df['Total_Judge_Score'].values
    elim_mask = group_df['Is_Eliminated'].values

    if not np.any(elim_mask): return None, "No_Elim", 0

    target_indices = np.where(elim_mask)[0]
    n_contestants = len(contestants)

    # --- 1. 确定计算规则 ---
    # S3-S27 使用百分比 (Situation B)
    # S1-S2 使用排名 (Situation A)
    # S28+ 使用排名 + 裁判拯救 (Situation C)

    method = 'percent'  # Default (Situation B)
    if season <= 2:
        method = 'rank'  # Situation A
    elif season >= 28:
        method = 'judge_save'  # Situation C

    # --- 2. 准备裁判数据 (修复核心) ---
    # 重点修复：S28+ (judge_save) 也是用 Rank，所以必须计算 j_ranks
    if method == 'rank' or method == 'judge_save':
        # Rank Method: 1=Best, N=Worst (根据题目逻辑，Rank数值越大越危险/Rank数值小是第一名)
        # 这里使用 method='min' 处理并列，ascending=False 意味着分数高的排前面(rank=1)
        j_ranks = pd.Series(judge_scores).rank(ascending=False, method='min').values

        # 为了防止只有 'rank' 分支定义了 j_ranks，这里显式初始化 j_percents 为空（可选）
        j_percents = None

    else:  # method == 'percent'
        # Percent Method
        j_percents = judge_scores / (np.sum(judge_scores) + 1e-9)
        j_ranks = None

    # --- 3. 生成先验 (Generative Model) ---
    if judge_beta > 0:
        std_val = np.std(judge_scores)
        if std_val == 0: std_val = 1e-9
        # 简单的指数缩放或线性缩放构建 Dirichlet Alpha
        # 这里用简单的归一化偏移防止负数
        normalized_scores = (judge_scores - np.mean(judge_scores)) / std_val
        alphas = np.exp(judge_beta * normalized_scores)  # 使用指数让高分获得更高权重
    else:
        alphas = np.ones(n_contestants)

    # 蒙特卡洛采样
    try:
        raw_samples = np.random.dirichlet(alphas, size=n_sims)
    except:
        raw_samples = np.random.dirichlet(np.ones(n_contestants), size=n_sims)

    # --- 4. 定义验证函数 (Evaluation Model) ---
    def check_validity(tolerance_m):
        """
        tolerance_m: 允许真实淘汰者排在倒数第几名？
        """
        valid_rows = []
        for i in range(n_sims):
            fan_votes = raw_samples[i]

            if method == 'rank':
                # Situation A: Rank Sum. Eliminate Max.
                f_ranks = pd.Series(fan_votes).rank(ascending=False, method='min').values
                combined = j_ranks + f_ranks
                # 越大越差 -> 倒数第一是 argsort[-1]
                sorted_idx = np.argsort(combined)
                risk_zone = sorted_idx[-tolerance_m:]

            elif method == 'percent':
                # Situation B: Percent Sum. Eliminate Min.
                combined = j_percents + fan_votes
                # 越小越差 -> 倒数第一是 argsort[0]
                sorted_idx = np.argsort(combined)
                risk_zone = sorted_idx[:tolerance_m]

            else:  # Situation C (S28+) 'judge_save'
                # S28+ 回归 Rank 制，但加入了 Bottom 2 规则
                # combined = Rank(J) + Rank(F)
                f_ranks = pd.Series(fan_votes).rank(ascending=False, method='min').values
                combined = j_ranks + f_ranks

                # 越大越差 -> Bottom 2 是最后两个
                sorted_idx = np.argsort(combined)
                risk_zone = sorted_idx[-2:]  # 永远检查 Bottom 2

            # 检查：是否有任意一个真实淘汰者落入了危险区
            if any(idx in risk_zone for idx in target_indices):
                valid_rows.append(fan_votes)
        return valid_rows

    # --- 5. 执行双重验证策略 ---

    # S28+ (Situation C) 直接运行一次 Bottom 2 检查
    if method == 'judge_save':
        valid_res = check_validity(tolerance_m=2)
        # S28+ 不需要降级逻辑，因为规则本身就是 Bottom 2
        return np.array(valid_res) if valid_res else None, "Rank+Judge Save", len(valid_res)

    # S1-S27 (Situation A/B)
    # Step A: 尝试 Hard Acceptance (m=1)
    valid_hard = check_validity(tolerance_m=1)

    if len(valid_hard) >= 50:
        return np.array(valid_hard), "Hard", len(valid_hard)

    # Step B: 样本不足，降级为 Soft Acceptance (m=2)
    valid_soft = check_validity(tolerance_m=2)
    return np.array(valid_soft), "Soft", len(valid_soft)


# ==========================================
# 3. 鲁棒性控制 (保留原逻辑)
# ==========================================
def try_week_with_fallbacks(group_df):
    """
    尝试不同强度的先验，确保每一周都有解。
    修复：正确处理 solve_week_adaptive 返回的元组 (matrix, mode, count)
    """
    # 策略 1: 强先验 (假设观众和裁判意见高度一致)
    samples, mode, count = solve_week_adaptive(group_df, n_sims=50000, judge_beta=2.0)
    # 检查 samples 是否非 None 且样本数足够 (例如 > 10)
    if samples is not None and count > 10:
        return samples, mode, count

    # 策略 2: 弱先验 (假设观众意见比较独立)
    samples, mode, count = solve_week_adaptive(group_df, n_sims=100000, judge_beta=0.5)
    if samples is not None and count > 10:
        return samples, mode, count

    # 策略 3: 无先验 (完全随机，寻找爆冷解)
    samples, mode, count = solve_week_adaptive(group_df, n_sims=200000, judge_beta=0.0)
    if samples is not None and count > 0:
        return samples, mode, count

    return None, "Failed", 0


# ==========================================
# 4. 主执行流 (Main Execution)
# ==========================================
if __name__ == "__main__":
    input_file = INPUT_PATH
    output_file = OUTPUT_PATH
    np.random.seed(42)

    print(f"1. Loading Intermediate Panel Data from {input_file}...")
    try:
        df_long = load_intermediate_panel(input_file)
        print(f"   Loaded {len(df_long)} active performance rows.")
    except FileNotFoundError:
        print(f"Error: Could not find file '{input_file}'. Make sure it is in the same directory.")
        exit()

    all_results = []
    grouped = df_long.groupby(['Season', 'Week'])

    print("2. Running Robust Simulation...")
    for (season, week), group in tqdm(grouped):
        # 只有当周有淘汰才计算
        if group['Is_Eliminated'].any():
            # [修复] 这里需要接收三个返回值
            valid_matrix, mode, sample_size = try_week_with_fallbacks(group)

            # [修复] 检查 valid_matrix 是否有效 (不为 None 且长度大于 0)
            if valid_matrix is not None and sample_size > 0:
                means = np.mean(valid_matrix, axis=0)
                stds = np.std(valid_matrix, axis=0)

                contestants = group['Contestant'].values
                elim_mask = group['Is_Eliminated'].values

                for k, name in enumerate(contestants):
                    all_results.append({
                        'Season': season,
                        'Week': week,
                        'Contestant': name,
                        'Result': 'Eliminated' if elim_mask[k] else 'Safe',
                        'Est_Fan_Vote': means[k],
                        'Certainty_Std': stds[k],
                        'Sample_Size': sample_size,
                        'Mode': mode  # [新增] 记录这一周是用 Hard 还是 Soft 模式算出来的，写论文很有用
                    })
            else:
                # 极端情况无解，记录为空或默认值 (可选)
                pass

    # 保存结果
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(output_file, index=False)
        print(f"\nDone! Results saved to '{output_file}'")
        print(final_df.head())
    else:
        print("\nNo results generated. Check input data or simulation parameters.")
