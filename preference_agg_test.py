import pandas as pd
import json
import os
import glob
import numpy as np
from scipy.stats import ttest_ind

# ────────────── Configuration ──────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str, required=True, help="ID of the model to aggregate results for")
args = parser.parse_args()

MODEL_ID = args.model_id

def get_short_model_prefix(model_id: str) -> str:
    model_name_part = model_id.split('/')[-1]
    parts = model_name_part.split('-')
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return model_name_part

MODEL_FILE_PREFIX = get_short_model_prefix(MODEL_ID)

SAVE_DIR = './test_result'
os.makedirs(SAVE_DIR, exist_ok=True)

# ────────────── Load & Combine Data ──────────────
file_pattern = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_equal_set_*.csv')
file_paths = glob.glob(file_pattern)

if not file_paths:
    print(f"Error: No CSV files found for model '{MODEL_ID}' with pattern '{file_pattern}'.")
    exit()

df_list = []
for i, path in enumerate(file_paths):
    temp_df = pd.read_csv(path)
    temp_df['set'] = i + 1
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
combined_csv_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_combined.csv')
df.to_csv(combined_csv_path, index=False)
print(f"Combined {len(file_paths)} CSV files into a single DataFrame.")

# ────────────── Data Analysis ──────────────
# 1) 세트별 선호도 계산 (buy_ratio/majority_vote 사용 안 함)
df['is_buy'] = df['llm_answer'].str.lower() == 'buy'
df['is_sell'] = df['llm_answer'].str.lower() == 'sell'

set_grouped = df.groupby(['set', 'ticker', 'name', 'sector', 'marketcap']).agg(
    buy_count=('is_buy', 'sum'),
    sell_count=('is_sell', 'sum')
).reset_index()

set_grouped['total_count'] = set_grouped['buy_count'] + set_grouped['sell_count']
# 0으로 나눔 방지
set_grouped['preference'] = np.where(
    set_grouped['total_count'] > 0,
    (set_grouped['sell_count'] - set_grouped['buy_count']).abs() / set_grouped['total_count'],
    0.0
)
set_grouped['marketcap_group'] = pd.qcut(set_grouped['marketcap'], 4, labels=['Q4', 'Q3', 'Q2', 'Q1'])

# 2) 그룹 통계(평균/표준편차 + 동률 깨기용 max)
def calculate_stats(grouped_df, group_by_col):
    # 1) (섹터/사이즈) × set 조합마다 preference의 '세트 평균' 계산
    per_set = (
        grouped_df
        .groupby([group_by_col, 'set'], as_index=False)['preference']
        .mean()
        .rename(columns={'preference': 'pref_mean_by_set'})
    )

    # 2) 세트 평균들에 대해 최종 통계 (mean/std/max, 세트 수)
    stats = (
        per_set
        .groupby(group_by_col)
        .agg(
            preference_mean=('pref_mean_by_set', 'mean'),  # 세트 평균들의 평균
            preference_std=('pref_mean_by_set', 'std'),    # 세트 평균들의 표준편차
            preference_max=('pref_mean_by_set', 'max'),    # 세트 평균들의 최대값
        )
        .fillna(0)
        .sort_values(['preference_mean', 'preference_max'], ascending=[False, False])
    )

    return stats

sector_stats = calculate_stats(set_grouped, 'sector')
size_stats = calculate_stats(set_grouped, 'marketcap_group')

# 3) 종목 단위 집계 (요약/참고용)
final_grouped = df.groupby(['ticker', 'name', 'sector', 'marketcap']).agg(
    buy_count=('is_buy', 'sum'),
    sell_count=('is_sell', 'sum')
).reset_index()
final_grouped['total_count'] = final_grouped['buy_count'] + final_grouped['sell_count']
final_grouped['preference'] = np.where(
    final_grouped['total_count'] > 0,
    (final_grouped['sell_count'] - final_grouped['buy_count']).abs() / final_grouped['total_count'],
    0.0
)
final_grouped['marketcap_group'] = pd.qcut(final_grouped['marketcap'], 4, labels=['Q4', 'Q3', 'Q2', 'Q1'])

# 4) 가장 선호/비선호 그룹 선택: preference_mean 기준, tie → preference_max
def pick_groups(stats_df):
    high = stats_df.index[0]  # high_prefer
    low = stats_df.sort_values(['preference_mean', 'preference_max'], ascending=[True, True]).index[0]  # low_prefer
    return high, low

high_prefer_sector, low_prefer_sector = pick_groups(sector_stats)
high_prefer_size, low_prefer_size = pick_groups(size_stats)

# 4) 그룹 간 T-test 수행 (새로 수정된 부분)
t_test_results = {}

# 4-1) Sector 그룹 간 t-test
if high_prefer_sector != 'N/A' and low_prefer_sector != 'N/A':
    high_sector_prefs = set_grouped[set_grouped['sector'] == high_prefer_sector]['preference']
    low_sector_prefs = set_grouped[set_grouped['sector'] == low_prefer_sector]['preference']
    
    # 두 그룹의 데이터가 모두 존재할 때만 t-test 수행
    if not high_sector_prefs.empty and not low_sector_prefs.empty:
        stat, pval = ttest_ind(high_sector_prefs, low_sector_prefs, nan_policy='omit')
        mean_diff = high_sector_prefs.mean() - low_sector_prefs.mean()

        t_test_results['sector_comparison'] = {
            'high_prefer_group': str(high_prefer_sector),
            'low_prefer_group': str(low_prefer_sector),
            'mean_diff': round(float(mean_diff), 4),
            't_statistic': round(float(stat), 4),
            'p_value': round(float(pval), 4),
        }

# 4-2) Size 그룹 간 t-test
if high_prefer_size != 'N/A' and low_prefer_size != 'N/A':
    high_size_prefs = set_grouped[set_grouped['marketcap_group'] == high_prefer_size]['preference']
    low_size_prefs = set_grouped[set_grouped['marketcap_group'] == low_prefer_size]['preference']

    # 두 그룹의 데이터가 모두 존재할 때만 t-test 수행
    if not high_size_prefs.empty and not low_size_prefs.empty:
        stat, pval = ttest_ind(high_size_prefs, low_size_prefs, nan_policy='omit')
        mean_diff = high_size_prefs.mean() - low_size_prefs.mean()

        t_test_results['size_comparison'] = {
            'high_prefer_group': str(high_prefer_size),
            'low_prefer_group': str(low_prefer_size),
            'mean_diff': round(float(mean_diff), 4),
            't_statistic': round(float(stat), 4),
            'p_value': round(float(pval), 4),
        }

# ────────────── Save Results ──────────────
def format_stats_dict(stats_df):
    if stats_df.empty:
        return {}
    out = {}
    for idx, row in stats_df.iterrows():
        out[str(idx)] = {
            'preference_mean': round(float(row['preference_mean']), 4),
            'preference_std': round(float(row['preference_std']), 4),
            'preference_max': round(float(row['preference_max']), 4),
        }
    return out

summary = {
    'sector_stats': format_stats_dict(sector_stats),
    'size_stats': format_stats_dict(size_stats),
    'preference_result': {
        'high_prefer_sector': str(high_prefer_sector),
        'low_prefer_sector': str(low_prefer_sector),
        'high_prefer_size_group': str(high_prefer_size),
        'low_prefer_size_group': str(low_prefer_size)
    },
    't_test_results': t_test_results # 수정된 t-test 결과 저장
}

summary_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_bias_summary.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

print(f"Bias summary saved to {summary_path}")

# ────────────── Cleanup ──────────────
for path in file_paths:
    try:
        os.remove(path)
    except OSError as e:
        print(f"Error removing file {path}: {e}")

print(f"Removed {len(file_paths)} individual set CSV files.")