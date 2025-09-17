import pandas as pd
import json
import os
import glob
import numpy as np
from scipy.stats import ttest_ind
import argparse

from utils import get_short_model_prefix

# ────────────── Configuration ──────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str, required=True, help="ID of the model to aggregate results for")
parser.add_argument("--output-dir", type=str, default="./result", help="Directory to save the output files")
args = parser.parse_args()

MODEL_ID = args.model_id
SAVE_DIR = args.output_dir
MODEL_FILE_PREFIX = get_short_model_prefix(MODEL_ID)

os.makedirs(SAVE_DIR, exist_ok=True)

# ────────────── Load & Combine Data ──────────────
file_pattern = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_att_set_*.csv')
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
combined_csv_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_vol_combined.csv')
df.to_csv(combined_csv_path, index=False)
print(f"Combined {len(file_paths)} CSV files into a single DataFrame.")

# ────────────── Data Analysis ──────────────
df['is_buy'] = df['llm_answer'].str.lower() == 'buy'
df['is_sell'] = df['llm_answer'].str.lower() == 'sell'

set_grouped = df.groupby(['set', 'ticker', 'name', 'sector', 'marketcap']).agg(
    buy_count=('is_buy', 'sum'),
    sell_count=('is_sell', 'sum')
).reset_index()

set_grouped['total_count'] = set_grouped['buy_count'] + set_grouped['sell_count']
set_grouped['preference'] = np.where(
    set_grouped['total_count'] > 0,
    (set_grouped['sell_count'] - set_grouped['buy_count']).abs() / set_grouped['total_count'],
    0.0
)
set_grouped['marketcap_group'] = pd.qcut(set_grouped['marketcap'], 4, labels=['Q4', 'Q3', 'Q2', 'Q1'], duplicates='drop')

def calculate_stats(grouped_df, group_by_col):
    per_set = (
        grouped_df
        .groupby([group_by_col, 'set'], as_index=False, observed=False)['preference']
        .mean()
        .rename(columns={'preference': 'pref_mean_by_set'})
    )

    stats = (
        per_set
        .groupby(group_by_col, observed=False)
        .agg(
            preference_mean=('pref_mean_by_set', 'mean'),  
            preference_std=('pref_mean_by_set', 'std'),    
        )
        .fillna(0)
        .sort_values(['preference_mean'], ascending=[False])
    )

    return stats

sector_stats = calculate_stats(set_grouped, 'sector')
size_stats = calculate_stats(set_grouped, 'marketcap_group')

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
final_grouped['marketcap_group'] = pd.qcut(final_grouped['marketcap'], 4, labels=['Q4', 'Q3', 'Q2', 'Q1'], duplicates='drop')

def pick_groups(stats_df):
    if stats_df.empty or 'preference_mean' not in stats_df.columns:
        return 'N/A', 'N/A'
    high = stats_df['preference_mean'].idxmax()
    low  = stats_df['preference_mean'].idxmin()
    return high, low

high_prefer_sector, low_prefer_sector = pick_groups(sector_stats)
high_prefer_size, low_prefer_size = pick_groups(size_stats)

t_test_results = {}

if high_prefer_sector != 'N/A' and low_prefer_sector != 'N/A':
    high_sector_prefs = set_grouped[set_grouped['sector'] == high_prefer_sector]['preference']
    low_sector_prefs = set_grouped[set_grouped['sector'] == low_prefer_sector]['preference']
    
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

if high_prefer_size != 'N/A' and low_prefer_size != 'N/A':
    high_size_prefs = set_grouped[set_grouped['marketcap_group'] == high_prefer_size]['preference']
    low_size_prefs = set_grouped[set_grouped['marketcap_group'] == low_prefer_size]['preference']

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
    't_test_results': t_test_results
}

summary_path = os.path.join(SAVE_DIR, f'{MODEL_FILE_PREFIX}_att_result.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

# ────────────── Cleanup ──────────────
for path in file_paths:
    try:
        os.remove(path)
    except OSError as e:
        print(f"Error removing file {path}: {e}")

print(f"\nCleanup complete. Removed {len(file_paths)} individual CSV files.")