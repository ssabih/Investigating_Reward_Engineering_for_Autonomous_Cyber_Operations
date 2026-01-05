import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import os

EXCEL_FILE = "All_Results.xlsx"
OUTDIR = "rl_analysis_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# --- LOAD DATA ---
df = pd.read_excel(EXCEL_FILE, sheet_name="EPISODES", engine="openpyxl")
if df['success'].dtype != int:
    df['success'] = df['success'].astype(str).str.lower().map({'true': 1, 'false': 0, '1': 1, '0': 0}).fillna(0).astype(int)

env_windows = {'Foundational': 10, 'Advanced': 50}
env_final_frac = 0.2

# --- AGGREGATE LEARNING CURVES ---
curve = (
    df.groupby(['env', 'config', 'episode'])
    .agg(reward_mean=('reward_env', 'mean'),
         reward_std=('reward_env', 'std'),
         success_rate=('success', 'mean'))
    .reset_index()
)

# --- SUMMARY METRICS PER RUN ---
summary_rows = []
for (env, config, seed), sub in df.groupby(['env', 'config', 'seed']):
    sub = sub.sort_values('episode')
    n_ep = len(sub)
    final_start = int(np.ceil(n_ep * (1 - env_final_frac)))
    final = sub.iloc[final_start:]
    auc_reward = sub['reward_env'].mean()
    final_reward = final['reward_env'].mean()
    final_success = final['success'].mean()
    window = env_windows.get(env, 10)
    rolling = sub['success'].rolling(window=window, min_periods=window).mean()
    idx = np.where(rolling.values >= 0.8)[0]
    ep_to80 = int(sub.iloc[idx[0]].episode) if len(idx) > 0 else np.nan
    summary_rows.append({
        'env': env, 'config': config, 'seed': seed,
        'final_reward_mean': final_reward,
        'final_success_rate': final_success,
        'auc_reward': auc_reward,
        'episodes_to_80pct_success': ep_to80,
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(f"{OUTDIR}/summary_per_seed.csv", index=False)

# --- AGGREGATE SUMMARY ---
agg = (
    summary.groupby(['env', 'config'])
    .agg(
        final_reward_mean=('final_reward_mean', 'mean'),
        final_reward_std=('final_reward_mean', 'std'),
        final_success_rate=('final_success_rate', 'mean'),
        final_success_std=('final_success_rate', 'std'),
        auc_reward=('auc_reward', 'mean'),
        auc_reward_std=('auc_reward', 'std'),
        ep80_median=('episodes_to_80pct_success', 'median'),
        ep80_iqr=('episodes_to_80pct_success', lambda x: np.subtract(*np.percentile(x.dropna(), [75, 25])) if len(x.dropna()) > 0 else np.nan)
    ).reset_index()
)
agg.to_csv(f"{OUTDIR}/agg_summary.csv", index=False)

# --- WILCOXON TESTS ---
def wilcoxon_by_seed(df, env, metric, a, b):
    a_vals = df[(df.env == env) & (df.config == a)].sort_values('seed')[metric].values
    b_vals = df[(df.env == env) & (df.config == b)].sort_values('seed')[metric].values
    mask = ~np.isnan(a_vals) & ~np.isnan(b_vals)
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan
    stat, p = wilcoxon(a_vals[mask], b_vals[mask])
    effect = (b_vals[mask] - a_vals[mask]).mean()
    return stat, p, effect

wilcoxon_rows = []
for env in df['env'].unique():
    for metric in ['final_reward_mean', 'final_success_rate', 'auc_reward', 'episodes_to_80pct_success']:
        for (a, b) in [('Baseline DQN', 'Shaped DQN'), ('Baseline DQN', 'Reward Machine DQN')]:
            stat, p, effect = wilcoxon_by_seed(summary, env, metric, a, b)
            wilcoxon_rows.append({
                'env': env, 'metric': metric, 'A': a, 'B': b,
                'stat': stat, 'p': p, 'effect': effect
            })
wilcoxon_df = pd.DataFrame(wilcoxon_rows)
# Holm correction
for env in df['env'].unique():
    for metric in ['final_reward_mean', 'final_success_rate', 'auc_reward', 'episodes_to_80pct_success']:
        mask = (wilcoxon_df.env == env) & (wilcoxon_df.metric == metric)
        ps = wilcoxon_df.loc[mask, 'p']
        order = np.argsort(ps)
        m = len(ps)
        adj = [min((m - k) * ps.iloc[idx], 1.0) for k, idx in enumerate(order)]
        wilcoxon_df.loc[mask, 'p_holm'] = adj
wilcoxon_df.to_csv(f"{OUTDIR}/wilcoxon_results.csv", index=False)

# --- PLOTS ---
sns.set(style="whitegrid", context="paper")
for env in df['env'].unique():
    sub = curve[curve.env == env]
    plt.figure(figsize=(10, 5))
    for config in sub['config'].unique():
        d = sub[sub.config == config]
        plt.plot(d.episode, d.reward_mean, label=config)
    plt.title(f"{env}: Reward Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{env}_reward_curve.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    for config in sub['config'].unique():
        d = sub[sub.config == config]
        plt.plot(d.episode, d.success_rate, label=config)
    plt.title(f"{env}: Success Rate Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{env}_success_curve.png")
    plt.close()

    # Final performance bars
    sub_sum = summary[summary.env == env]
    plt.figure(figsize=(8, 4))
    sns.barplot(data=sub_sum, x='config', y='final_reward_mean', ci=95)
    plt.title(f"{env}: Final Reward (last 20%)")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{env}_final_reward_bar.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=sub_sum, x='config', y='final_success_rate', ci=95)
    plt.title(f"{env}: Final Success Rate (last 20%)")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{env}_final_success_bar.png")
    plt.close()

# --- LATEX TABLES ---
with open(f"{OUTDIR}/table_performance_summary.tex", "w") as f:
    f.write(agg.to_latex(index=False, float_format="%.3f"))
with open(f"{OUTDIR}/table_wilcoxon.tex", "w") as f:
    f.write(wilcoxon_df.to_latex(index=False, float_format="%.4f"))

print(f"Analysis complete! All outputs are in the '{OUTDIR}' directory.")