
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 4)

def infer_cols(df):
    colmap = {}
    cols = [c.lower() for c in df.columns]
    def find_one(candidates):
        for cand in candidates:
            if cand.lower() in cols:
                return df.columns[cols.index(cand.lower())]
        return None
    colmap['env'] = find_one(['env','environment','environment_name','environment_type','env name'])
    colmap['config'] = find_one(['config','configuration','agent','method','method_name','alg','algorithm'])
    colmap['seed'] = find_one(['seed','run','run_id','random_seed','trial','seed id'])
    colmap['episode'] = find_one(['episode','ep','episode_index','episode_no','episode number','episode idx'])
    colmap['env_reward'] = find_one(['env_reward','reward','episode_reward','environment_reward','reward_env','return','env return'])
    colmap['success'] = find_one(['success','done','success_flag','is_success','is success','success?'])
    colmap['steps'] = find_one(['steps','episode_length','length','timesteps','step_count','steps_taken','steps_in_episode'])
    return colmap

def ensure_columns(df, cmap):
    rename = {}
    for k,v in cmap.items():
        if v is None:
            raise ValueError(f"Could not find required column for '{k}'. Found mapping: {cmap}")
        rename[v] = k
    df = df.rename(columns=rename)
    df['episode'] = df['episode'].astype(int)
    df['seed'] = df['seed'].astype(int)
    df['env_reward'] = pd.to_numeric(df['env_reward'], errors='coerce')
    def conv_success(x):
        if pd.isna(x):
            return 0
        if isinstance(x,(int,float)):
            return 1 if x==1 else 0
        s = str(x).strip().lower()
        if s in ['1','true','t','yes','y']:
            return 1
        return 0
    df['success'] = df['success'].apply(conv_success)
    df['steps'] = pd.to_numeric(df['steps'], errors='coerce').fillna(0).astype(int)
    return df

def segment_periods(df, episode_col='episode'):
    def assign_period(group):
        eps = sorted(group[episode_col].unique())
        total = len(eps)
        i1 = int(np.floor(total*0.25))
        i2 = int(np.floor(total*0.75))
        mapping = {}
        for idx, ep in enumerate(eps):
            if idx < i1:
                mapping[ep] = 'Init'
            elif idx < i2:
                mapping[ep] = 'Mid'
            else:
                mapping[ep] = 'Final'
        return group[episode_col].map(mapping)
    return df.groupby(['env','config','seed']).apply(assign_period).reset_index(level=[0,1,2], drop=True)

def centered_moving_average(series, window=10):
    return series.rolling(window=window, center=True, min_periods=1).mean()

def main(xlsx_path, sheet_name='EPISODES', outdir='results_output'):
    os.makedirs(outdir, exist_ok=True)
    figs_dir = os.path.join(outdir, 'figs'); os.makedirs(figs_dir, exist_ok=True)
    tables_dir = os.path.join(outdir, 'tables'); os.makedirs(tables_dir, exist_ok=True)

    # Load
    df_raw = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    colmap = infer_cols(df_raw)
    df = ensure_columns(df_raw.copy(), colmap)

    # Normalize config and env strings
    df['config'] = df['config'].astype(str).str.strip().str.replace('_',' ').str.title()
    df['config'] = df['config'].replace({
        'Baseline Dqn': 'Baseline','Baseline DQN':'Baseline','Shaped Dqn':'Shaped','Shaped DQN':'Shaped',
        'Reward Machine Dqn':'Reward Machine','Reward Machine DQN':'Reward Machine','Rm':'Reward Machine'
    })
    df['env'] = df['env'].astype(str).str.strip().str.title()
    df['env'] = df['env'].replace({
        'Twosubnetenv':'Foundational','Twosubnet Env':'Foundational','TwoSubnetEnv':'Foundational',
        'Enhanced':'Advanced'
    })

    # Aggregated table
    agg = df.groupby(['env','config']).agg(
        AvgReward=('env_reward','mean'),
        StdReward=('env_reward','std'),
        SuccessRate=('success', 'mean'),
        AvgSteps=('steps','mean'),
        StdSteps=('steps','std'),
        CountEpisodes=('episode','count')
    ).reset_index()
    agg.to_csv(os.path.join(tables_dir,'aggregated_by_env_config.csv'), index=False)

    # Period segmentation
    df['period'] = segment_periods(df)
    period_stats = df.groupby(['env','config','period']).agg(
        AvgReward=('env_reward','mean'),
        SuccessRate=('success','mean'),
        AvgSteps=('steps','mean'),
        StdReward=('env_reward','std'),
        CountEpisodes=('episode','count')
    ).reset_index()
    period_stats.to_csv(os.path.join(tables_dir,'period_segment_stats.csv'), index=False)

    # Smoothed curves
    smoothing_window = 10
    smoothed_results = {}
    for env_name in df['env'].unique():
        smoothed_results[env_name] = {}
        env_df = df[df['env']==env_name]
        for config in env_df['config'].unique():
            cfg_df = env_df[env_df['config']==config]
            ep_min = int(cfg_df['episode'].min())
            ep_max = int(cfg_df['episode'].max())
            idx_range = range(ep_min, ep_max+1)
            ep_group = cfg_df.groupby('episode').agg(
                avg_reward=('env_reward','mean'),
                avg_success=('success','mean'),
                avg_steps=('steps','mean'),
                cnt_seeds=('seed','nunique')
            ).reindex(idx_range).reset_index().fillna(method='ffill').fillna(0).set_index('episode')
            ep_group['smoothed_reward'] = centered_moving_average(ep_group['avg_reward'], window=smoothing_window)
            ep_group['smoothed_success'] = centered_moving_average(ep_group['avg_success'], window=smoothing_window)
            ep_group['smoothed_steps'] = centered_moving_average(ep_group['avg_steps'], window=smoothing_window)
            smoothed_results[env_name][config] = ep_group
            # plots
            plt.figure(); plt.plot(ep_group.index, ep_group['smoothed_reward']); plt.title(f"{env_name} - {config} (smoothed reward)")
            plt.xlabel("Episode"); plt.ylabel("Smoothed Avg Reward"); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(figs_dir, f"{env_name}_{config}_smoothed_reward.png")); plt.close()
            plt.figure(); plt.plot(ep_group.index, ep_group['smoothed_success']); plt.title(f"{env_name} - {config} (smoothed success)")
            plt.xlabel("Episode"); plt.ylabel("Smoothed Success Rate"); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(figs_dir, f"{env_name}_{config}_smoothed_success.png")); plt.close()
            plt.figure(); plt.plot(ep_group.index, ep_group['smoothed_steps']); plt.title(f"{env_name} - {config} (smoothed steps)")
            plt.xlabel("Episode"); plt.ylabel("Smoothed Avg Steps"); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(figs_dir, f"{env_name}_{config}_smoothed_steps.png")); plt.close()
            # save smoothed csv
            fname = f"{env_name}__{config}__smoothed.csv".replace(' ','_')
            ep_group.to_csv(os.path.join(tables_dir, fname))

    # Final-quarter summary
    final_summary_rows = []
    for env_name in df['env'].unique():
        for config in df[df['env']==env_name]['config'].unique():
            sub = df[(df['env']==env_name)&(df['config']==config)]
            eps = sorted(sub['episode'].unique())
            total = len(eps)
            cutoff = int(np.floor(total*0.75))
            final_eps = eps[cutoff:] if total>0 else []
            final_df = sub[sub['episode'].isin(final_eps)]
            sm = smoothed_results[env_name].get(config, pd.DataFrame())
            if not sm.empty:
                quarter_start_idx = int(len(sm.index)*0.75)
                quarter_idx = list(sm.index)[quarter_start_idx:]
                mu_reward = sm.loc[quarter_idx,'smoothed_reward'].mean() if len(quarter_idx)>0 else final_df['env_reward'].mean()
                r_final = sm['smoothed_reward'].iloc[-1] if len(sm)>0 else np.nan
                mu_success = sm.loc[quarter_idx,'smoothed_success'].mean() if len(quarter_idx)>0 else final_df['success'].mean()
                s_final = sm['smoothed_success'].iloc[-1] if len(sm)>0 else np.nan
                mu_steps = sm.loc[quarter_idx,'smoothed_steps'].mean() if len(quarter_idx)>0 else final_df['steps'].mean()
                t_final = sm['smoothed_steps'].iloc[-1] if len(sm)>0 else np.nan
            else:
                mu_reward = final_df['env_reward'].mean()
                r_final = final_df['env_reward'].iloc[-1] if len(final_df)>0 else np.nan
                mu_success = final_df['success'].mean()
                s_final = final_df['success'].iloc[-1] if len(final_df)>0 else np.nan
                mu_steps = final_df['steps'].mean()
                t_final = final_df['steps'].iloc[-1] if len(final_df)>0 else np.nan
            final_summary_rows.append({
                'env': env_name, 'config': config,
                'mureward': mu_reward, 'rfinal': r_final,
                'musuccess': mu_success, 'sfinal': s_final,
                'musteps': mu_steps, 'tfinal': t_final
            })
    final_summary = pd.DataFrame(final_summary_rows)
    final_summary.to_csv(os.path.join(tables_dir,'final_quarter_summary.csv'), index=False)

    # Wilcoxon tests
    wilco_results = []
    for env_name in df['env'].unique():
        env_df = df[df['env']==env_name]
        per_seed_mean = env_df.groupby(['config','seed'])['env_reward'].mean().unstack(level=0)
        if 'Baseline' not in per_seed_mean.columns:
            continue
        baseline = per_seed_mean['Baseline'].dropna()
        for other in [c for c in per_seed_mean.columns if c!='Baseline']:
            other_series = per_seed_mean[other].dropna()
            common_index = baseline.index.intersection(other_series.index)
            if len(common_index) < 2:
                wilco_results.append({'env':env_name,'comparison':f'Baseline vs {other}','n_pairs':len(common_index),'W_statistic':np.nan,'p_value':np.nan})
                continue
            b = baseline.loc[common_index]
            o = other_series.loc[common_index]
            try:
                stat, p = wilcoxon(b, o, alternative='two-sided')
            except Exception as e:
                stat, p = np.nan, np.nan
            wilco_results.append({'env':env_name,'comparison':f'Baseline vs {other}','n_pairs':len(common_index),'W_statistic':stat,'p_value':p})
    wilco_df = pd.DataFrame(wilco_results)
    wilco_df.to_csv(os.path.join(tables_dir,'wilcoxon_results.csv'), index=False)

    print(f"All results saved to: {outdir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python reproduce_results.py /path/to/All_Results.xlsx [sheet_name]")
        sys.exit(1)
    xlsx = sys.argv[1]
    sheet = sys.argv[2] if len(sys.argv)>2 else 'EPISODES'
    main(xlsx, sheet_name=sheet)
