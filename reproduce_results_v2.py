
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Settings ---
XLSX_PATH = "All_Results.xlsx"  # hardcoded as requested
SHEET_NAME = "EPISODES"
OUTDIR = "results_output_v2"
FIGS_DIR = os.path.join(OUTDIR, "figs")

os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (9, 4.8)  # comfortable width

def centered_moving_average(series, window=10):
    return series.rolling(window=window, center=True, min_periods=1).mean()

def load_and_normalize(xlsx_path, sheet_name="EPISODES"):
    df_raw = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # Flexible column mapping
    cols = {c.lower(): c for c in df_raw.columns}
    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    col_env = pick("env", "environment", "environment_name")
    col_cfg = pick("config", "configuration", "agent", "method")
    col_seed = pick("seed", "run", "run_id", "random_seed", "trial")
    col_ep = pick("episode", "ep", "episode_index", "episode no", "episode number")
    col_rew = pick("reward_env", "env_reward", "environment_reward", "return", "episode_reward")
    col_succ = pick("success", "is_success", "done")
    col_steps = pick("steps_in_episode", "steps", "episode_length", "length", "timesteps")

    required = [col_env, col_cfg, col_seed, col_ep, col_rew, col_succ, col_steps]
    if any(c is None for c in required):
        missing = ["env","config","seed","episode","reward_env","success","steps_in_episode"]
        raise ValueError(f"Missing expected columns (found mapping: {dict(zip(missing, required))})")

    df = df_raw.rename(columns={
        col_env: "env",
        col_cfg: "config",
        col_seed: "seed",
        col_ep: "episode",
        col_rew: "env_reward",
        col_succ: "success",
        col_steps: "steps"
    })

    # Normalize strings / types
    df["config"] = (df["config"].astype(str).str.strip().str.replace("_"," ").str.title()
                    .replace({"Baseline Dqn":"Baseline","Shaped Dqn":"Shaped",
                              "Reward Machine Dqn":"Reward Machine","Rm":"Reward Machine"}))
    df["env"] = (df["env"].astype(str).str.strip().str.title()
                 .replace({"Twosubnetenv":"Foundational","Twosubnet Env":"Foundational",
                           "Twosubnet":"Foundational",
                           "Enhanced":"Advanced"}))

    df["episode"] = pd.to_numeric(df["episode"], errors="coerce").astype(int)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype(int)
    df["env_reward"] = pd.to_numeric(df["env_reward"], errors="coerce")
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")
    # success to 0/1
    def conv_success(x):
        if pd.isna(x): return 0
        if isinstance(x,(int,float)): return 1 if x==1 else 0
        s = str(x).strip().lower()
        return 1 if s in ("1","true","t","yes","y") else 0
    df["success"] = df["success"].apply(conv_success)

    return df

def per_episode_averages(df, env_name):
    sub = df[df["env"]==env_name]
    # group by episode and config, average across seeds
    g = sub.groupby(["episode","config"]).agg(
        avg_reward=("env_reward","mean"),
        avg_steps=("steps","mean"),
        avg_success=("success","mean")
    ).reset_index()
    # pivot per metric
    piv_reward = g.pivot(index="episode", columns="config", values="avg_reward").sort_index()
    piv_steps = g.pivot(index="episode", columns="config", values="avg_steps").sort_index()
    piv_success = g.pivot(index="episode", columns="config", values="avg_success").sort_index()
    return piv_reward, piv_steps, piv_success

def plot_multiline(df_metric, title, ylab, filename):
    plt.figure()
    for col in [c for c in ["Baseline","Reward Machine","Shaped"] if c in df_metric.columns]:
        plt.plot(df_metric.index, df_metric[col], label=col)
    plt.xlabel("Episode")
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, filename))
    plt.close()

def advanced_last_quarter_smoothed(df, start_ep=751, end_ep=1000, window=10):
    env_name = "Advanced"
    sub = df[df["env"]==env_name]
    results = {}
    for metric, colname in [("reward","env_reward"), ("steps","steps"), ("success","success")]:
        # per-seed smoothing
        smoothed_runs = []
        for (cfg, seed), run in sub.groupby(["config","seed"]):
            s = run.set_index("episode").sort_index()[colname]
            s_sm = s.rolling(window=window, center=True, min_periods=1).mean()
            s_sm = s_sm.loc[start_ep:end_ep]
            smoothed_runs.append((cfg, s_sm))
        # average across seeds per config for each episode (align by index)
        cfg_frames = {}
        for cfg in sub["config"].unique():
            series_list = [sr for (c, sr) in smoothed_runs if c==cfg]
            if not series_list:
                continue
            df_cfg = pd.concat(series_list, axis=1)
            df_cfg["mean"] = df_cfg.mean(axis=1)
            cfg_frames[cfg] = df_cfg["mean"]
        # combine configs
        if cfg_frames:
            combined = pd.DataFrame(cfg_frames).sort_index()
            results[metric] = combined
    return results  # dict: metric -> DataFrame with columns per config

def main():
    df = load_and_normalize(XLSX_PATH, SHEET_NAME)

    # --- Foundational environment (unsmoothed averages) ---
    f_reward, f_steps, f_succ = per_episode_averages(df, "Foundational")
    plot_multiline(f_reward, "Foundational: Average Reward vs Episode", "Average Reward",
                   "Foundational_AvgReward_vs_Episode.png")
    plot_multiline(f_steps, "Foundational: Average Steps vs Episode", "Average Steps",
                   "Foundational_AvgSteps_vs_Episode.png")
    plot_multiline(f_succ, "Foundational: Success Rate vs Episode", "Success Rate",
                   "Foundational_SuccessRate_vs_Episode.png")

    # --- Advanced environment (unsmoothed averages) ---
    a_reward, a_steps, a_succ = per_episode_averages(df, "Advanced")
    plot_multiline(a_reward, "Advanced: Average Reward vs Episode", "Average Reward",
                   "Advanced_AvgReward_vs_Episode.png")
    plot_multiline(a_steps, "Advanced: Average Steps vs Episode", "Average Steps",
                   "Advanced_AvgSteps_vs_Episode.png")
    plot_multiline(a_succ, "Advanced: Success Rate vs Episode", "Success Rate",
                   "Advanced_SuccessRate_vs_Episode.png")

    # --- Advanced last 25% smoothed (per-seed smoothing, window=10) ---
    smoothed = advanced_last_quarter_smoothed(df, start_ep=751, end_ep=1000, window=10)
    # reward
    if "reward" in smoothed:
        plot_multiline(smoothed["reward"], "Advanced (Last 25%): Smoothed Average Reward vs Episode",
                       "Smoothed Avg Reward", "Advanced_Last25_Smoothed_AvgReward.png")
    # steps
    if "steps" in smoothed:
        plot_multiline(smoothed["steps"], "Advanced (Last 25%): Smoothed Average Steps vs Episode",
                       "Smoothed Avg Steps", "Advanced_Last25_Smoothed_AvgSteps.png")
    # success
    if "success" in smoothed:
        plot_multiline(smoothed["success"], "Advanced (Last 25%): Smoothed Success Rate vs Episode",
                       "Smoothed Success Rate", "Advanced_Last25_Smoothed_SuccessRate.png")

    print(f"Saved figures to: {FIGS_DIR}")

if __name__ == "__main__":
    main()
