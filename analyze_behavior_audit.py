# analyze_reward_semantics.py
# Single-file analysis based purely on environment reward semantics.
# Reads:
#   behavior_audit/Shaped_per_step.csv
#   behavior_audit/RM_per_step.csv
# Writes (into behavior_audit/analysis_from_rewards/):
#   - action_distribution_summary.csv / .tex
#   - behavior_summary_from_rewards.csv / .tex
#   - RM_episode_agg_from_rewards.csv
#   - Shaped_episode_agg_from_rewards.csv
#   - compromise_breakdown_means.csv
#   - figs/*.png

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

# -----------------------------------
# Config
# -----------------------------------
IN_DIR = "behavior_audit"
IN_SH = os.path.join(IN_DIR, "Shaped_per_step.csv")
IN_RM = os.path.join(IN_DIR, "RM_per_step.csv")

OUT_DIR = os.path.join(IN_DIR, "analysis_from_rewards")
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Reward semantics (float-safe)
STEP_COST = -0.1               # scan/idle
GOOD_COMP = 9.9                # successful compromise (10 - 0.1 step cost)
BAD_COMP  = -10.1              # wrong compromise (healthy target)
USE_EPS = True
EPS = 1e-6

REQUIRED_COLS = {"seed", "episode", "t", "env_reward_step"}

ACTION_LABELS = [
    "Scan/Idle (-0.1)",
    "Correct Compromise (+9.9)",
    "Wrong Compromise (-10.1)",
    "Other",
]

# -----------------------------------
# Helpers
# -----------------------------------
def check_exists(path: str):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Missing file: {path}")

def read_per_step(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] {path} is missing columns: {missing}\n"
                 f"Columns present: {list(df.columns)}")
    # Robust type coercion
    for col in ["seed", "episode", "t"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["env_reward_step"] = pd.to_numeric(df["env_reward_step"], errors="coerce")
    bad = df["env_reward_step"].isna().sum()
    if bad:
        print(f"[WARN] {bad} rows with non-numeric env_reward_step will be dropped.")
        df = df.dropna(subset=["env_reward_step"])
    df = df.dropna(subset=["seed", "episode", "t"])
    df["seed"] = df["seed"].astype(int)
    df["episode"] = df["episode"].astype(int)
    df["t"] = df["t"].astype(int)
    return df

def close_enough(x: float, target: float, eps: float) -> bool:
    return abs(x - target) < eps

def classify_reward(r: float) -> str:
    if USE_EPS:
        if close_enough(r, STEP_COST, EPS): return "Scan/Idle (-0.1)"
        if close_enough(r, GOOD_COMP, EPS): return "Correct Compromise (+9.9)"
        if close_enough(r, BAD_COMP,  EPS): return "Wrong Compromise (-10.1)"
    else:
        if r == STEP_COST: return "Scan/Idle (-0.1)"
        if r == GOOD_COMP: return "Correct Compromise (+9.9)"
        if r == BAD_COMP:  return "Wrong Compromise (-10.1)"
    return "Other"

def summarize_action_distribution(df: pd.DataFrame, method_label: str) -> pd.DataFrame:
    """
    Whole-dataset counts/percentages by action type.
    Always returns rows for all ACTION_LABELS (zeros if absent).
    """
    vc = df["action_type"].value_counts()
    counts = pd.Series(0, index=ACTION_LABELS, dtype=float)
    for k, v in vc.items():
        if k in counts.index:
            counts.loc[k] = v
        else:
            counts.loc["Other"] += v
    total = counts.sum()
    perc = (counts / total * 100.0).round(3) if total > 0 else pd.Series(0.0, index=counts.index)
    out = pd.DataFrame({
        "Method": method_label,
        "Action": counts.index,
        "Count": counts.values.astype(int),
        "Percent": perc.values,
    })
    return out

def per_episode_metrics(df: pd.DataFrame, method_label: str) -> pd.DataFrame:
    """
    Per (seed, episode) aggregates:
      steps (t max), scans, correct/wrong compromises, env_reward_sum,
      scan_to_comp_ratio, wrong_compromise_rate
    """
    df["is_scan"]  = (df["action_type"] == "Scan/Idle (-0.1)").astype(int)
    df["is_ok"]    = (df["action_type"] == "Correct Compromise (+9.9)").astype(int)
    df["is_wrong"] = (df["action_type"] == "Wrong Compromise (-10.1)").astype(int)

    agg = df.groupby(["seed","episode"], as_index=False).agg(
        steps=("t","max"),
        scans=("is_scan","sum"),
        correct_compromises=("is_ok","sum"),
        wrong_compromises=("is_wrong","sum"),
        env_reward_sum=("env_reward_step","sum"),
    )
    total_comp = agg["correct_compromises"] + agg["wrong_compromises"]
    agg["scan_to_comp_ratio"] = np.where(total_comp > 0, agg["scans"] / total_comp, np.nan)
    agg["wrong_compromise_rate"] = np.where(total_comp > 0, agg["wrong_compromises"] / total_comp, np.nan)
    agg["Method"] = method_label
    return agg

def agg_summary(agg: pd.DataFrame, method_label: str) -> pd.DataFrame:
    total_comp = agg["correct_compromises"] + agg["wrong_compromises"]
    out = {
        "Method": method_label,
        "Mean steps": agg["steps"].mean(),
        "Mean scans": agg["scans"].mean(),
        "Mean compromises": total_comp.mean(),
        "Wrong-compromise rate (episodes w/comp)": agg["wrong_compromise_rate"].mean(skipna=True),
        "Scan-to-comp ratio": agg["scan_to_comp_ratio"].mean(skipna=True),
        "Mean env reward": agg["env_reward_sum"].mean(),
        "Mean correct compromises": agg["correct_compromises"].mean(),
        "Mean wrong compromises": agg["wrong_compromises"].mean(),
        "Episodes counted": len(agg),
    }
    return pd.DataFrame([out])

def save_latex_table(df: pd.DataFrame, path: str, caption: str, label: str):
    df2 = df.copy().fillna("")
    cols = list(df2.columns)
    colspec = "l" + "c"*(len(cols)-1)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df2.iterrows():
        lines.append(" & ".join(str(row[c]) for c in cols) + r" \\")
    lines += [
        r"\bottomrule",
        r"\\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}"
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# -----------------------------------
# Main
# -----------------------------------
def main():
    check_exists(IN_SH); check_exists(IN_RM)

    shaped = read_per_step(IN_SH)
    rm     = read_per_step(IN_RM)

    shaped["action_type"] = shaped["env_reward_step"].apply(classify_reward)
    rm["action_type"]     = rm["env_reward_step"].apply(classify_reward)

    # 1) Whole-dataset action distributions
    dist_sh = summarize_action_distribution(shaped, "Shaped DQN")
    dist_rm = summarize_action_distribution(rm,     "Reward Machine DQN")
    dist_all = pd.concat([dist_rm, dist_sh], ignore_index=True)

    out_csv = os.path.join(OUT_DIR, "action_distribution_summary.csv")
    dist_all.to_csv(out_csv, index=False)
    save_latex_table(
        dist_all,
        os.path.join(OUT_DIR, "action_distribution_summary.tex"),
        caption=("Action-type distribution derived from environment reward semantics. "
                 "Scan/Idle corresponds to -0.1 per step; Correct Compromise to +9.9; "
                 "Wrong Compromise to -10.1."),
        label="tab:action-dist"
    )

    # 2) Per-episode aggregates
    agg_rm = per_episode_metrics(rm, "Reward Machine DQN")
    agg_sh = per_episode_metrics(shaped, "Shaped DQN")

    agg_rm.to_csv(os.path.join(OUT_DIR, "RM_episode_agg_from_rewards.csv"), index=False)
    agg_sh.to_csv(os.path.join(OUT_DIR, "Shaped_episode_agg_from_rewards.csv"), index=False)

    # 3) High-level summary
    sum_rm = agg_summary(agg_rm, "Reward Machine DQN")
    sum_sh = agg_summary(agg_sh, "Shaped DQN")
    summary = pd.concat([sum_rm, sum_sh], ignore_index=True)

    summary.to_csv(os.path.join(OUT_DIR, "behavior_summary_from_rewards.csv"), index=False)
    save_latex_table(
        summary.round(3),
        os.path.join(OUT_DIR, "behavior_summary_from_rewards.tex"),
        caption=("Summary of per-episode behavior metrics derived purely from reward semantics. "
                 "Wrong-compromise rate and Scan-to-comp ratio are computed over episodes that contained at least one compromise attempt."),
        label="tab:behavior-summary-reward-based"
    )

    # 4) Figures
    # Action distribution (%)
    pivot = (dist_all.pivot_table(index="Action", columns="Method", values="Percent", aggfunc="sum")
             .reindex(ACTION_LABELS).fillna(0.0))
    ax = pivot.plot(kind="bar", figsize=(9,5))
    ax.set_ylabel("Percentage of steps (%)")
    ax.set_title("Action Distribution by Method (from rewards)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "action_distribution_by_method.png"))
    plt.close()

    # Steps vs env reward (per-episode)
    plt.figure(figsize=(8,5))
    plt.scatter(agg_rm["steps"], agg_rm["env_reward_sum"], alpha=0.35, label="RM")
    plt.scatter(agg_sh["steps"], agg_sh["env_reward_sum"], alpha=0.35, label="Shaped")
    plt.xlabel("Episode steps")
    plt.ylabel("Env reward (per episode)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "steps_vs_reward_scatter_from_rewards.png"))
    plt.close()

    # Mean correct vs wrong compromises per episode (bar)
    bar_df = pd.DataFrame({
        "Method": ["Reward Machine DQN", "Shaped DQN"],
        "Mean correct compromises": [agg_rm["correct_compromises"].mean(), agg_sh["correct_compromises"].mean()],
        "Mean wrong compromises":   [agg_rm["wrong_compromises"].mean(),   agg_sh["wrong_compromises"].mean()],
    })
    bar_df.to_csv(os.path.join(OUT_DIR, "compromise_breakdown_means.csv"), index=False)

    x = np.arange(len(bar_df["Method"]))
    width = 0.35
    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, bar_df["Mean correct compromises"], width, label="Correct")
    plt.bar(x + width/2, bar_df["Mean wrong compromises"],   width, label="Wrong")
    plt.xticks(x, bar_df["Method"], rotation=0)
    plt.ylabel("Mean per episode")
    plt.title("Correct vs Wrong Compromises (per episode)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "compromise_breakdown_means.png"))
    plt.close()

    print("[OK] Wrote:")
    for f in [
        "action_distribution_summary.csv",
        "behavior_summary_from_rewards.csv",
        "compromise_breakdown_means.csv",
        "action_distribution_summary.tex",
        "behavior_summary_from_rewards.tex",
        "RM_episode_agg_from_rewards.csv",
        "Shaped_episode_agg_from_rewards.csv",
        "figs/action_distribution_by_method.png",
        "figs/steps_vs_reward_scatter_from_rewards.png",
        "figs/compromise_breakdown_means.png",
    ]:
        print("   -", os.path.join(OUT_DIR, f))

if __name__ == "__main__":
    main()
