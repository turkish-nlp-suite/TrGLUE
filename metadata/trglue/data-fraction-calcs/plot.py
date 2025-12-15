import os
import json
import math
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Config: full-data metrics
# Fill from your final dev scores (choose a single metric per task)
# For pairs like "87.4/91.3", we’ll take the second as your "best" unless you change below.
full_data_metrics = {
    # task: (metric_name, value)
    "cola": ("matthews_correlation", 42.0),       # CoLA 42 (already %ish; leave as is)
    "sst2": ("accuracy", 91.3),                   # SST-2 87.4/91.3 -> using 91.3
    "mrpc": ("f1", 72.7),                         # MRPC 74.4/72.7 -> using F1=72.7
    "stsb": ("spearmanr", 69.9),                  # STS-B 71.3/69.9 -> using Spearman=69.9
    "qqp":  ("f1", 94.3),                         # QQP 95.4/94.3 -> using F1=94.3
    "mnli": ("accuracy", 90.8),                   # MNLI 87.9/90.8 -> using 90.8 (pick matched or your chosen)
    "qnli": ("accuracy", 90.6),                   # QNLI 90.6
    "rte":  ("accuracy", 92.2),                   # RTE 92.2
}

# If you want to flip which one to use, just change the value above.

# -------------------------
# Optional: record the training hyperparameters you used
hyperparams = {
    "cola": {"lr": 2e-5, "batch_train": 128, "batch_eval": 128, "epochs": 3},
    "sst2": {"lr": 2e-5, "batch_train": 128, "batch_eval": 128, "epochs": 3},
    "qqp":  {"lr": 2e-5, "batch_train": 128, "batch_eval": 128, "epochs": 3},
    "mnli": {"lr": 2e-5, "batch_train": 128, "batch_eval": 128, "epochs": 3},
    "qnli": {"lr": 2e-5, "batch_train": 128, "batch_eval": 128, "epochs": 3},
    "stsb": {"lr": 3e-5, "batch_train": 16,  "batch_eval": 16,  "epochs": 3},
    "rte":  {"lr": 3e-5, "batch_train": 16,  "batch_eval": 16,  "epochs": 3},
    "mrpc": {"lr": 3e-5, "batch_train": 16,  "batch_eval": 16,  "epochs": 3},
}

# -------------------------
# Where to read subsample results and where to write outputs
results_dir = Path("results")          # put CSVs like results/cola.csv, etc.
fig_dir = Path("figures"); fig_dir.mkdir(parents=True, exist_ok=True)
out_csv = fig_dir / "learning_curves_fractional_points.csv"
out_fig = fig_dir / "learning_curves_fractional.png"
out_hparams = fig_dir / "training_hyperparameters.csv"

# -------------------------
# Helper to load optional per-task subsample results
# Expected CSV columns: frac, metric, seed (optional)
def load_task_subsamples(task: str):
    f = results_dir / f"{task}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    # Basic validation
    need_cols = {"frac", "metric"}
    if not need_cols.issubset(set(df.columns)):
        raise ValueError(f"{f} must contain columns: {need_cols}")
    # Clip frac to [0, 1]
    df["frac"] = df["frac"].clip(0.0, 1.0)
    return df

# -------------------------
# Build a tidy DataFrame of points to plot (normalized)
rows = []
for task, (metric_name, full_val) in full_data_metrics.items():
    full_val = float(full_val)
    subs = load_task_subsamples(task)
    if subs is None or len(subs) == 0:
        # Only full-data point at x=1.0, y=1.0
        rows.append({
            "task": task.upper(),
            "metric_name": metric_name,
            "frac": 1.0,
            "metric_raw": full_val,
            "metric_norm": 1.0,
            "seed": np.nan,
            "is_full": True,
        })
    else:
        # Normalize each row by the full-data value
        for _, r in subs.iterrows():
            y_norm = r["metric"] / full_val if full_val != 0 else np.nan
            rows.append({
                "task": task.upper(),
                "metric_name": metric_name,
                "frac": float(r["frac"]),
                "metric_raw": float(r["metric"]),
                "metric_norm": float(y_norm),
                "seed": r["seed"] if "seed" in subs.columns else np.nan,
                "is_full": False,
            })
        # Ensure we also have the full-data anchor at (1,1)
        rows.append({
            "task": task.upper(),
            "metric_name": metric_name,
            "frac": 1.0,
            "metric_raw": full_val,
            "metric_norm": 1.0,
            "seed": np.nan,
            "is_full": True,
        })

df = pd.DataFrame(rows)

# Save tidy CSV of points
df.to_csv(out_csv, index=False)

# Save hyperparameters table
with open(out_hparams, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["task","lr","batch_train","batch_eval","epochs"])
    for t, hp in hyperparams.items():
        writer.writerow([t.upper(), hp["lr"], hp["batch_train"], hp["batch_eval"], hp["epochs"]])

# -------------------------
# Plot: single panel with fractions (like your example)
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=200)

# Color map per task
palette = {
    "QQP": "#D62728",   # red
    "QNLI": "#2CA02C",  # green
    "MNLI": "#1F77B4",  # blue
    "SST2": "#FF7F0E",  # orange
    "MRPC": "#9467BD",  # purple
    "STSB": "#8C564B",  # brown
    "COLA": "#17BECF",  # teal
    "RTE":  "#7F7F7F",  # gray
}

# Aggregate by frac for mean ± SE across seeds
plot_tasks = sorted(df["task"].unique(), key=lambda x: ["QQP","QNLI","MNLI","SST2","MRPC","STSB","COLA","RTE"].index(x) if x in ["QQP","QNLI","MNLI","SST2","MRPC","STSB","COLA","RTE"] else 999)

for task in plot_tasks:
    d = df[df["task"] == task]
    # group by frac
    g = d.groupby("frac", as_index=False).agg(metric_norm_mean=("metric_norm","mean"),
                                              metric_norm_se=("metric_norm", lambda x: x.std(ddof=1)/np.sqrt(max(len(x),1))))
    g = g.sort_values("frac")
    color = palette.get(task, None)
    ax.plot(g["frac"], g["metric_norm_mean"], label=task, color=color, linewidth=2.0, marker="o", markersize=4)
    # light error band
    se = g["metric_norm_se"].fillna(0.0).values
    ax.fill_between(g["frac"], g["metric_norm_mean"]-se, g["metric_norm_mean"]+se,
                    color=color, alpha=0.12, linewidth=0)

# Axes in fractions
ax.set_xlim(0.35, 1.02)  # tweak if your smallest frac is ~0.4 like the old plot
ax.set_ylim(0.65, 1.03)

ax.set_xlabel("Fraction of training data")
ax.set_ylabel("Fraction of full-data score")

# Light grid, ticks at nice fractions
ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticks([0.7, 0.8, 0.9, 1.0])

# Legend outside right to reduce overlap
ax.legend(title="Task", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)

fig.tight_layout()
fig.savefig(out_fig, bbox_inches="tight")
print(f"Wrote: {out_fig}")
print(f"Wrote points CSV: {out_csv}")
print(f"Wrote hyperparams CSV: {out_hparams}")
