"""Render report figures from terminal evidence copied from OpenResearch logs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path("reports/trace-reproduction/images")
OUT.mkdir(parents=True, exist_ok=True)

evidence = json.loads(Path("results/evidence.json").read_text())
pair = evidence["fast_matched_pair"]
steps = [point["step"] for point in pair["outcome_only"]["curve"]]
outcome = [point["success_mean"] for point in pair["outcome_only"]["curve"]]
trace = [point["success_mean"] for point in pair["trace"]["curve"]]
outcome_variance = pair["outcome_only"]["mean_between_rollout_learning_signal_variance"]
trace_variance = pair["trace"]["mean_between_rollout_learning_signal_variance"]

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), gridspec_kw={"width_ratios": [1.65, 1]})

ax = axes[0]
ax.plot(steps, [100 * x for x in outcome], marker="o", linewidth=2.5, label="Outcome-only")
ax.plot(steps, [100 * x for x in trace], marker="o", linewidth=2.5, label="TRACE")
ax.set(xlabel="RL update", ylabel="Held-out exact-match success (%)", xticks=steps, ylim=(0, 18))
ax.legend(frameon=False, loc="upper left")
ax.set_title("Qwen3-4B-Instruct bounded reconstruction")

ax = axes[1]
bars = ax.bar(["Outcome-only", "TRACE"], [outcome_variance, trace_variance],
              color=["#4C78A8", "#F58518"])
ax.set_ylabel("Mean between-rollout signal variance")
ax.set_title("Predeclared variance diagnostic")
ax.bar_label(bars, fmt="%.3f", padding=3)
drop = 100 * (1 - trace_variance / outcome_variance)
ax.text(0.5, 0.245, f"{drop:.1f}% lower", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, 0.36)

fig.suptitle("TRACE lowered signal variance but not held-out success in the fast matched pair",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "scout-results.png", dpi=180, bbox_inches="tight")
fig.savefig(OUT / "scout-results.svg", bbox_inches="tight")
