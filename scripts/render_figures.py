"""Render report figures from terminal evidence copied from OpenResearch logs."""

from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path("reports/trace-reproduction/images")
OUT.mkdir(parents=True, exist_ok=True)

steps = [0, 4, 8, 12]
outcome = [0.109375, 0.109375, 0.140625, 0.140625]
trace = [0.109375, 0.078125, 0.0625, 0.0625]

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), gridspec_kw={"width_ratios": [1.65, 1]})

ax = axes[0]
ax.plot(steps, [100 * x for x in outcome], marker="o", linewidth=2.5, label="Outcome-only")
ax.plot(steps, [100 * x for x in trace], marker="o", linewidth=2.5, label="TRACE")
ax.set(xlabel="RL update", ylabel="Held-out exact-match success (%)", xticks=steps, ylim=(0, 18))
ax.legend(frameon=False, loc="upper left")
ax.set_title("Qwen3-4B-Instruct bounded reconstruction")

ax = axes[1]
bars = ax.bar(["Outcome-only", "TRACE"], [0.30208200897897125, 0.2201836684598186],
              color=["#4C78A8", "#F58518"])
ax.set_ylabel("Mean between-rollout signal variance")
ax.set_title("Predeclared variance diagnostic")
ax.bar_label(bars, fmt="%.3f", padding=3)
ax.text(0.5, 0.245, "27.1% lower", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, 0.36)

fig.suptitle("TRACE lowered signal variance but not held-out success in the fast matched pair",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "scout-results.png", dpi=180, bbox_inches="tight")
fig.savefig(OUT / "scout-results.svg", bbox_inches="tight")

