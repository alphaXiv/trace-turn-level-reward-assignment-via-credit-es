"""Render report figures from terminal evidence copied from OpenResearch logs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path("reports/trace-reproduction/images")
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
evidence = json.loads(Path("results/evidence.json").read_text())


def render_pair(pair, output_stem, model_label, headline):
    steps = [point["step"] for point in pair["outcome_only"]["curve"]]
    outcome = [point["success_mean"] for point in pair["outcome_only"]["curve"]]
    trace = [point["success_mean"] for point in pair["trace"]["curve"]]
    outcome_variance = pair["outcome_only"]["mean_between_rollout_learning_signal_variance"]
    trace_variance = pair["trace"]["mean_between_rollout_learning_signal_variance"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), gridspec_kw={"width_ratios": [1.65, 1]})

    ax = axes[0]
    ax.plot(steps, [100 * x for x in outcome], marker="o", linewidth=2.5, label="Outcome-only")
    ax.plot(steps, [100 * x for x in trace], marker="o", linewidth=2.5, label="TRACE")
    upper = max(18, 100 * max(outcome + trace) + 4)
    ax.set(xlabel="RL update", ylabel="Held-out exact-match success (%)", xticks=steps, ylim=(0, upper))
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(model_label)

    ax = axes[1]
    bars = ax.bar(["Outcome-only", "TRACE"], [outcome_variance, trace_variance],
                  color=["#4C78A8", "#F58518"])
    ax.set_ylabel("Mean between-rollout signal variance")
    ax.set_title("Predeclared variance diagnostic")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    change = 100 * (trace_variance / outcome_variance - 1)
    direction = "lower" if change < 0 else "higher"
    ax.text(0.5, max(outcome_variance, trace_variance) * 0.78,
            f"{abs(change):.1f}% {direction}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(outcome_variance, trace_variance) * 1.2)

    fig.suptitle(headline, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"{output_stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{output_stem}.svg", bbox_inches="tight")
    plt.close(fig)


render_pair(
    evidence["fast_matched_pair"],
    "scout-results",
    "Qwen3-4B-Instruct bounded reconstruction",
    "TRACE lowered signal variance but not held-out success in the fast matched pair",
)

if "exact_matched_pair" in evidence:
    render_pair(
        evidence["exact_matched_pair"],
        "exact-results",
        "Qwen3-4B-Thinking-2507 bounded reconstruction",
        "Exact-checkpoint matched result on the held-out closed-web slice",
    )
