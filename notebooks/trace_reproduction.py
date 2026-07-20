import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # TRACE turn-level credit: a bounded reproduction

    This notebook is a self-contained, tutorial-style companion to the
    reproduction of **TRACE: Turn-level Reward Assignment via Credit
    Estimation for Long-Horizon Agents** (arXiv:2607.13988). It opens with
    the measured evidence; no model download or expensive rerun is needed.

    The paper asks whether a frozen reference model can turn changes in the
    probability of the gold answer at tool-call boundaries into useful dense
    rewards. We tested that claim against a matched outcome-only control on a
    deterministic, disjoint slice of the public BrowseComp-Plus release.
    """)
    return


@app.cell
def _():
    steps = [0, 4, 8, 12]
    scout = {
        "Outcome-only": [0.109375, 0.109375, 0.140625, 0.140625],
        "TRACE": [0.109375, 0.078125, 0.0625, 0.0625],
    }
    variances = {"Outcome-only": 0.30208200897897125, "TRACE": 0.2201836684598186}
    return scout, steps, variances


@app.cell
def _(mo, scout, steps, variances):
    rows = []
    for method, curve in scout.items():
        for step, success in zip(steps, curve):
            rows.append({"method": method, "update": step, "success_percent": 100 * success})
    chart = mo.ui.altair_chart(
        __import__("altair").Chart(__import__("pandas").DataFrame(rows))
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=__import__("altair").X("update:Q", title="RL update"),
            y=__import__("altair").Y("success_percent:Q", title="Held-out exact-match success (%)", scale=__import__("altair").Scale(domain=[0, 18])),
            color=__import__("altair").Color("method:N", title="Method"),
            tooltip=["method", "update", "success_percent"],
        )
        .properties(height=320, title="Fast matched Qwen3-4B-Instruct pair (8 independent GPU replicas)")
    )
    variance_drop = 100 * (1 - variances["TRACE"] / variances["Outcome-only"])
    mo.vstack([
        chart,
        mo.callout(
            mo.md(
                f"""**Observed:** final success was **6.25% for TRACE** and
                **14.06% for outcome-only**. TRACE's predeclared learning-signal
                variance was **{variance_drop:.1f}% lower** (0.220 vs 0.302)."""
            ),
            kind="neutral",
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From answer likelihood to turn credit

    At each tool boundary (s_t), the frozen reference model scores the
    gold answer (y^*) by its mean token log-probability:

    \[
    \ell_t = \frac{1}{|y^*|}\log p_{\mathrm{ref}}(y^*\mid s_t),
    \qquad g_t=-\ell_t+\epsilon.
    \]

    A tool call receives a log-ratio temporal difference
    (\delta_t=\log(g_t/g_{t+1})). Positive values mean the next state made
    the gold answer easier for the reference model to predict. TRACE averages
    the next (K=3) differences with discount (\gamma=0.8), then combines
    the result with the group-relative outcome advantage.

    This is *critic-free*: the frozen language model supplies the state value,
    while the policy is updated through LoRA. Our implementation scores only
    tool-call boundaries and disables the LoRA adapter during reference
    scoring, so the reward model remains fixed.
    """)
    return


@app.cell
def _(mo):
    epsilon = mo.ui.slider(0.01, 0.5, value=0.1, step=0.01, label="epsilon")
    before = mo.ui.slider(-8.0, -0.5, value=-4.0, step=0.1, label="log p before")
    after = mo.ui.slider(-8.0, -0.5, value=-3.0, step=0.1, label="log p after")
    mo.md("## Explore one boundary reward\n\nMove the controls to see how a gold-answer likelihood change becomes a TRACE TD reward.")
    mo.hstack([epsilon, before, after], justify="start")
    return after, before, epsilon


@app.cell
def _(after, before, epsilon, mo):
    import math
    delta = math.log((-before.value + epsilon.value) / (-after.value + epsilon.value))
    direction = "helpful" if delta > 0 else "harmful" if delta < 0 else "neutral"
    mo.callout(
        mo.md(f"The boundary reward is **{delta:.3f}**: the tool transition looks **{direction}** to the reference model."),
        kind="success" if delta > 0 else "warn",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What was matched, and what was downscaled

    Both methods used the same public data split, eight GPU replicas, rollout
    seeds, LoRA policy, four rollouts per prompt, optimizer, 12 updates, and
    held-out exact-match evaluator. Only the dense TRACE term changed.

    The bounded setup is much smaller than the paper: 16 training and 8 held-out
    questions rather than the full environment, 4 tool turns rather than 60,
    and 12 updates rather than 200. Retrieval used each question's released
    gold documents plus up to 32 released hard negatives, not the paper's full
    closed-web index. The update is group-relative REINFORCE rather than a
    complete clipped GRPO implementation. The report therefore assesses the
    tested setup, not the truth of the full-scale claim.

    ## Evidence boundary

    The paper reports **35.6% TRACE vs 30.0% GRPO** on its controlled
    Qwen3-4B BrowseComp-Plus result. The fast matched reconstruction showed
    **6.25% vs 14.06%** at update 12, so this bounded run did not show the
    reported success advantage or earlier gains. It did show the expected
    variance direction: **0.220 vs 0.302**, 27.1% lower.

    All formal measurements came from OpenResearch Kubernetes runs on
    NVIDIA RTX PRO 6000 Blackwell GPUs. See the public report for exact-model
    results, complete provenance, and the final claim-by-claim assessment.
    """)
    return


if __name__ == "__main__":
    app.run()
