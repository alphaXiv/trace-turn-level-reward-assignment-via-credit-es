# TRACE turn-level reward reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/trace-turn-level-reward-assignment-via-credit-es/blob/main/notebooks/trace_reproduction.py)

This repository is a bounded, public reproduction of the central claim in
**[TRACE: Turn-level Reward Assignment via Credit Estimation for Long-Horizon
Agents](https://arxiv.org/abs/2607.13988)**: reference-model gold-answer
log-ratio TD rewards should improve a Qwen3-class closed-web search agent over
matched outcome-only RL, with earlier gains and lower reward variance.

**Assessment: partially reproduced.** The completed fast matched pair did not
show the paper's success advantage: the paper reports **35.6% TRACE versus 30.0%
GRPO (+5.6 pp)**, while our Qwen3-4B-Instruct reconstruction finished at **6.25%
TRACE versus 14.06% outcome-only (−7.81 pp)**. It also did not show earlier
success gains. It did show the expected variance direction: TRACE's predeclared
between-rollout learning-signal variance was **0.220 versus 0.302**, or **27.1%
lower**. An exact Qwen3-4B-Thinking matched pair was then run to test the model
substitution directly; its terminal results are reported in the detailed report.

The experiment is intentionally smaller than the paper: 16 training and 8
held-out questions from one official BrowseComp-Plus shard, four tool turns,
four rollouts per prompt, 12 RL updates, a query-local public offline corpus,
LoRA with a 2,048-token differentiated suffix, and group-relative REINFORCE
rather than the reported full index, 60 turns, eight rollouts, 200 updates, and
clipped GRPO. Rollout generation and reference scoring retain a 4,096-token
budget. All formal runs used
**OpenResearch Kubernetes** on **NVIDIA RTX PRO 6000 Blackwell** GPUs, reaching
**16 concurrent GPUs**.

- [Read the detailed claim-by-claim report](reports/trace-reproduction/report.md)
- [Open the self-contained tutorial notebook in Molab](https://molab.marimo.io/github/alphaXiv/trace-turn-level-reward-assignment-via-credit-es/blob/main/notebooks/trace_reproduction.py)
- Run locally with `marimo edit notebooks/trace_reproduction.py` or
  `marimo run notebooks/trace_reproduction.py`

## Experiment log

The command column is copied verbatim from `orx exp status`. Every formal node
used the same entrypoint; only committed code/config differs between branches.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public report, notebook, figures, and reference implementation | Not run as an experiment (publication surface) | Presentation-only | No experiment allocation |
| [Qwen3-4B-Instruct tool-agent baseline](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es/tree/orx/qwen3-4b-instruct-tool-agent) | Establish functional native tool protocol | `bash run.sh` | 10.94% held-out success; 2.52 tool calls | Kubernetes, 8× Blackwell, 88 s model time |
| [Outcome-only RL](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es/tree/orx/outcome-only-rl) | Matched terminal-reward control | `bash run.sh` | 14.06% final success; variance 0.3021 | Kubernetes, 8× Blackwell, 1,027 s model time |
| [TRACE RL](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es/tree/orx/trace-rl) | Add frozen-reference log-ratio TD rewards | `bash run.sh` | 6.25% final success; variance 0.2202 | Kubernetes, 8× Blackwell, 1,515 s model time |
| [Exact Qwen3-4B-Thinking baseline](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es/tree/orx/paper-length-thinking-baseline) | Restore the paper checkpoint and 4,096-token actions | `bash run.sh` | 10.94% held-out success | Kubernetes, 8× Blackwell, 1,210 s model time |
| [Memory-safe exact outcome-only RL](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es/tree/orx/memory-safe-exact-outcome-only) | Exact-model control; 2,048-token differentiated suffix | `bash run.sh` | See report | Kubernetes, 8× Blackwell |
| [Memory-safe exact TRACE RL](https://github.com/alphaXiv/trace-turn-level-reward-assignment-via-credit-es/tree/orx/memory-safe-exact-trace) | Exact model plus TRACE; matched memory change | `bash run.sh` | See report | Kubernetes, 8× Blackwell |

## Reproduce the implementation

The fixed experiment entrypoint is:

```bash
bash run.sh
```

`trace_repro.py` performs the complete run: public data acquisition and
decryption, deterministic split selection, offline browser construction,
tool-agent rollout, outcome-only or TRACE training, held-out evaluation, and
terminal JSON evidence. `config.json` contains every scientific setting. The
Kubernetes manifest launches eight independent one-GPU replicas in one job.

Formal results should be reproduced through an orchestrated experiment branch;
the notebook intentionally embeds the completed evidence and does not rerun the
4B model.

## Repository map

- `trace_repro.py` — agent, browser, TRACE credit, training, and aggregation
- `config.json` — committed baseline configuration
- `.orx/k8s.yaml` — 8-GPU Kubernetes job contract
- `reports/trace-reproduction/report.md` — detailed public report
- `reports/trace-reproduction/images/` — figures rendered from terminal evidence
- `notebooks/trace_reproduction.py` — self-contained marimo tutorial
- `autoresearch.json` — machine-readable result and measured compute
