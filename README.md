# TRACE claim reproduction

This repository reconstructs the turn-level reward mechanism from
**TRACE: Turn-level Reward Assignment via Credit Estimation for Long-Horizon
Agents** (arXiv:2607.13988). The experiment code uses the public
Qwen3-4B-Thinking-2507 checkpoint and an offline slice of the official
BrowseComp-Plus release. Reader-facing results will be added after the controlled
Kubernetes runs finish.

The fixed experiment entrypoint is:

```bash
bash run.sh
```

Experiment branches change only `config.json`; the baseline and both RL methods
therefore execute the same command.
