#!/usr/bin/env bash
set -euo pipefail

python -m pip install --disable-pip-version-check -q -r requirements.txt
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=8 trace_repro.py
