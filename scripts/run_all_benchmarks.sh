#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$(pwd)"

.venv/bin/python -m sweep --sweep configs/experiments/benchmark_matrix.yaml
