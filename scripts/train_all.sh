#!/usr/bin/env bash
set -euo pipefail

python scripts/run_experiments.py --mode train "$@"
