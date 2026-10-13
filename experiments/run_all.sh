#!/usr/bin/env bash
# Reproduce every table in results/RESULTS.md.
# Usage: experiments/run_all.sh [--quick]   (run from the repository root)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
PY=${PYTHON:-python}
DATA=${DATA_HOME:-../data}
QUICK=${1:-}
$PY synthetic.py $QUICK
$PY real_models.py --data-home "$DATA" $QUICK
$PY dowhy_e2e.py --data-home "$DATA" $QUICK
$PY report.py
