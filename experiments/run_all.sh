#!/usr/bin/env bash
# Reproduce every table in results/RESULTS.md.
# Usage: experiments/run_all.sh [--quick]   (run from the repository root)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
# single-threaded inference so that per-row baselines are not dominated by thread start-up
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PYTHON:-python}
DATA=${DATA_HOME:-../data}
QUICK=${1:-}
$PY synthetic.py $QUICK
$PY real_models.py --data-home "$DATA" $QUICK
$PY dowhy_e2e.py --data-home "$DATA" $QUICK
$PY report.py
