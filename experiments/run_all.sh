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
$PY dowhy_e2e.py --data-home "$DATA" --dims 6,8 --suffix _d68 $QUICK
for ds in topo_2_1:422 yprop_4_1:416; do
  for m in sklearn-GBR sklearn-HGB XGBoost-hist LightGBM; do
    $PY highdim.py --dataset "$ds" --model "$m" --data-home "$DATA"
  done
done
$PY report.py
