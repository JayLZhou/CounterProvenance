#!/bin/bash
# Run scm_full.py on every causal dataset and model family, PAR jobs at a time.
# Usage: run_scm_full.sh DATA_DIR [PAR]
DATA=$1; PAR=${2:-10}
cd "$(dirname "$0")"
mkdir -p ../../logs/scm_full
# largest tables first, so the long jobs do not form the tail
for ds in $(ls -S "$DATA"/*/data.parquet | xargs -n1 dirname | xargs -n1 basename | grep -v '^_'); do
  for fam in LGBM XGB HGB RF GBR; do echo "$ds $fam"; done
done | xargs -P "$PAR" -L 1 bash -c 'OMP_NUM_THREADS=1 ../../env/bin/python scm_full.py --data '"$DATA"' --dataset $0 --family $1 > ../../logs/scm_full/$0_$1.log 2>&1'
