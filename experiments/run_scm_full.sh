#!/bin/bash
# Run scm_full.py on every causal dataset and model family, PAR jobs at a time.
# Usage: [SKIP=regex] run_scm_full.sh DATA_DIR [PAR]   (SKIP defaults to year: dense PC graph, dropped)
DATA=$1; PAR=${2:-10}
cd "$(dirname "$0")"
# ChiRho (torch, pyro, chirho) lives in a separate directory next to the env, when installed
[ -d ../../chirho_pkgs ] && export PYTHONPATH=$(cd ../../chirho_pkgs && pwd)
mkdir -p ../../logs/scm_full
# largest tables first, so the long jobs do not form the tail
for ds in $(ls -S "$DATA"/*/data.parquet | xargs -n1 dirname | xargs -n1 basename | grep -v '^_' | grep -v -x -E "${SKIP:-year}"); do
  for fam in LGBM XGB HGB RF GBR; do echo "$ds $fam"; done
done | xargs -P "$PAR" -L 1 bash -c 'OMP_NUM_THREADS=1 ../../env/bin/python scm_full.py --data '"$DATA"' --dataset $0 --family $1 > ../../logs/scm_full/$0_$1.log 2>&1'
