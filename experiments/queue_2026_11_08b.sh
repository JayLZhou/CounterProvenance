#!/bin/bash
# Queue of 2026-11-08 (2): Exp.4 CellForest with one group while grid x trees <= 5e10, then memory without grouping.
W=/mnt/data/yingli/counterprovenance; export TMPDIR=$W/tmp; LOG=$W/logs/queue3.txt
say() { echo "$(date '+%F %T') $*" >> $LOG; }
running() { ps -u yingli -o args= | grep -v -E "^timeout|grep|bash -c" | grep -c -E "$1"; }
wait_for() { while [ "$(running "$1")" -gt 0 ]; do sleep 120; done; }
PY=$W/env/bin/python
say "B. Exp.4 CellForest, one group while grid x trees <= 5e10"
cd $W/repo_hybrid/experiments; mkdir -p $W/logs/trees_pairs
for n in 100 300 3000; do mkdir -p $W/repo_hybrid/results/trees_pairs/$n; ln -sfn $W/repo_dev2/results/trees/$n/cache $W/repo_hybrid/results/trees_pairs/$n/cache; done
for n in 100 300 3000; do for f in GBR LGBM XGB HGB RF; do for ds in concrete light_tunnel ailerons pol elevators jdk postgres baf mslr census1990 susy methane higgs airline nyc_taxi; do
  [ -f $W/repo_dev2/results/trees/$n/cache/${ds}_$f.pkl ] && echo "$n $ds $f"; done; done; done | \
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$W/repo_hybrid:$W/chirho_pkgs xargs -P 6 -L 1 bash -c 'timeout 43200 '$PY' mech_main.py --data '$W'/data/causal --dataset $1 --family $2 --backend twin --trees $0 --out trees_pairs/$0 >> '$W'/logs/trees_pairs/$1_$2_$0.log 2>&1'
say "B. done"
say "C. memory without grouping"
cd $W/repo_hybrid/experiments
for f in HGB LGBM; do OMP_NUM_THREADS=1 PYTHONPATH=$W/repo_hybrid timeout 86400 $PY mech_rss.py --dataset ailerons --family $f --grouping none --threads 24 >> $W/logs/none/ailerons_$f.log 2>&1 & done
cd $W; $PY run_none64.py 6 >> $W/logs/run_none64.txt 2>&1
wait_for "mech_rss.py.*grouping none"; say "C. done"
say "queue3 ended"
