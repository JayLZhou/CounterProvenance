#!/bin/bash
# Overnight queue on 2T, 2026-11-07: each stage starts when the previous one has ended.
W=/mnt/data/yingli/counterprovenance; export TMPDIR=$W/tmp; LOG=$W/logs/night.txt
say() { echo "$(date '+%F %T') $*" >> $LOG; }
running() { ps -u yingli -o args= | grep -v -E "^timeout|grep|bash -c" | grep -c -E "$1"; }
wait_for() { while [ "$(running "$1")" -gt 0 ]; do sleep 120; done; }
PY=$W/env/bin/python
say "queue started"

# 1. the runs of this morning
say "1. waiting for the one-group runs and the Exp.4 CellForest runs"; wait_for "mech_main.py.*(one1e8|trees_hybrid)"; say "1. done"

# 2. Exp.4 CellForest (hybrid index): the models without all results, e.g. stopped by the memory guard
say "2. Exp.4: runs that ended without results"
cd $W/repo_hybrid/experiments
for n in 100 300 3000; do for f in GBR LGBM XGB HGB RF; do for ds in concrete light_tunnel ailerons pol elevators jdk postgres baf mslr census1990 susy methane higgs airline nyc_taxi; do
  [ -f $W/repo_dev2/results/trees/$n/cache/${ds}_$f.pkl ] || continue
  [ -f $W/repo_hybrid/results/trees_hybrid/$n/${ds}_${f}_d4_twin_lat.npy ] && continue
  echo "$n $ds $f"; done; done; done > $W/night_trees_missing.txt
say "2. $(wc -l < $W/night_trees_missing.txt) to run again"
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$W/repo_hybrid:$W/chirho_pkgs xargs -a $W/night_trees_missing.txt -P 4 -L 1 bash -c 'timeout 43200 '$PY' mech_main.py --data '$W'/data/causal --dataset $1 --family $2 --backend twin --trees $0 --out trees_hybrid/$0 >> '$W'/logs/trees_hybrid/$1_$2_$0.log 2>&1'
say "2. done"

# 3. applications: CellForest with the hybrid index on every table of the paper (no feature relevance on RF200)
say "3. applications, CellForest with the hybrid index"
mkdir -p $W/logs/shard_hybrid
printf "%s\n" "german_full HGB100 dowhy_feature_relevance" "heloc HGB100 dowhy_feature_relevance" "adult HGB100 dowhy_feature_relevance" "give_credit HGB100 dowhy_feature_relevance" \
  "adult RF100 dowhy_feature_relevance" "adult RF200 mcce_1000" "adult RF200 mcce_10000" "adult RF500D10 geco_genetic" "give_credit RF100 dowhy_feature_relevance" | \
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$W/repo_hybrid:$W/shap_pkgs:$W/chirho_pkgs xargs -P 2 -L 1 bash -c 'cd '$W'/repo_hybrid/experiments && timeout 172800 '$PY' e2e_shard.py --data '$W'/data/causal --dataset $0 --family $1 --app $2 --backend twin >> '$W'/logs/shard_hybrid/$0_$1_$2.log 2>&1'
say "3. done"

# 4. the main experiment's CellForest again with the hybrid index, while the machine is quieter: clean query times
say "4. main CellForest, clean times"; cd $W; $PY run_compact.py 8 repo_hybrid hybrid_clean >> $W/logs/run_hybrid_clean.txt 2>&1
wait_for "mech_main.py.*hybrid_clean"; say "4. done"

# 5. memory without grouping (only the memory is reported; several threads): ailerons, then the six 64-bit ones
say "5. memory without grouping"
cd $W/repo_hybrid/experiments
for f in HGB LGBM; do OMP_NUM_THREADS=1 PYTHONPATH=$W/repo_hybrid timeout 86400 $PY mech_rss.py --dataset ailerons --family $f --grouping none --threads 24 >> $W/logs/none/ailerons_$f.log 2>&1 & done
cd $W; $PY run_none64.py 6 >> $W/logs/run_none64.txt 2>&1
wait_for "mech_rss.py.*grouping none"; say "5. done"
say "queue ended"
