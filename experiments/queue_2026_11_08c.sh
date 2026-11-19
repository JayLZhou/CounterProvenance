#!/bin/bash
# Queue of 2026-11-08 (3): Exp.4 CellForest with one group while grid <= min(5e7, 5e10 / trees), the real memory of the final index,
# then memory without grouping.
W=/mnt/data/yingli/counterprovenance; export TMPDIR=$W/tmp; LOG=$W/logs/queue4.txt
say() { echo "$(date '+%F %T') $*" >> $LOG; }
PY=$W/env/bin/python
say "B. Exp.4 CellForest, one group while grid <= min(5e7, 5e10 / trees)"
cd $W/repo_hybrid/experiments
for n in 100 300 3000; do for f in GBR LGBM XGB HGB RF; do for ds in concrete light_tunnel ailerons pol elevators jdk postgres baf mslr census1990 susy methane higgs airline nyc_taxi; do
  [ -f $W/repo_dev2/results/trees/$n/cache/${ds}_$f.pkl ] || continue
  # 3000 trees: the rule is unchanged, so a model already done stays
  [ $n = 3000 ] && [ -f $W/repo_hybrid/results/trees_pairs/3000/${ds}_${f}_d4_twin_lat.npy ] && continue
  echo "$n $ds $f"; done; done; done > $W/queue4_exp4.txt
say "B. $(wc -l < $W/queue4_exp4.txt) runs"
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$W/repo_hybrid:$W/chirho_pkgs xargs -a $W/queue4_exp4.txt -P 8 -L 1 bash -c 'timeout 43200 '$PY' mech_main.py --data '$W'/data/causal --dataset $1 --family $2 --backend twin --trees $0 --out trees_pairs/$0 >> '$W'/logs/trees_pairs/$1_$2_$0.log 2>&1'
say "B. done"
say "R. real memory of the final index"; cd $W; $PY run_rss.py 6 >> $W/logs/run_rss_final.txt 2>&1
while pgrep -f "mech_rss[.]py --dataset .* --family [A-Z]*$" >/dev/null; do sleep 120; done; say "R. done"
say "C. memory without grouping"
cd $W/repo_hybrid/experiments
for f in HGB LGBM; do OMP_NUM_THREADS=1 PYTHONPATH=$W/repo_hybrid timeout 86400 $PY mech_rss.py --dataset ailerons --family $f --grouping none --threads 24 >> $W/logs/none/ailerons_$f.log 2>&1 & done
cd $W; $PY run_none64.py 6 >> $W/logs/run_none64.txt 2>&1
while pgrep -f "mech_rss[.]py.*grouping none" >/dev/null; do sleep 120; done; say "C. done"
say "queue4 ended"
