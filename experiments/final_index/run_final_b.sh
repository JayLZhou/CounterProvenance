#!/bin/bash
# The final index with the two-level grids read by rows over the intervals of the ensemble (layout B): the indexes are the same
# as in run_final.sh; the timings are taken again.  Main (Exp.1-3), trees (Exp.4), Exp.7, beta (Exp.5), then batch 3.
W=/mnt/data/yingli/counterprovenance; export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 TMPDIR=$W/tmp CUDA_VISIBLE_DEVICES=
L=$W/logs/finalb; mkdir -p $L
repo() { [ $1 = RF ] && echo repo_final_lrf || echo repo_final; }
export -f repo; export W L
models_by_size() {
  du -s $W/repo_lean/results/indexes/* $W/repo_lrf/results/indexes/* | sort -n -r | awk '{print $2}' | xargs -n1 basename | sed -E 's/_t1000_.*//; s/_(GBR|HGB|LGBM|XGB|RF)$/ \1/'
}
main() { R=$(repo $2); ls $W/$R/results/main/$1_$2_d*_twin.csv > /dev/null 2>&1 && return 0
  cd $W/$R/experiments && PYTHONPATH=$W/$R:$W/chirho_pkgs $W/env/bin/python mech_main.py --data $W/data/causal --dataset $1 --family $2 --backend twin --out main >> $L/main_$1_$2.log 2>&1
  echo "$(date '+%F %T') main $1 $2 exit=$?" >> $L/progress.txt; }
trees() { R=$(repo $2); ls $W/$R/results/trees/$3/$1_$2_d*_twin.csv > /dev/null 2>&1 && return 0
  cd $W/$R/experiments && PYTHONPATH=$W/$R:$W/chirho_pkgs $W/env/bin/python mech_main.py --data $W/data/causal --dataset $1 --family $2 --backend twin --trees $3 >> $L/trees_$1_$2_$3.log 2>&1
  echo "$(date '+%F %T') trees $1 $2 $3 exit=$?" >> $L/progress.txt; }
combine() { R=$(repo $2); [ -s $W/$R/results/combine/$1_$2_d4.csv ] && return 0
  cd $W/$R/experiments && PYTHONPATH=$W/$R timeout 28800 $W/env/bin/python combine_bench.py --data $W/data/causal --dataset $1 --family $2 --build-threads 8 >> $L/combine_$1_$2.log 2>&1
  echo "$(date '+%F %T') combine $1 $2 exit=$?" >> $L/progress.txt; }
beta() { R=$(repo $2); ls $W/$R/results/beta/$3/$1_$2_d*_twin.csv > /dev/null 2>&1 && return 0
  cd $W/$R/experiments && PYTHONPATH=$W/$R:$W/chirho_pkgs timeout 28800 $W/env/bin/python mech_main.py --data $W/data/causal --dataset $1 --family $2 --backend twin --out beta/$3 --budget $4 >> $L/beta_$1_$2_$3.log 2>&1
  echo "$(date '+%F %T') beta $1 $2 $3 exit=$?" >> $L/progress.txt; }
export -f main trees combine beta
models_by_size | xargs -P 6 -L 1 bash -c 'main $0 $1'
models_by_size | xargs -P 1 -L 1 bash -c 'main $0 $1'
echo MAIN_DONE >> $L/done.txt
for n in 100 300 3000; do models_by_size | sed "s/$/ $n/"; done | xargs -P 4 -L 1 bash -c 'trees $0 $1 $2'
for n in 100 300 3000; do models_by_size | sed "s/$/ $n/"; done | xargs -P 1 -L 1 bash -c 'trees $0 $1 $2'
echo TREES_DONE >> $L/done.txt
models_by_size | xargs -P 4 -L 1 bash -c 'combine $0 $1'
echo COMBINE_DONE >> $L/done.txt
for b in "2e6 2e+06" "2e8 2e+08"; do models_by_size | sed "s/$/ $b/"; done | xargs -P 3 -L 1 bash -c 'beta $0 $1 $2 $3'
for b in "2e6 2e+06" "2e8 2e+08"; do models_by_size | sed "s/$/ $b/"; done | xargs -P 1 -L 1 bash -c 'beta $0 $1 $2 $3'
echo BETA_DONE >> $L/done.txt
sed -i 's#L=\$W/logs/final; export W L#L=$W/logs/finalb; export W L#' $W/run_final3.sh
bash $W/run_final3.sh
