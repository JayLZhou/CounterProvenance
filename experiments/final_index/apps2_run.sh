#!/bin/bash
# The applications with mediators (index runs): fairness_cf with --sensitive, what_if_mediated and recourse_mint with --actionable.
W=/mnt/data/yingli/counterprovenance; L=$W/logs/finalb; export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 TMPDIR=$W/tmp CUDA_VISIBLE_DEVICES=
app() { [ -s $W/repo_final/results/shard/$1_$2_$3_twin_0.npy ] && return 0
  flag=--actionable; [ $3 = fairness_cf ] && flag=--sensitive
  cd $W/repo_final/experiments && PYTHONPATH=$W/repo_final:$W/shap_pkgs:$W/chirho_pkgs timeout 172800 $W/env/bin/python e2e_shard.py --data $W/data/causal --dataset $1 --family $2 --app $3 --backend twin $flag --build-threads 1 >> $L/app2_$1_$2_$3.log 2>&1
  echo "$(date '+%F %T') app2 $1 $2 $3 exit=$?" >> $L/progress.txt; }
export -f app; export W L
for ds in adult german_mint; do for f in HGB100 RF100; do for a in fairness_cf what_if_mediated recourse_mint; do echo "$ds $f $a"; done; done; done | xargs -P 6 -L 1 bash -c 'app $0 $1 $2'
echo APPS2_DONE >> $L/done.txt
