#!/bin/bash
W=/mnt/data/yingli/counterprovenance; L=$W/logs/finalb; export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 TMPDIR=$W/tmp CUDA_VISIBLE_DEVICES=
[ -f $W/repo_final/experiments/mint_paths.py ] || cp $W/repo_lean/experiments/mint_paths.py $W/repo_final/experiments/
cd $W/repo_final/experiments
for m in "adult HGB100" "adult RF100" "german_mint HGB100" "german_mint RF100"; do set -- $m
  ( PYTHONPATH=$W/repo_final:$W/shap_pkgs:$W/chirho_pkgs timeout 14400 $W/env/bin/python mint_paths.py --data $W/data/causal --dataset $1 --family $2 --persons 30 > $L/paths_$1_$2.log 2>&1; echo "$(date '+%F %T') paths $1 $2 exit=$?" >> $L/progress.txt ) &
done
wait
echo PATHS_DONE >> $L/done.txt
