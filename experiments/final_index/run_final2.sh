#!/bin/bash
# After MAIN_DONE: Exp.6 (final_structure.py, in repo_final_abl) and Exp.7 (combine_bench.py) on the final indexes.
W=/mnt/data/yingli/counterprovenance; export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 TMPDIR=$W/tmp CUDA_VISIBLE_DEVICES=
L=$W/logs/final; export W L
structure() { m=$(basename $1 | sed -E 's/_t1000_.*//'); [ -s $W/repo_final_abl/results/structure_final/${m}_sample.csv ] && return 0
  cd $W/repo_final_abl/experiments && PYTHONPATH=$W/repo_final_abl nice $W/env/bin/python final_structure.py --dir $1 --out $W/repo_final_abl/results/structure_final >> $L/structure_$m.log 2>&1
  echo "$(date '+%F %T') structure $m exit=$?" >> $L/progress.txt; }
combine() { R=repo_final; [ $2 = RF ] && R=repo_final_lrf; [ -s $W/$R/results/combine/$1_$2_d4.csv ] && return 0
  cd $W/$R/experiments && PYTHONPATH=$W/$R timeout 28800 $W/env/bin/python combine_bench.py --data $W/data/causal --dataset $1 --family $2 --build-threads 8 >> $L/combine_$1_$2.log 2>&1
  echo "$(date '+%F %T') combine $1 $2 exit=$?" >> $L/progress.txt; }
export -f structure combine
ls -d $W/repo_final/results/indexes/*_t1000_* $W/repo_final_lrf/results/indexes/*_t1000_* | xargs -P 3 -I{} bash -c 'structure {}'
echo STRUCTURE_DONE >> $L/done.txt
for ds in methane mslr higgs airline nyc_taxi susy census1990 baf postgres jdk elevators pol ailerons light_tunnel concrete; do for f in GBR HGB LGBM XGB RF; do echo "$ds $f"; done; done | xargs -P 4 -L 1 bash -c 'combine $0 $1'
echo COMBINE_DONE >> $L/done.txt
