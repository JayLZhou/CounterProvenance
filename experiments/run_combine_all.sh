#!/bin/bash
# Other combinations of the trees (combine_bench.py) on all 75 models at d=4; the models already done are skipped.
W=/mnt/data/yingli/counterprovenance; export OMP_NUM_THREADS=1 TMPDIR=$W/tmp
mkdir -p $W/logs/combine
for ds in methane mslr higgs airline nyc_taxi susy census1990 baf postgres jdk elevators pol ailerons light_tunnel concrete; do
  for f in GBR HGB LGBM XGB RF; do
    R=repo_lean; [ $f = RF ] && R=repo_lrf
    [ -f $W/$R/results/combine/${ds}_${f}_d4.csv ] || echo "$R $ds $f"
  done
done | xargs -P 8 -L 1 bash -c 'cd '$W'/$0/experiments && timeout 28800 '$W'/env/bin/python combine_bench.py --data '$W'/data/causal --dataset $1 --family $2 --build-threads 8 > '$W'/logs/combine/$1_$2.log 2>&1'
echo COMBINE_ALL_DONE >> $W/logs/combine/done_all.txt
