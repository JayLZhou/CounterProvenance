#!/bin/bash
# Other combinations of the trees (combine_bench.py) on the quiet sample: one model per table at d=4 (the models of quiet_sample.sh).
W=/mnt/data/yingli/counterprovenance; export OMP_NUM_THREADS=1 TMPDIR=$W/tmp
mkdir -p $W/logs/combine
F=(GBR HGB RF XGB LGBM); i=0
for ds in concrete light_tunnel ailerons pol elevators jdk postgres baf mslr census1990 susy methane higgs airline nyc_taxi; do
  f=${F[$((i % 5))]}; i=$((i+1)); R=repo_lean; [ $f = RF ] && R=repo_lrf; echo "$R $ds $f"
done | xargs -P 6 -L 1 bash -c 'cd '$W'/$0/experiments && timeout 21600 '$W'/env/bin/python combine_bench.py --data '$W'/data/causal --dataset $1 --family $2 --build-threads 8 > '$W'/logs/combine/$1_$2.log 2>&1'
echo COMBINE_DONE >> $W/logs/combine/done.txt
