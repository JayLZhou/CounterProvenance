#!/bin/bash
# Save the indexes of all 75 models (build_index_cache.py), the slow ones first.
W=/mnt/data/yingli/counterprovenance; export OMP_NUM_THREADS=1 TMPDIR=$W/tmp
mkdir -p $W/logs/index_cache
for ds in methane mslr higgs airline nyc_taxi susy census1990 baf postgres jdk elevators pol ailerons light_tunnel concrete; do
  for f in XGB LGBM HGB GBR RF; do R=repo_lean; [ $f = RF ] && R=repo_lrf; echo "$R $ds $f"; done
done | xargs -P 4 -L 1 bash -c 'cd '$W'/$0/experiments && '$W'/env/bin/python build_index_cache.py --data '$W'/data/causal --dataset $1 --family $2 >> '$W'/logs/index_cache/$1_$2.log 2>&1'
echo INDEX_CACHE_DONE >> $W/logs/index_cache/done.txt
