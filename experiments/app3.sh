#!/bin/bash
# ThreeStep of the applications (repo_dev2, as the other baselines): W5 (Adult RF100 relevance) and W7 (Credit HGB100 relevance)
# timed on the first 100 rows of the random row order, and W3 (MCCE K=1e4) on the rows 2,000-45,222 that are left.
W=/mnt/data/yingli/counterprovenance; Q=$W/repo_dev2; L=$W/logs/app3; mkdir -p $L
grep -q -- "--stop" $Q/experiments/e2e_shard.py || { echo "no --stop in e2e_shard" > $L/progress.txt; exit 1; }
: > $W/app3_jobs.txt
for s in 0 25 50 75; do echo "adult RF100 dowhy_feature_relevance $s $((s+25))" >> $W/app3_jobs.txt; echo "give_credit HGB100 dowhy_feature_relevance $s $((s+25))" >> $W/app3_jobs.txt; done
for s in $(seq 2000 1000 45000); do e=$((s+1000)); [ $e -gt 45222 ] && e=45222; echo "adult RF200 mcce_10000 $s $e" >> $W/app3_jobs.txt; done
echo "$(date '+%F %T') $(wc -l < $W/app3_jobs.txt) jobs" >> $L/progress.txt
cd $Q/experiments
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= TMPDIR=$W/tmp PYTHONPATH=$Q:$W/chirho_pkgs
xargs -a $W/app3_jobs.txt -P 30 -L 1 bash -c 'timeout 108000 '$W'/env/bin/python e2e_shard.py --data '$W'/data/causal --dataset $0 --family $1 --app $2 --backend three_step --start $3 --stop $4 > '$L'/$0_$1_$2_$3.log 2>&1; echo "$(date "+%F %T") end $0 $1 $2 $3 rc=$?" >> '$L'/progress.txt'
echo "$(date '+%F %T') APP3_DONE" >> $L/progress.txt
