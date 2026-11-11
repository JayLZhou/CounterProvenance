#!/bin/bash
# The batch read without expanding a value table: TWIN of MINT (4) and of the seven workloads again, on repo_lean with the fix.
W=/mnt/data/yingli/counterprovenance; Q=$W/repo_lean; LG=$W/logs/twinfix; mkdir -p $LG $Q/results/shard_before_batchfix
cp $W/index_fixed.py $Q/counterprovenance/index.py
P=$(ps -u yingli -o pid=,args= | grep "[e]2e_shard.py" | grep -- "--dataset german_mint --family RF100 --app recourse_minimal --backend twin" | grep -v "bash -c" | awk '{print $1}')
echo "$(date '+%F %T') stop slow german RF100 twin: $P" >> $LG/progress.txt; [ -n "$P" ] && kill $P; sleep 3
mv $Q/results/shard/*_twin_0.* $Q/results/shard_before_batchfix/ 2>/dev/null
J=$W/twinfix_jobs.txt
cat > $J <<J
german_mint HGB100 recourse_minimal --actionable
german_mint RF100 recourse_minimal --actionable
adult HGB100 recourse_minimal --actionable
adult RF100 recourse_minimal --actionable
adult RF500D10 geco_genetic
adult RF200 mcce_1000
adult RF200 mcce_10000
adult HGB100 dowhy_feature_relevance
adult RF100 dowhy_feature_relevance
german_full HGB100 dowhy_feature_relevance
give_credit HGB100 dowhy_feature_relevance
J
cd $Q/experiments; export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= TMPDIR=$W/tmp PYTHONPATH=$Q:$W/chirho_pkgs
xargs -a $J -P 4 -L 1 bash -c 'timeout 172800 '$W'/env/bin/python e2e_shard.py --data '$W'/data/causal --dataset $0 --family $1 --app $2 --backend twin $3 > '$LG'/$0_$1_$2.log 2>&1; echo "$(date "+%F %T") end $0 $1 $2 rc=$?" >> '$LG'/progress.txt'
echo "$(date '+%F %T') TWINFIX_DONE" >> $LG/progress.txt
