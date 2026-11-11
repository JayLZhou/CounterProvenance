#!/bin/bash
# The remaining MINT jobs (not ended, not running, not adult RF100 three_step), 8 at once.
W=/mnt/data/yingli/counterprovenance; Q=$W/repo_lean; PY=$W/env/bin/python; DATA=$W/data/causal
ps -u yingli -o args= | grep "[e]2e_shard.py" | grep -v "bash -c" | grep -v "^timeout" > $W/logs/mint/running_now.txt
: > $W/mint3_jobs.txt
while read ds f b s e; do
  [ "$ds $f $b" = "adult RF100 three_step" ] && continue
  grep -q " end $ds $f $b $s rc=0" $W/logs/mint/progress.txt && continue
  grep -q -- "--dataset $ds --family $f --app recourse_mint --backend $b --start $s " $W/logs/mint/running_now.txt && continue
  echo "$ds $f $b $s $e" >> $W/mint3_jobs.txt
done < $W/mint2_jobs.txt
echo "$(date '+%F %T') requeue2: MINT left $(wc -l < $W/mint3_jobs.txt), running $(grep -c recourse_mint $W/logs/mint/running_now.txt)" >> $W/logs/mint/progress.txt
cd $Q/experiments; export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= TMPDIR=$W/tmp PYTHONPATH=$Q:$W/chirho_pkgs
xargs -a $W/mint3_jobs.txt -P 8 -L 1 bash -c 'timeout 432000 '$PY' e2e_shard.py --data '$DATA' --dataset $0 --family $1 --app recourse_mint --backend $2 --start $3 --stop $4 --actionable --build-threads 1 > '$W'/logs/mint/$0_$1_$2_$3.log 2>&1; echo "$(date "+%F %T") end $0 $1 $2 $3 rc=$?" >> '$W'/logs/mint/progress.txt'
echo "$(date '+%F %T') MINT3_DONE" >> $W/logs/mint/progress.txt
