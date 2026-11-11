#!/bin/bash
# Stop the ThreeStep of MINT on Adult RF100 (INF: 880 rows project to 9.4 days); W3 with 12 shards at once instead of 30;
# the remaining MINT jobs with 10 at once.  Shards resume from their checkpoints.
W=/mnt/data/yingli/counterprovenance; Q=$W/repo_lean; D=$W/repo_dev2; PY=$W/env/bin/python; DATA=$W/data/causal
k() { ps -u yingli -o pid=,args= | grep -- "$1" | grep -v grep | grep -v requeue | awk '{print $1}' | xargs -r kill; }
k "app3.sh"; k "xargs -a $W/app3_jobs.txt"; k "mintrun2.sh"; k "xargs -a $W/mint2_jobs.txt"; sleep 2
k "e2e_shard.py --data $DATA --dataset adult --family RF200 --app mcce_10000 --backend three_step"
k "e2e_shard.py --data $DATA --dataset adult --family RF100 --app recourse_mint --backend three_step"
sleep 3
echo "$(date '+%F %T') requeue: adult RF100 three_step of MINT stopped (INF); W3 to 12 at once; MINT to 10 at once" | tee -a $W/logs/mint/progress.txt >> $W/logs/app3/progress.txt
# MINT: the jobs neither ended nor running, without adult RF100 three_step
: > $W/mint3_jobs.txt
while read ds f b s e; do
  [ "$ds $f $b" = "adult RF100 three_step" ] && continue
  grep -q " end $ds $f $b $s rc=0" $W/logs/mint/progress.txt && continue
  ps -u yingli -o args= | grep -q -- "--dataset $ds --family $f --app recourse_mint --backend $b --start $s " && continue
  echo "$ds $f $b $s $e" >> $W/mint3_jobs.txt
done < $W/mint2_jobs.txt
grep "mcce_10000" $W/app3_jobs.txt > $W/w3_jobs.txt
echo "$(date '+%F %T') MINT left $(wc -l < $W/mint3_jobs.txt), W3 shards $(wc -l < $W/w3_jobs.txt), MINT still running $(ps -u yingli -o args= | grep '[e]2e_shard' | grep -v 'bash -c' | grep -v '^timeout' | grep -c recourse_mint)" >> $W/logs/mint/progress.txt
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= TMPDIR=$W/tmp
( cd $Q/experiments; PYTHONPATH=$Q:$W/chirho_pkgs xargs -a $W/mint3_jobs.txt -P 10 -L 1 bash -c 'timeout 432000 '$PY' e2e_shard.py --data '$DATA' --dataset $0 --family $1 --app recourse_mint --backend $2 --start $3 --stop $4 --actionable --build-threads 1 > '$W'/logs/mint/$0_$1_$2_$3.log 2>&1; echo "$(date "+%F %T") end $0 $1 $2 $3 rc=$?" >> '$W'/logs/mint/progress.txt'; echo "$(date '+%F %T') MINT3_DONE" >> $W/logs/mint/progress.txt ) &
( cd $D/experiments; PYTHONPATH=$D:$W/chirho_pkgs xargs -a $W/w3_jobs.txt -P 12 -L 1 bash -c 'timeout 108000 '$PY' e2e_shard.py --data '$DATA' --dataset $0 --family $1 --app $2 --backend three_step --start $3 --stop $4 >> '$W'/logs/app3/$0_$1_$2_$3.log 2>&1; echo "$(date "+%F %T") end $0 $1 $2 $3 rc=$?" >> '$W'/logs/app3/progress.txt'; echo "$(date '+%F %T') W3_DONE" >> $W/logs/app3/progress.txt ) &
wait
