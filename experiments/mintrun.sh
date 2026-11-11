#!/bin/bash
# MINT (causal recourse through minimal interventions) on whole tables: Adult and German (Karimi et al.), HGB100 and RF100,
# with TWIN (one index per ensemble, mediators replayed), ThreeStep and DoWhy-GCM; Adult baselines in shards of 1,000 rows.
W=/mnt/data/yingli/counterprovenance; Q=$W/repo_lean; LG=$W/logs/mint; mkdir -p $LG
rm -f $Q/results/shard/*recourse_minimal*
J=$W/mint_jobs.txt; : > $J
for ds in german_mint adult; do for f in HGB100 RF100; do
  n=$( [ $ds = adult ] && echo 45222 || echo 1000 )
  echo "$ds $f twin 0 $n" >> $J
done; done
for f in HGB100 RF100; do for b in three_step dowhy; do echo "german_mint $f $b 0 1000" >> $J; done; done
for s in $(seq 0 1000 45000); do e=$((s+1000)); [ $e -gt 45222 ] && e=45222
  for f in HGB100 RF100; do for b in three_step dowhy; do echo "adult $f $b $s $e" >> $J; done; done; done
echo "$(date '+%F %T') $(wc -l < $J) jobs" >> $LG/progress.txt
cd $Q/experiments; export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= TMPDIR=$W/tmp PYTHONPATH=$Q:$W/chirho_pkgs
xargs -a $J -P 16 -L 1 bash -c 'timeout 432000 '$W'/env/bin/python e2e_shard.py --data '$W'/data/causal --dataset $0 --family $1 --app recourse_minimal --backend $2 --start $3 --stop $4 --actionable --build-threads 1 > '$LG'/$0_$1_$2_$3.log 2>&1; echo "$(date "+%F %T") end $0 $1 $2 $3 rc=$?" >> '$LG'/progress.txt'
echo "$(date '+%F %T') MINT_DONE" >> $LG/progress.txt
