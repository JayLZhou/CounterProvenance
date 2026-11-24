#!/bin/bash
# After BETA_DONE: the applications (index runs only), the one-thread build time and size (mech_rss.py), and the depth.
W=/mnt/data/yingli/counterprovenance; export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 TMPDIR=$W/tmp CUDA_VISIBLE_DEVICES=
L=$W/logs/final; export W L
app() { [ -s $W/repo_final/results/shard/$1_$2_$3_twin_0.npy ] && return 0
  cd $W/repo_final/experiments && PYTHONPATH=$W/repo_final:$W/shap_pkgs:$W/chirho_pkgs timeout 172800 $W/env/bin/python e2e_shard.py --data $W/data/causal --dataset $1 --family $2 --app $3 --backend twin >> $L/app_$1_$2_$3.log 2>&1
  echo "$(date '+%F %T') app $1 $2 $3 exit=$?" >> $L/progress.txt; }
rss() { R=repo_final; [ $2 = RF ] && R=repo_final_lrf; [ -s $W/$R/results/rss/$1_$2.csv ] && grep -q . $L/rss_$1_$2.done 2>/dev/null && return 0
  cd $W/$R/experiments && PYTHONPATH=$W/$R $W/env/bin/python mech_rss.py --dataset $1 --family $2 >> $L/rss_$1_$2.log 2>&1 && echo ok > $L/rss_$1_$2.done
  echo "$(date '+%F %T') rss $1 $2 exit=$?" >> $L/progress.txt; }
export -f app rss
cat > $L/apps.txt <<'APPS'
adult HGB100 dowhy_feature_relevance
adult HGB100 fairness_cf
adult HGB100 recourse_mint
adult HGB100 what_if_mediated
adult RF100 dowhy_feature_relevance
adult RF100 fairness_cf
adult RF100 recourse_mint
adult RF100 what_if_mediated
adult RF200 mcce_10000
adult RF200 mcce_1000
adult RF500D10 geco_genetic
german_full HGB100 dowhy_feature_relevance
german_full RF200 mcce_10000
german_full RF200 mcce_1000
german_full RF500D10 geco_genetic
german_mint HGB100 fairness_cf
german_mint HGB100 recourse_mint
german_mint HGB100 what_if_mediated
german_mint RF100 fairness_cf
german_mint RF100 recourse_mint
german_mint RF100 what_if_mediated
give_credit HGB100 dowhy_feature_relevance
give_credit RF100 dowhy_feature_relevance
give_credit RF500D10 geco_genetic
heloc HGB100 dowhy_feature_relevance
APPS
xargs -P 4 -L 1 bash -c 'app $0 $1 $2' < $L/apps.txt &
du -s $W/repo_lean/results/indexes/* $W/repo_lrf/results/indexes/* | sort -n -r | awk '{print $2}' | xargs -n1 basename | sed -E 's/_t1000_.*//; s/_(GBR|HGB|LGBM|XGB|RF)$/ \1/' | xargs -P 3 -L 1 bash -c 'rss $0 $1'
wait
echo APPS_RSS_DONE >> $L/done.txt
cd $W/depth_final && $W/env/bin/python depth_driver.py 6 >> $L/depth_driver.log 2>&1
echo DEPTH_DONE >> $L/done.txt
