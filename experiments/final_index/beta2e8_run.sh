#!/bin/bash
# The 2e8 budget runs that failed on the library mismatch, after the MET indexes without grouping (cf-none2): 3 at a time,
# then once more alone; then the records of their indexes (sizes corrected) for Exp.5.
W=/mnt/data/yingli/counterprovenance; L=$W/logs/finalb; export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 TMPDIR=$W/tmp XDG_RUNTIME_DIR=/run/user/$(id -u)
while systemctl --user is-active -q cf-none2.scope; do sleep 120; done
beta() { R=repo_final; [ $2 = RF ] && R=repo_final_lrf; ls $W/$R/results/beta/2e8/$1_$2_d*_twin.csv > /dev/null 2>&1 && return 0
  cd $W/$R/experiments && PYTHONPATH=$W/$R:$W/chirho_pkgs timeout 28800 $W/env/bin/python mech_main.py --data $W/data/causal --dataset $1 --family $2 --backend twin --out beta/2e8 --budget 2e+08 >> $L/beta2_$1_$2_2e8.log 2>&1
  echo "$(date '+%F %T') beta2 $1 $2 2e8 exit=$?" >> $L/progress.txt; }
export -f beta; export W L
models() { du -s $W/repo_lean/results/indexes/* $W/repo_lrf/results/indexes/* | sort -n -r | awk '{print $2}' | xargs -n1 basename | sed -E 's/_t1000_.*//; s/_(GBR|HGB|LGBM|XGB|RF)$/ \1/'; }
models | xargs -P 3 -L 1 bash -c 'beta $0 $1'
models | xargs -P 1 -L 1 bash -c 'beta $0 $1'
for d in $W/repo_final*/results/indexes/*_t1000_b2e+08_p2e+10_*; do out=$W/repo_final_abl/results/structure_more/$(basename $d)
  rm -rf $out; mkdir -p $out; cd $W/repo_final_abl/experiments && PYTHONPATH=$W/repo_final_abl $W/env/bin/python final_structure.py --dir $d --out $out --sample 0 > $out/log 2>&1 && echo ok > $out/done; done
echo BETA2_DONE >> $L/done.txt
