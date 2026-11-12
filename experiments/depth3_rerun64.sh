#!/bin/bash
# depth_exp3 again for the runs that the index code itself would try: no cap on the total blocks (only the code's own
# limit of 1e9 grid cells for one tree), 64-bit persistent tree (repo_lean64), one run at a time, 1 TB of address space.
W=/mnt/data/yingli/counterprovenance; D=$W/depth_exp3
cd $D; mkdir -p first_pass; cp -n *.csv first_pass/ 2>/dev/null
sed -e 's#REPO = ROOT + "/repo_lean"#REPO = ROOT + "/repo_lean64"#' \
    -e 's#feasible = G.max() <= 1e9 and G.sum() <= 5e10#feasible = G.max() <= 1e9#' \
    -e 's#"infeasible: over 1e9 blocks in a tree or 5e10 in total"#"infeasible: a tree over 1e9 grid cells, the limit of the index for one tree"#' depth_exp3.py > depth_exp3_64.py
grep -c repo_lean64 depth_exp3_64.py; grep -c "G.max() <= 1e9$" depth_exp3_64.py
while read d n f h; do
  [ -z "$d" ] && continue
  echo "$(date '+%F %T') start $d $n $f $h" >> progress.txt
  ( ulimit -v 1000000000; timeout 21600 $W/env/bin/python depth_exp3_64.py $d $n $f $h ) >> rerun64.log 2>&1
  echo "$(date '+%F %T') end $d $n $f $h rc=$?" >> progress.txt
done <<L
mslr relevance LGBM 100
jdk total_unit_test_time XGB 20
jdk total_unit_test_time XGB 50
jdk total_unit_test_time XGB 100
pol foo HGB 50
pol foo LGBM 50
pol foo HGB 100
pol foo LGBM 100
pol foo XGB 50
pol foo XGB 100
mslr relevance XGB 20
pol foo XGB 20
mslr relevance XGB 100
mslr relevance XGB 50
jdk total_unit_test_time RF 10
ailerons goal HGB 50
ailerons goal LGBM 50
ailerons goal HGB 100
ailerons goal LGBM 100
mslr relevance HGB 20
mslr relevance HGB 50
mslr relevance HGB 100
L
echo "$(date '+%F %T') RERUN64_DONE" >> progress.txt
