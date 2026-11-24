#!/bin/bash
# Figure 19 with the final index: the index without grouping (mech_rss.py --grouping none, leaves as a log) for the models whose
# index without grouping was built before and is not one group anyway.  MET first (inside 500 GB, while the queue runs in the
# other 500 GB); AIR after the queue (cf-finalb) has finished, alone in 1000 GB.
W=/mnt/data/yingli/counterprovenance; export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 TMPDIR=$W/tmp CUDA_VISIBLE_DEVICES= XDG_RUNTIME_DIR=/run/user/$(id -u)
L=$W/logs/finalb
none() { R=repo_final; [ $2 = RF ] && R=repo_final_lrf
  systemd-run --user --scope -q --unit=cf-none-$1-$2 -p MemoryMax=$3 -p MemorySwapMax=0 -p OOMPolicy=continue \
    env PYTHONPATH=$W/$R bash -c "cd $W/$R/experiments && $W/env/bin/python mech_rss.py --dataset $1 --family $2 --grouping none" >> $L/none_$1_$2.log 2>&1
  echo "$(date '+%F %T') none $1 $2 exit=$?" >> $L/progress.txt; }
systemctl --user set-property --runtime cf-finalb.scope MemoryMax=500G
for f in GBR HGB LGBM; do none methane $f 500G; done
while systemctl --user is-active -q cf-finalb.scope; do sleep 120; done
for f in GBR HGB; do none airline $f 1000G; done
echo NONE_DONE >> $L/done.txt
