#!/bin/bash
# Start depth_exp3 once the ThreeStep shards of W3 (MCCE K=10^4) are gone.
W=/mnt/data/yingli/counterprovenance
while pgrep -f "app mcce_10000 --backend three_step" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') W3 gone, depth_exp3 starts" >> $W/depth_exp3/progress.txt
cd $W/depth_exp3 && $W/env/bin/python depth_driver3.py 16 > driver.log 2>&1
echo "$(date '+%F %T') DEPTH3_DONE" >> $W/depth_exp3/progress.txt
