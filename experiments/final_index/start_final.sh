#!/bin/bash
W=/mnt/data/yingli/counterprovenance; cd $W
export XDG_RUNTIME_DIR=/run/user/$(id -u)
pgrep -f "memguard2.sh" > /dev/null || (nohup setsid bash $W/memguard2.sh > /dev/null 2>&1 < /dev/null &)
nohup setsid systemd-run --user --scope -q --unit=cf-final -p MemoryMax=1000G -p MemorySwapMax=0 -p OOMPolicy=continue bash $W/run_final.sh > $W/logs/final_run.out 2>&1 < /dev/null &
sleep 5
systemctl --user show cf-final.scope -p MemoryMax -p ActiveState 2>&1 | tr "\n" " "; echo
pgrep -af "run_final.sh|build_index_cache.py" | cut -c1-140 | head -5
