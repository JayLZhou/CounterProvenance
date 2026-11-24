#!/bin/bash
# Starts batch 2 after MAIN_DONE (1 TB split 700 + 300 GB) and batch 3 after BETA_DONE and COMBINE_DONE (1 TB).
W=/mnt/data/yingli/counterprovenance; L=$W/logs/final; export XDG_RUNTIME_DIR=/run/user/$(id -u)
while ! grep -q MAIN_DONE $L/done.txt 2>/dev/null; do sleep 120; done
systemctl --user set-property --runtime cf-final.scope MemoryMax=700G
systemd-run --user --scope -q --unit=cf-final2 -p MemoryMax=300G -p MemorySwapMax=0 -p OOMPolicy=continue bash $W/run_final2.sh > $W/logs/final_run2.out 2>&1
while ! { grep -q BETA_DONE $L/done.txt && grep -q COMBINE_DONE $L/done.txt; } 2>/dev/null; do sleep 120; done
systemd-run --user --scope -q --unit=cf-final3 -p MemoryMax=1000G -p MemorySwapMax=0 -p OOMPolicy=continue bash $W/run_final3.sh > $W/logs/final_run3.out 2>&1
