#!/bin/bash
# Keep the experiments running: restart a scheduler that ended while units are missing, and the memory guard.
W=/mnt/data/yingli/counterprovenance
cd $W
LOG=$W/logs/keeper.txt
end=$(( $(date +%s) + 3600 * ${1:-24} ))
while [ $(date +%s) -lt $end ]; do
  pgrep -f "memguard[.]sh" > /dev/null || { setsid nohup ./memguard.sh > /dev/null 2>&1 < /dev/null & echo "$(date '+%F %T') memguard restarted" >> $LOG; }
  for side in ours base; do
    if ! pgrep -f "rerun_units[.]py --$side" > /dev/null; then
      left=$(env/bin/python rerun_units.py --$side --dry 2>/dev/null | head -1 | cut -d' ' -f1)
      if [ "${left:-0}" -gt 0 ]; then
        setsid nohup env/bin/python rerun_units.py --$side >> logs/rerun_keeper_$side.txt 2>&1 < /dev/null &
        echo "$(date '+%F %T') $side restarted, $left units left" >> $LOG
      fi
    fi
  done
  sleep 600
done
