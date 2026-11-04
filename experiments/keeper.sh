#!/bin/bash
# Keep the experiments running: restart a scheduler that ended while units are missing, and the memory guard.
W=/mnt/data/yingli/counterprovenance
cd $W
LOG=$W/logs/keeper.txt
end=$(( $(date +%s) + 3600 * ${1:-24} ))
while [ $(date +%s) -lt $end ]; do
  pgrep -f "memguard[.]sh" > /dev/null || { setsid nohup ./memguard.sh > /dev/null 2>&1 < /dev/null & echo "$(date '+%F %T') memguard restarted" >> $LOG; }
  if ! pgrep -f "run_main[.]py" > /dev/null && ! tail -1 logs/run_main.txt 2>/dev/null | grep -q "all launched"; then
    setsid nohup env/bin/python run_main.py 40 >> logs/run_main.txt 2>&1 < /dev/null &
    echo "$(date '+%F %T') main restarted" >> $LOG
  fi
  sleep 600
done
