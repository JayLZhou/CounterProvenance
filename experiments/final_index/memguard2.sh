#!/bin/bash
# All Python jobs of the experiments of this project (env/bin/python) together: at most LIMIT GB of memory, and the machine
# keeps at least 150 GB available; otherwise the largest of them is killed.  Other work of the account is not touched.
LOG=/mnt/data/yingli/counterprovenance/logs/memguard2.txt
LIMIT=1000
while true; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  ours=$(ps -u yingli -o rss=,args= | grep "counterprovenance/env/bin/python" | awk '{s+=$1} END {print int(s/1048576)}')
  if [ "$ours" -gt "$LIMIT" ] || [ "$avail" -lt 150 ]; then
    line=$(ps -u yingli -o pid=,rss=,args= --sort=-rss | grep "counterprovenance/env/bin/python" | grep -v -E "bash|timeout|xargs" | head -1)
    pid=$(echo "$line" | awk '{print $1}')
    if [ -n "$pid" ]; then
      echo "$(date '+%F %T') ours=${ours}GB avail=${avail}GB kill $(echo "$line" | cut -c1-200)" >> $LOG
      kill $pid; sleep 20; kill -9 $pid 2>/dev/null
      continue
    fi
  fi
  sleep 20
done
