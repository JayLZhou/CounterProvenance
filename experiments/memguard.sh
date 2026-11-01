#!/bin/bash
# Keep our memory at most half of the shared machine: whenever all our processes together hold more than
# 500 GB, or the machine has less than 80 GB available, kill our most recently started scm_full job
# (the one that loses the least work).
LOG=/mnt/data/yingli/counterprovenance/logs/memguard.txt
LIMIT=500
while true; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  ours=$(ps -u yingli -o rss= | awk '{s+=$1} END {print int(s/1048576)}')
  if [ "$ours" -gt "$LIMIT" ] || [ "$avail" -lt 80 ]; then
    line=$(ps -u yingli -o pid=,rss=,etimes=,args= --sort=etimes | grep "python scm_full" | grep -v bash | awk '$2 > 2097152' | head -1)
    pid=$(echo "$line" | awk '{print $1}')
    if [ -n "$pid" ]; then
      echo "$(date '+%F %T') ours=${ours}GB avail=${avail}GB kill $(echo "$line" | awk '{printf "%s %dGB age=%dh", $1, $2/1048576, $3/3600}') $(echo "$line" | grep -o -- '--tag [^ ]*') $(echo "$line" | grep -o -- '--unit-ids [^ ]*') $(echo "$line" | grep -o ours-only)" >> $LOG
      kill $pid; sleep 20; kill -9 $pid 2>/dev/null
      continue
    fi
  fi
  sleep 30
done
