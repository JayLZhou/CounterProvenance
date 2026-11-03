#!/bin/bash
# Keep our memory within the limit on the shared machine: whenever the processes of this experiment hold more than
# 800 GB, or the machine has less than 200 GB available, kill our largest job.  Other work of the account is not counted.
LOG=/mnt/data/yingli/counterprovenance/logs/memguard.txt
LIMIT=800
while true; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  ours=$(ps -u yingli -o rss=,args= | grep "counterprovenance/env/bin/python" | awk '{s+=$1} END {print int(s/1048576)}')
  if [ "$ours" -gt "$LIMIT" ] || [ "$avail" -lt 200 ]; then
    line=$(ps -u yingli -o pid=,rss=,etimes=,args= --sort=-rss | grep -E "python (scm_full|e2e_apps|e2e_table|e2e_shard|beta_probe|box_scan|weighted_bench)" | grep -v -E "bash|timeout" | head -1)
    pid=$(echo "$line" | awk '{print $1}')
    if [ -n "$pid" ]; then
      echo "$(date '+%F %T') ours=${ours}GB avail=${avail}GB kill $(echo "$line" | awk '{printf "%s %dGB age=%dh", $1, $2/1048576, $3/3600}') $(echo "$line" | grep -o -- '--tag [^ ]*') $(echo "$line" | grep -o -- '--unit-ids [^ ]*') $(echo "$line" | grep -o ours-only)" >> $LOG
      kill $pid; sleep 20; kill -9 $pid 2>/dev/null
      continue
    fi
  fi
  sleep 30
done
