"""Every (table, family) of mech_map2.py, smallest index first; a large index waits until the memory allows it."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

W = Path("/mnt/data/yingli/counterprovenance")
M = W / "repo_dev2/results/main"
R = W / "repo_dev2/results/map2"
SETS = "concrete,light_tunnel,ailerons,pol,elevators,jdk,postgres,baf,mslr,census1990,susy,methane,higgs,airline,nyc_taxi".split(",")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 12
RESERVE_GB = 200
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
(W / "logs/map2").mkdir(parents=True, exist_ok=True)


def need_gb(ds, f):
    return float(pd.read_csv(M / f"{ds}_{f}_build.csv").bytes.max()) * 1.5 / 1e9 + 2


def running():
    return int(subprocess.run("pgrep -fc 'mech_map2[.]py'", shell=True, capture_output=True, text=True).stdout.strip() or 0)


def available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


jobs = sorted(((ds, f) for ds in SETS for f in ("GBR", "LGBM", "XGB", "HGB", "RF")), key=lambda j: need_gb(*j))
print(len(jobs), "jobs", flush=True)
for ds, f in jobs:
    log = W / "logs/map2" / f"{ds}_{f}.log"
    if log.exists() and "done" in log.read_text():
        continue
    need = need_gb(ds, f)
    while running() >= LIMIT or available_gb() < need + RESERVE_GB:
        time.sleep(30)
    subprocess.Popen([str(W / "env/bin/python"), "mech_map2.py", "--dataset", ds, "--family", f], cwd=W / "repo_dev2/experiments", env=env,
                     stdout=open(log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(time.strftime("%F %T"), "started", ds, f, round(need, 1), flush=True)
    time.sleep(3 if need < 50 else 60)
print("all launched", flush=True)
