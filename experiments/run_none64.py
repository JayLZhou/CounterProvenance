"""With 64-bit grid numbering, the six models whose grid exceeds 2^32 but whose map could fit: TWIN without grouping (one group, cell graph and MST) for every model, all mechanisms loaded at once: which models fit in 512 GB.
A run is stopped when its resident memory exceeds 512 GB.  Output: ``results/rss_none``, logs in ``logs/none``."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

W = Path("/mnt/data/yingli/counterprovenance")
Q = W / "repo_64"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 12
CAP_KB = 512 << 20
SETS = "concrete,light_tunnel,ailerons,pol,elevators,jdk,postgres,baf,mslr,census1990,susy,methane,higgs,airline,nyc_taxi".split(",")
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(Q))
LOGS = W / "logs/none64"
LOGS.mkdir(parents=True, exist_ok=True)
B = pd.concat([pd.read_csv(f) for f in (W / "repo_dev2/results/structure").glob("*_beta.csv")])
GRID = B[B.beta == 2e7].groupby(["dataset", "family"]).grid.apply(lambda g: max(map(float, g))).to_dict()


def procs():
    out = subprocess.run(["ps", "-u", "yingli", "-o", "pid=,rss=,args="], capture_output=True, text=True).stdout.splitlines()
    return [l.split() for l in out if "mech_rss.py" in l and "none" in l and not l.split()[2].endswith("timeout")]


def watch():
    for p in procs():
        if int(p[1]) > CAP_KB:
            ds, f = p[p.index("--dataset") + 1], p[p.index("--family") + 1]
            with open(LOGS / f"{ds}_{f}.log", "a") as log:
                log.write(f"{time.strftime('%F %T')} resident memory over 512 GB ({int(p[1]) >> 20} GB): stopped\n")
            subprocess.run(["kill", p[0]])


def available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


def wait(seconds):
    watch()
    for _ in range(seconds // 10):
        time.sleep(10)
        watch()


SIX = {("susy", "XGB"), ("nyc_taxi", "GBR"), ("methane", "RF"), ("jdk", "XGB"), ("jdk", "HGB"), ("jdk", "LGBM")}
jobs = sorted(SIX, key=lambda j: GRID.get(j, 0))
print(len(jobs), "jobs", flush=True)
for ds, f in jobs:
    log = LOGS / f"{ds}_{f}.log"
    if log.exists() and any(k in log.read_text(errors="ignore") for k in ("done", "over 512 GB")):
        continue
    if any(p[p.index("--dataset") + 1] == ds and p[p.index("--family") + 1] == f for p in procs()):
        continue  # already running
    big = GRID.get((ds, f), 0) > 1e9
    while len(procs()) >= LIMIT or available_gb() < (600 if big else 100):
        wait(30)
    subprocess.Popen(["timeout", "86400", str(W / "env/bin/python"), "mech_rss.py", "--dataset", ds, "--family", f, "--grouping", "none"], cwd=Q / "experiments", env=env,
                     stdout=open(log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(time.strftime("%F %T"), "started", ds, f, f"{GRID.get((ds, f), 0):.3g}", flush=True)
    wait(120 if big else 0)
print("all launched", flush=True)
while procs():
    wait(30)
print("all ended", flush=True)
