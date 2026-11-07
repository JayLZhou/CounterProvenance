"""The index without the value table, with short cell numbers: TWIN on every (table, family), smallest first, on the cached models of the main experiment."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

W = Path("/mnt/data/yingli/counterprovenance")
Q = W / (sys.argv[2] if len(sys.argv) > 2 else "repo_compact")
OUT = sys.argv[3] if len(sys.argv) > 3 else "compact"
M = W / "repo_dev2/results/main"
R = Q / "results" / OUT
SETS = "concrete,light_tunnel,ailerons,pol,elevators,jdk,postgres,baf,mslr,census1990,susy,methane,higgs,airline,nyc_taxi".split(",")
FAMILIES = ("LGBM", "XGB", "HGB", "GBR", "RF")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 20
RESERVE_GB = 200
anc = {ds: json.loads((W / "data/causal" / ds / "meta.json").read_text())["target_ancestors"] for ds in SETS}
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{Q}:{W / 'chirho_pkgs'}")
(W / "logs" / OUT).mkdir(parents=True, exist_ok=True)


def need_gb(ds, f):
    # the old index held 12 bytes per grid cell; building the map needs 4 of them
    return float(pd.read_csv(M / f"{ds}_{f}_build.csv").bytes.sum()) * 0.6 / 1e9 + 2


def running():
    out = subprocess.run(["ps", "-u", "yingli", "-o", "args="], capture_output=True, text=True).stdout.splitlines()
    return sum("env/bin/python mech_main.py" in l and f"--out {OUT}" in l for l in out)


def available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


SKIP = set(tuple(l.split()) for l in (W / "skip_final.txt").read_text().splitlines() if l.strip()) if (W / "skip_final.txt").exists() else set()
jobs = sorted(((ds, f) for ds in SETS for f in FAMILIES if (ds, f) not in SKIP), key=lambda j: need_gb(*j))
print(len(jobs), "jobs", flush=True)
for ds, f in jobs:
    if all((R / f"{ds}_{f}_d{v}_twin_lat.npy").exists() for v in (1, 2, 4, 8, 16) if v <= anc[ds]):
        continue
    need = need_gb(ds, f)
    while running() >= LIMIT or available_gb() < need + RESERVE_GB:
        time.sleep(30)
    subprocess.Popen([str(W / "env/bin/python"), "mech_main.py", "--data", str(W / "data/causal"), "--dataset", ds, "--family", f, "--backend", "twin", "--out", OUT],
                     cwd=Q / "experiments", env=env, stdout=open(W / "logs" / OUT / f"{ds}_{f}.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(time.strftime("%F %T"), "started", ds, f, round(need, 1), flush=True)
    time.sleep(3 if need < 50 else 60)
print("all launched", flush=True)
