"""CellForest with larger grouping budgets on every (table, family), smallest first, at most LIMIT at once; a model that ended without all its results (e.g. stopped by the memory guard) runs again."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

W = Path("/mnt/data/yingli/counterprovenance")
Q = W / "repo_hybrid"
BETAS = ("1e8", "5e8", "2e9")
M = W / "repo_dev2/results/main"
SETS = "concrete,light_tunnel,ailerons,pol,elevators,jdk,postgres,baf,mslr,census1990,susy,methane,higgs,airline,nyc_taxi".split(",")
FAMILIES = ("LGBM", "XGB", "HGB", "GBR", "RF")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 20
RESERVE_GB = 200
anc = {ds: json.loads((W / "data/causal" / ds / "meta.json").read_text())["target_ancestors"] for ds in SETS}
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{Q}:{W / 'chirho_pkgs'}", TMPDIR=str(W / "tmp"))
(W / "logs/beta_twin").mkdir(parents=True, exist_ok=True)


def need_gb(ds, f):
    # the old index held 12 bytes per grid cell; building the map needs 4 of them
    return float(pd.read_csv(M / f"{ds}_{f}_build.csv").bytes.sum()) * 0.6 / 1e9 + 2


def running():
    out = subprocess.run(["ps", "-u", "yingli", "-o", "args="], capture_output=True, text=True).stdout.splitlines()
    return sum("env/bin/python mech_main.py" in l and "--out beta_twin/" in l and not l.startswith("timeout") for l in out)


def available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


jobs = sorted(((ds, f) for ds in SETS for f in FAMILIES), key=lambda j: need_gb(*j))
print(len(jobs), "models", flush=True)
for beta in BETAS:
    R = Q / "results/beta_twin" / beta
    for ds, f in jobs:
        if all((R / f"{ds}_{f}_d{v}_twin_lat.npy").exists() for v in (1, 2, 4, 8, 16) if v <= anc[ds]):
            continue
        if f"--dataset {ds} --family {f} --backend twin --budget {beta} " in subprocess.run(["ps", "-u", "yingli", "-o", "args="], capture_output=True, text=True).stdout:
            continue
        need = need_gb(ds, f) * float(beta) / 2e7  # a larger budget, larger tables
        while running() >= LIMIT or available_gb() < min(need, 600) + RESERVE_GB:
            time.sleep(30)
        subprocess.Popen(["timeout", "43200", str(W / "env/bin/python"), "mech_main.py", "--data", str(W / "data/causal"), "--dataset", ds, "--family", f, "--backend", "twin", "--budget", beta, "--out", f"beta_twin/{beta}"],
                         cwd=Q / "experiments", env=env, stdout=open(W / "logs/beta_twin" / f"{ds}_{f}_{beta}.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(time.strftime("%F %T"), "started", beta, ds, f, flush=True)
        time.sleep(3)
print("all launched", flush=True)
