"""Exp.2: the main experiment again on the cached models, keeping the time of every query. Larger indexes wait until the memory allows them."""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

W = Path("/mnt/data/yingli/counterprovenance")
M = W / "repo_dev2/results/main"
R = W / "repo_dev2/results/latency"
SETS = "concrete,light_tunnel,ailerons,pol,elevators,jdk,postgres,baf,mslr,census1990,susy,methane,higgs,airline,nyc_taxi".split(",")
FAMILIES = ("LGBM", "XGB", "HGB", "GBR", "RF")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
RESERVE_GB = 200
anc = {ds: json.loads((W / "data/causal" / ds / "meta.json").read_text())["target_ancestors"] for ds in SETS}
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(W / "chirho_pkgs"))
(W / "logs/latency").mkdir(parents=True, exist_ok=True)


def need_gb(ds, f):
    return float(pd.read_csv(M / f"{ds}_{f}_build.csv").bytes.sum()) * 1.5 / 1e9 + 2


def running():
    out = subprocess.run(["ps", "-u", "yingli", "-o", "args="], capture_output=True, text=True).stdout.splitlines()
    return sum("env/bin/python mech_main.py" in l and "--out latency" in l for l in out)


def available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


def finished(ds, f, b, d):
    if b == "twin":
        return all((R / f"{ds}_{f}_d{v}_twin_lat.npy").exists() for v in (1, 2, 4, 8, 16) if v <= anc[ds])
    path = R / f"{ds}_{f}_d{d}_{b}.csv"
    if not path.exists():
        return False
    s = pd.read_csv(path).iloc[0]
    return int(s.units_done) >= int(s.units) or str(s.status).startswith("stopped")


twins = sorted(((ds, f, "twin", 0) for ds in SETS for f in FAMILIES), key=lambda j: need_gb(j[0], j[1]))
jobs = [j for j in twins if need_gb(j[0], j[1]) < 50]
jobs += [(ds, f, b, d) for f in FAMILIES for b in ("three_step", "chirho", "dowhy") for ds in SETS for d in (1, 2, 4, 8, 16) if d <= anc[ds]]
jobs += [j for j in twins if need_gb(j[0], j[1]) >= 50]
print(len(jobs), "jobs", flush=True)
for ds, f, b, d in jobs:
    if finished(ds, f, b, d):
        continue
    tag = f"--dataset {ds} --family {f} --backend {b}" + (f" --d {d}" if d else "")
    if tag in subprocess.run(["ps", "-u", "yingli", "-o", "args="], capture_output=True, text=True).stdout:
        continue
    need = need_gb(ds, f) if b == "twin" else 2
    while running() >= LIMIT or available_gb() < need + RESERVE_GB:
        time.sleep(30)
    cmd = [str(W / "env/bin/python"), "mech_main.py", "--data", str(W / "data/causal"), "--dataset", ds, "--family", f, "--backend", b, "--out", "latency"]
    if d:
        cmd += ["--d", str(d)]
    subprocess.Popen(cmd, cwd=W / "repo_dev2/experiments", env=env, stdout=open(W / "logs/latency" / f"{ds}_{f}_{b}_d{d}.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(time.strftime("%F %T"), "started", ds, f, b, d, flush=True)
    time.sleep(3 if need < 50 else 60)
print("all launched", flush=True)
