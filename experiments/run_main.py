"""All jobs of the main experiment: the index of every (table, family) first, then the baselines by increasing cost."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

W = Path("/mnt/data/yingli/counterprovenance")
R = W / "repo_dev2/results/main"
SETS = "concrete,light_tunnel,ailerons,pol,elevators,jdk,postgres,baf,mslr,census1990,susy,methane,higgs,airline,nyc_taxi".split(",")
BIG = {"nyc_taxi", "airline", "higgs", "methane", "susy", "census1990", "mslr", "baf", "postgres"}
FAMILIES = ("LGBM", "XGB", "HGB", "GBR")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 40
START_GB = 1150
anc = {ds: json.loads((W / "data/causal" / ds / "meta.json").read_text())["target_ancestors"] for ds in SETS}
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(W / "chirho_pkgs"))
(W / "logs/main").mkdir(parents=True, exist_ok=True)


def running():
    out = subprocess.run(["ps", "-u", "yingli", "-o", "rss=,args="], capture_output=True, text=True).stdout.splitlines()
    rows = [l for l in out if "env/bin/python mech_main.py" in l]
    return len(rows), sum(int(l.split()[0]) for l in rows) / 1048576, " ".join(rows)


def finished(ds, f, backend, d):
    if backend == "twin":
        path = R / f"{ds}_{f}_build.csv"
        if path.exists() and pd.read_csv(path).status.astype(str).str.startswith("infeasible").any():
            return True
        return all((R / f"{ds}_{f}_d{v}_twin.csv").exists() for v in (1, 2, 4, 8, 16) if v <= anc[ds])
    path = R / f"{ds}_{f}_d{d}_{backend}.csv"
    if not path.exists():
        return False
    s = pd.read_csv(path).iloc[0]
    return int(s.units_done) >= int(s.units) or str(s.status).startswith("stopped")


jobs = [(ds, f, "twin", 0) for ds in SETS for f in FAMILIES]
jobs += [(ds, f, b, d) for b in ("three_step", "chirho", "dowhy") for ds in SETS for d in (1, 2, 4, 8, 16) if d <= anc[ds] for f in FAMILIES]
print(len(jobs), "jobs", flush=True)
for ds, f, b, d in jobs:
    if finished(ds, f, b, d):
        continue
    while True:
        count, gb, text = running()
        tag = f"--dataset {ds} --family {f} --backend {b}" + (f" --d {d}" if d else "")
        if tag in text:
            break
        if count < LIMIT and gb < START_GB:
            if b != "twin" and not (R / "cache" / f"{ds}_{f}.pkl").exists():
                time.sleep(30)
                continue
            log = W / "logs/main" / f"{ds}_{f}_{b}_d{d}.log"
            cmd = [str(W / "env/bin/python"), "mech_main.py", "--data", str(W / "data/causal"), "--dataset", ds, "--family", f, "--backend", b]
            if d:
                cmd += ["--d", str(d)]
            subprocess.Popen(cmd, cwd=W / "repo_dev2/experiments", env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            print(time.strftime("%F %T"), "started", ds, f, b, d, flush=True)
            time.sleep(90 if b == "twin" and ds in BIG else 3)
            break
        time.sleep(30)
print("all launched", flush=True)
