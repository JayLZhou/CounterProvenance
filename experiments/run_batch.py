"""Batch experiment (batch_main.py), launcher; adapted from: CausalAI made exact (causalai_baseline.py) on every setting of the main experiment: same cached SCMs, units, and
queries; boosted families in repo_lean, RF in repo_lrf; output results/causalai of each repo."""
import json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

W = Path("/mnt/data/yingli/counterprovenance")
SETS = "concrete,light_tunnel,ailerons,pol,elevators,jdk,postgres,baf,mslr,census1990,susy,methane,higgs,airline,nyc_taxi".split(",")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 48
START_GB = 300
anc = {ds: json.loads((W / "data/causal" / ds / "meta.json").read_text())["target_ancestors"] for ds in SETS}
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", CAUSALAI_PATH=str(W / "causalai_src/repo"))
(W / "logs/batch").mkdir(parents=True, exist_ok=True)

def repo(f): return W / ("repo_lrf" if f == "RF" else "repo_lean")

def running():
    out = subprocess.run(["ps", "-u", "yingli", "-o", "rss=,args="], capture_output=True, text=True).stdout.splitlines()
    rows = [l for l in out if "env/bin/python batch_main.py" in l]
    return len(rows), sum(int(l.split()[0]) for l in rows) / 1048576, " ".join(rows)

def finished(ds, f, d, B):
    p = repo(f) / "results/batch" / f"{ds}_{f}_d{d}_{B}.csv"
    if not p.exists(): return False
    s = pd.read_csv(p).iloc[0]
    return True

jobs = [(ds, f, d, B) for B in ("causalai", "dowhy") for f in ("GBR", "LGBM", "XGB", "HGB", "RF") for ds in SETS for d in (1, 2, 4, 8, 16) if d <= anc[ds]]
jobs = [j for j in jobs if (repo(j[1]) / "results/main/cache" / f"{j[0]}_{j[1]}.pkl").exists()]
print(len(jobs), "jobs", flush=True)
for ds, f, d, B in jobs:
    if finished(ds, f, d, B): continue
    while True:
        count, gb, text = running()
        tag = f"--dataset {ds} --family {f} --backend {B} --d {d} "
        if tag in text + " ": break
        if count < LIMIT and gb < START_GB:
            log = W / "logs/batch" / f"{ds}_{f}_d{d}_{B}.log"
            cmd = [str(W / "env/bin/python"), "batch_main.py", "--data", str(W / "data/causal"), "--dataset", ds, "--family", f, "--backend", B, "--d", str(d), "--out", "batch"]
            subprocess.Popen(cmd, cwd=repo(f) / "experiments", env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            print(time.strftime("%F %T"), "started", ds, f, d, B, flush=True)
            time.sleep(2)
            break
        time.sleep(20)
print("all launched", flush=True)
