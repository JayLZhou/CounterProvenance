"""Batch experiment, TWIN: one build per (table, family), then every d (batch_main.py --backend twin --d 0)."""
import json, os, subprocess, sys, time
from pathlib import Path
W = Path("/mnt/data/yingli/counterprovenance")
SETS = "concrete,light_tunnel,ailerons,pol,elevators,jdk,postgres,baf,mslr,census1990,susy,methane,higgs,airline,nyc_taxi".split(",")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 16
START_GB = 500
anc = {ds: json.loads((W / "data/causal" / ds / "meta.json").read_text())["target_ancestors"] for ds in SETS}
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
(W / "logs/batch").mkdir(parents=True, exist_ok=True)
def repo(f): return W / ("repo_lrf" if f == "RF" else "repo_lean")
def running():
    out = subprocess.run(["ps", "-u", "yingli", "-o", "rss=,args="], capture_output=True, text=True).stdout.splitlines()
    rows = [l for l in out if "env/bin/python batch_main.py" in l and "--backend twin" in l]
    return len(rows), sum(int(l.split()[0]) for l in rows) / 1048576, " ".join(rows)
def finished(ds, f):
    return all((repo(f) / "results/batch" / f"{ds}_{f}_d{d}_twin.csv").exists() for d in (1, 2, 4, 8, 16) if d <= anc[ds])
jobs = [(ds, f) for f in ("GBR", "XGB", "LGBM", "HGB", "RF") for ds in SETS if (repo(f) / "results/main/cache" / f"{ds}_{f}.pkl").exists()]
print(len(jobs), "jobs", flush=True)
for ds, f in jobs:
    if finished(ds, f): continue
    while True:
        count, gb, text = running()
        if f"--dataset {ds} --family {f} " in text: break
        if count < LIMIT and gb < START_GB:
            cmd = [str(W / "env/bin/python"), "batch_main.py", "--data", str(W / "data/causal"), "--dataset", ds, "--family", f, "--backend", "twin", "--d", "0", "--out", "batch"]
            subprocess.Popen(cmd, cwd=repo(f) / "experiments", env=env, stdout=open(W / "logs/batch" / f"{ds}_{f}_twin.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            print(time.strftime("%F %T"), "started", ds, f, flush=True)
            time.sleep(5)
            break
        time.sleep(20)
print("all launched", flush=True)
