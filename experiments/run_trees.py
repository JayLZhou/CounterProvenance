"""Exp.4: the main experiment with 100, 300 and 3,000 trees (1,000 is the main run): the index of every (table, family), then ThreeStep at d=4."""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

W = Path("/mnt/data/yingli/counterprovenance")
SETS = "concrete,light_tunnel,ailerons,pol,elevators,jdk,postgres,baf,mslr,census1990,susy,methane,higgs,airline,nyc_taxi".split(",")
FAMILIES = ("GBR", "LGBM", "XGB", "HGB", "RF")
TREES = (100, 300, 3000)
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
MIN_AVAILABLE_GB = 400
PROCESS_GB = 400
anc = {ds: json.loads((W / "data/causal" / ds / "meta.json").read_text())["target_ancestors"] for ds in SETS}
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(W / "chirho_pkgs"))
(W / "logs/trees").mkdir(parents=True, exist_ok=True)


def running():
    out = subprocess.run(["ps", "-u", "yingli", "-o", "args="], capture_output=True, text=True).stdout.splitlines()
    return sum("env/bin/python mech_main.py" in l for l in out)


def available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


def limit_memory():
    resource.setrlimit(resource.RLIMIT_AS, (PROCESS_GB << 30, PROCESS_GB << 30))


def finished(n, ds, f, b):
    out = W / f"repo_dev2/results/trees/{n}"
    if b == "twin":
        return all((out / f"{ds}_{f}_d{v}_twin.csv").exists() for v in (1, 2, 4, 8, 16) if v <= anc[ds]) or (out / f"{ds}_{f}_build.csv").exists()
    return (out / f"{ds}_{f}_d4_three_step.csv").exists()


jobs = [(n, ds, f, "twin") for n in TREES for f in FAMILIES for ds in SETS]
jobs += [(n, ds, f, "three_step") for n in TREES for f in FAMILIES for ds in SETS if anc[ds] >= 4]
print(len(jobs), "jobs", flush=True)
for n, ds, f, b in jobs:
    if finished(n, ds, f, b):
        continue
    while running() >= LIMIT or available_gb() < MIN_AVAILABLE_GB or (b != "twin" and not (W / f"repo_dev2/results/trees/{n}/cache/{ds}_{f}.pkl").exists()):
        time.sleep(30)
    cmd = [str(W / "env/bin/python"), "mech_main.py", "--data", str(W / "data/causal"), "--dataset", ds, "--family", f, "--backend", b, "--trees", str(n)]
    if b != "twin":
        cmd += ["--d", "4"]
    log = W / "logs/trees" / f"{ds}_{f}_{n}_{b}.log"
    subprocess.Popen(cmd, cwd=W / "repo_dev2/experiments", env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, preexec_fn=limit_memory)
    print(time.strftime("%F %T"), "started", n, ds, f, b, flush=True)
    time.sleep(20 if b == "twin" else 2)
print("all launched", flush=True)
