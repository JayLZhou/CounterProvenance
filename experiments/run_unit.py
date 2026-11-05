"""Exp.7: the optional index for one unit on the final code: 3 units per setting, one thread, the boosted models first."""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

W = Path("/mnt/data/yingli/counterprovenance")
SETS = "concrete,light_tunnel,ailerons,pol,elevators,jdk,postgres,baf,mslr,census1990,susy,methane,higgs,airline,nyc_taxi".split(",")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MIN_AVAILABLE_GB = 300
anc = {ds: json.loads((W / "data/causal" / ds / "meta.json").read_text())["target_ancestors"] for ds in SETS}
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(W / "chirho_pkgs"))
(W / "logs/exp7").mkdir(parents=True, exist_ok=True)


def running():
    return int(subprocess.run("pgrep -fc 'scm_full[.]py'", shell=True, capture_output=True, text=True).stdout.strip() or 0)


def available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


jobs = [(ds, f, d, u) for f in ("GBR", "LGBM", "XGB", "HGB", "RF") for ds in SETS for d in (1, 2, 4, 8, 16) if d <= anc[ds] for u in (0, 1, 2)]
print(len(jobs), "jobs", flush=True)
for ds, f, d, u in jobs:
    tag = f"exp7_{ds}_{f}_d{d}"
    if glob.glob(str(W / "repo_dev2/results" / f"scm_full_{tag}_u{u}.csv")):
        continue
    while running() >= LIMIT or available_gb() < MIN_AVAILABLE_GB:
        time.sleep(30)
    cmd = [str(W / "env/bin/python"), "scm_full.py", "--data", str(W / "data/causal"), "--dataset", ds, "--family", f, "--dims", str(d), "--tag", tag,
           "--unit-ids", str(u), "--build-threads", "1", "--ours-only", "--queries", "1000", "--dowhy-queries", "1000"]
    subprocess.Popen(cmd, cwd=W / "repo_dev2/experiments", env=env, stdout=open(W / "logs/exp7" / f"{tag}_u{u}.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(time.strftime("%F %T"), "started", tag, u, flush=True)
    time.sleep(20)
print("all launched", flush=True)
