"""CellTable without groups on all trees, at most 1 TB and 24 hours per model: the models where it does not fit.  A new build starts
only while the machine keeps more than 1.1 TB available, so that a build that grows to 1 TB does not starve the others."""
import os
import subprocess
import sys
import time
from pathlib import Path

W = Path("/mnt/data/yingli/counterprovenance")
Q = W / "repo_hybrid"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 4
START_GB = float(sys.argv[2]) if len(sys.argv) > 2 else 1100
JOBS = [tuple(l.split()) for l in (W / "ct1tb_jobs.txt").read_text().split("\n") if l.strip()]
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(Q))
(W / "logs/ct1tb").mkdir(parents=True, exist_ok=True)


def running():
    return int(subprocess.run("pgrep -fc 'mech_celltable[.]py.*ct1tb'", shell=True, capture_output=True, text=True).stdout.strip() or 0)


def available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


def one_tb():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (1 << 40, 1 << 40))


print(len(JOBS), "jobs", flush=True)
for ds, f in JOBS:
    log = W / "logs/ct1tb" / f"{ds}_{f}.log"
    if log.exists() and any(k in log.read_text(errors="ignore") for k in ("done", "bad_alloc", "MemoryError", "dumped core")):
        continue  # finished, or out of the 1 TB: a second try would end the same way
    if f"--dataset {ds} --family {f} " in subprocess.run(["ps", "-u", "yingli", "-o", "args="], capture_output=True, text=True).stdout:
        continue
    while running() >= LIMIT or available_gb() < START_GB:
        time.sleep(60)
    cmd = ["timeout", "86400", str(W / "env/bin/python"), "mech_celltable.py", "--dataset", ds, "--family", f, "--trees", "1000", "--cap", "2.5e11", "--out", "ct1tb"]
    subprocess.Popen(cmd, cwd=Q / "experiments", env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, preexec_fn=one_tb)
    print(time.strftime("%F %T"), "started", ds, f, flush=True)
    time.sleep(120)  # let it grow before the next one is judged
print("all launched", flush=True)
