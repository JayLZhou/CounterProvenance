"""Naive CellTable (mech_naive.py) on all trees, at most 1 TB of resident memory and 24 hours per model: the models where it does not fit.
A build is stopped when its resident memory exceeds 1 TB (a limit on the address space would stop builds that reserve more than they use).  A new build starts
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
JOBS = [tuple(l.split()) for l in (W / "naive_jobs.txt").read_text().split("\n") if l.strip()]
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(Q))
(W / "logs/naive").mkdir(parents=True, exist_ok=True)


def running():
    return int(subprocess.run("pgrep -fc 'mech_naive[.]py'", shell=True, capture_output=True, text=True).stdout.strip() or 0)


def available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


def watch():
    """Stop every build whose resident memory exceeds 1 TB, and say so in its log."""
    out = subprocess.run(["ps", "-u", "yingli", "-o", "pid=,rss=,args="], capture_output=True, text=True).stdout.splitlines()
    for line in out:
        p = line.split()
        if "mech_naive.py" in line and "timeout" not in p[2] and int(p[1]) > (1 << 30):
            ds, f = p[p.index("--dataset") + 1], p[p.index("--family") + 1]
            with open(W / "logs/naive" / f"{ds}_{f}.log", "a") as log:
                log.write(f"{time.strftime('%F %T')} resident memory over 1 TB ({int(p[1]) >> 20} GB): stopped\n")
            subprocess.run(["kill", p[0]])


def wait(seconds):
    for _ in range(seconds // 10):
        watch()
        time.sleep(10)


print(len(JOBS), "jobs", flush=True)
for ds, f in JOBS:
    log = W / "logs/naive" / f"{ds}_{f}.log"
    if log.exists() and any(k in log.read_text(errors="ignore") for k in ("done", "over 1 TB", "MemoryError")):
        continue  # finished, or over 1 TB: a second try would end the same way
    if f"--dataset {ds} --family {f} " in subprocess.run(["ps", "-u", "yingli", "-o", "args="], capture_output=True, text=True).stdout:
        continue
    while running() >= LIMIT or available_gb() < START_GB:
        wait(60)
    cmd = ["timeout", "86400", str(W / "env/bin/python"), "mech_naive.py", "--dataset", ds, "--family", f]
    subprocess.Popen(cmd, cwd=Q / "experiments", env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(time.strftime("%F %T"), "started", ds, f, flush=True)
    wait(120)  # let it grow before the next one is judged
print("all launched", flush=True)
while running():
    wait(60)
print("all ended", flush=True)
