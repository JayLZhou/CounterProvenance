"""Night of 2026-11-08, restart (c): every remaining experiment on the final (lean) index, in one scheduler.

Streams, by priority: applications on the final index; the real memory of the final index (with the sizes of the
ablation); the RF depth study; the grouping budget beta (2e6, 2e8) on every model with groups; the one-group grid
kappa (fills of 1e7, the missing models of 1e8).  A job reserves the build peak measured in results/rss for its model;
jobs start while the reservations fit into BUDGET_GB and the machine keeps FLOOR_GB available.  A job that ends
without its output runs once more."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

W = Path("/mnt/data/yingli/counterprovenance")
LEAN = W / "repo_lean"
PY = str(W / "env/bin/python")
DATA = str(W / "data/causal")
LOGS = W / "logs/night6c"
FIX_T = float(sys.argv[3]) if len(sys.argv) > 3 else time.time()   # the rank-vector fix: memory measured before it runs again
LOGS.mkdir(parents=True, exist_ok=True)
BUDGET_GB = float(sys.argv[1]) if len(sys.argv) > 1 else 1450
WEIGHT = int(sys.argv[2]) if len(sys.argv) > 2 else 30
FLOOR_GB = 250
SETS = "concrete,light_tunnel,ailerons,pol,elevators,jdk,postgres,baf,mslr,census1990,susy,methane,higgs,airline,nyc_taxi".split(",")
D16 = {"ailerons", "census1990", "mslr", "pol"}
FAMS = ("GBR", "HGB", "RF", "XGB", "LGBM")
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{LEAN}:{W / 'chirho_pkgs'}", TMPDIR=str(W / "tmp"))


def say(*a):
    with open(LOGS / "sched.txt", "a") as f:
        print(time.strftime("%F %T"), *a, file=f, flush=True)


def peak_gb(ds, f):
    p = W / "repo_hybrid/results/rss" / f"{ds}_{f}.csv"
    return float(pd.read_csv(p).peak.max()) / 1e9 if p.exists() else 50.0


def grouped(ds, f):
    b = pd.read_csv(W / "repo_hybrid/results/final" / f"{ds}_{f}_build.csv")
    return int(b.groups.max()) > 1


def available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


def last_lat(out, ds, f):
    return LEAN / "results" / out / f"{ds}_{f}_d{16 if ds in D16 else 8}_twin_lat.npy"


jobs = []   # (name, cmd, cwd, est_gb, weight, timeout_s, done)


def rss_done(ds, f):
    p = LEAN / "results/rss" / f"{ds}_{f}.csv"
    if not p.exists() or p.stat().st_mtime < FIX_T:
        return False
    t = pd.read_csv(p)
    return "pft_nodes" in t.columns and len(t) == len(pd.read_csv(W / "repo_dev2/results/main" / f"{ds}_{f}_build.csv"))


def external(job):
    """A process of this job that is already running (started by the scheduler before the restart)."""
    sig = " ".join(job[1][1:])
    out = subprocess.run(["ps", "-u", "yingli", "-o", "args="], capture_output=True, text=True).stdout.splitlines()
    return any(line.startswith(PY) and sig in line for line in out)
apps = [("adult", "RF500D10", "geco_genetic"), ("adult", "RF200", "mcce_1000"), ("adult", "RF200", "mcce_10000"),
        ("adult", "HGB100", "dowhy_feature_relevance"), ("adult", "RF100", "dowhy_feature_relevance"),
        ("german_full", "HGB100", "dowhy_feature_relevance"), ("give_credit", "HGB100", "dowhy_feature_relevance"),
        ("heloc", "HGB100", "dowhy_feature_relevance")]
for ds, fam, app in apps:
    out = LEAN / "results/shard" / f"{ds}_{fam}_{app}_twin_0.csv"
    jobs.append((f"app_{ds}_{fam}_{app}", [PY, "e2e_shard.py", "--data", DATA, "--dataset", ds, "--family", fam, "--app", app, "--backend", "twin"],
                 LEAN / "experiments", 80.0, 4, 30 * 3600, lambda out=out: out.exists()))
for ds in SETS:
    out = LEAN / "results/rf_depth" / f"{ds}.csv"
    jobs.append((f"rfdepth_{ds}", [PY, "rf_depth.py", "--data", DATA, "--dataset", ds],
                 LEAN / "experiments", 40.0, 4, 3 * 3600, lambda out=out: out.exists() and len(pd.read_csv(out)) == 2))
rss = sorted(((ds, f) for ds in SETS for f in FAMS), key=lambda j: peak_gb(*j))
for ds, f in rss:
    jobs.append((f"rss_{ds}_{f}", [PY, "mech_rss.py", "--dataset", ds, "--family", f], LEAN / "experiments",
                 peak_gb(ds, f) * 1.05 + 5, 1, 8 * 3600, lambda ds=ds, f=f: rss_done(ds, f)))
for beta, scale in (("2e6", 1.0), ("2e8", None)):
    for ds, f in rss:
        if not grouped(ds, f):
            continue
        jobs.append((f"beta{beta}_{ds}_{f}", [PY, "mech_main.py", "--data", DATA, "--dataset", ds, "--family", f, "--backend", "twin",
                                              "--budget", beta, "--out", f"beta_final/{beta}"],
                     LEAN / "experiments", peak_gb(ds, f) + 5 if scale else max(350.0, 1.6 * peak_gb(ds, f)), 1, 8 * 3600,
                     lambda o=f"beta_final/{beta}", ds=ds, f=f: last_lat(o, ds, f).exists()))
H = W / "repo_hybrid/results"
for name in sorted(os.listdir(H / "final")):
    if not name.endswith("_d4_twin_lat.npy"):
        continue
    if not (H / "final_1e7" / name).exists():
        ds, f = name[:-len("_d4_twin_lat.npy")].rsplit("_", 1)
        jobs.append((f"kappa1e7_{ds}_{f}", [PY, "mech_main.py", "--data", DATA, "--dataset", ds, "--family", f, "--backend", "twin",
                                            "--one-pairs", "1e10", "--out", "kappa/1e7"],
                     LEAN / "experiments", peak_gb(ds, f) + 5, 1, 8 * 3600, lambda ds=ds, f=f: last_lat("kappa/1e7", ds, f).exists()))
for name in sorted(os.listdir(H / "one1e8")):
    if name.endswith("_build.csv"):
        ds, f = name[:-len("_build.csv")].rsplit("_", 1)
        if not (H / "one1e8" / f"{ds}_{f}_d4_twin_lat.npy").exists():
            jobs.append((f"kappa1e8_{ds}_{f}", [PY, "mech_main.py", "--data", DATA, "--dataset", ds, "--family", f, "--backend", "twin",
                                                "--one-pairs", "1e11", "--one-max", "1e8", "--out", "kappa/1e8"],
                         LEAN / "experiments", peak_gb(ds, f) * 2.5 + 5, 1, 8 * 3600, lambda ds=ds, f=f: last_lat("kappa/1e8", ds, f).exists()))

say(f"{len(jobs)} jobs, budget {BUDGET_GB} GB, weight {WEIGHT}")
queue = [j for j in jobs if not j[6]()]
# the largest builds first, while the machine is empty; the rest fill the remaining budget around them
queue.sort(key=lambda j: -j[3] if j[3] > 300 else 0)
say(f"{len(queue)} to run:", " ".join(sorted({j[0].split('_')[0] for j in queue})))
running, tries = [], {}
for job in list(queue):
    if external(job):
        queue.remove(job)
        running.append((None, job, time.time()))
        say(f"adopt {job[0]} est {job[3]:.0f} GB")
while queue or running:
    for r in running[:]:
        p, job, t0 = r
        if (p is None and external(job)) or (p is not None and p.poll() is None):
            continue
        running.remove(r)
        ok = job[6]()
        rc = None if p is None else p.returncode
        say(f"end {job[0]} rc={rc} ok={ok} {(time.time() - t0) / 60:.0f} min")
        if not ok and tries.get(job[0], 0) < 2 and rc != 124:
            queue.append(job)
    used = sum(j[3] for _, j, _ in running)
    weight = sum(j[4] for _, j, _ in running)
    started = False
    for job in list(queue):
        if job[3] > BUDGET_GB:
            say(f"skip {job[0]}: needs {job[3]:.0f} GB")
            queue.remove(job)
            continue
        if used + job[3] <= BUDGET_GB and weight + job[4] <= WEIGHT and available_gb() - job[3] >= FLOOR_GB:
            queue.remove(job)
            tries[job[0]] = tries.get(job[0], 0) + 1
            log = LOGS / f"{job[0]}.log"
            p = subprocess.Popen(["timeout", str(job[5])] + job[1], cwd=job[2], env=env, stdout=open(log, "a"),
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            running.append((p, job, time.time()))
            used += job[3]
            weight += job[4]
            say(f"start {job[0]} est {job[3]:.0f} GB (reserved {used:.0f} GB, weight {weight})")
            started = True
            if job[3] > 100:
                break   # let a large build allocate before the next decision
    time.sleep(20 if started else 30)
say("all done")
