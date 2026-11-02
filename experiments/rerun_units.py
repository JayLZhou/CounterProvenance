"""Launch the missing units of unfinished jobs, one process per (job, unit), while our memory allows.

Base jobs run with --baselines-only (no index), ours jobs with --ours-only; each writes its own CSV
(scm_full_<tag>_u<unit>.csv).  A new process starts only while all our processes hold < START_GB."""
import glob, json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

W = Path("/mnt/data/yingli/counterprovenance")
SETS = "concrete,light_tunnel,ailerons,pol,elevators,jdk,postgres,baf,mslr,census1990,susy,methane,higgs,airline,nyc_taxi".split(",")
START_GB = 400
BIG = {"nyc_taxi", "airline", "higgs", "methane", "susy", "census1990", "mslr", "baf", "postgres"}
MAX_RUNNING = {True: 30, False: 40}   # processes of ours and of the baselines
anc = {ds: json.loads((W / "data/causal" / ds / "meta.json").read_text())["target_ancestors"] for ds in SETS}
jobs = [(ds, f, d) for ds in SETS for d in (1, 2, 4, 8, 16) if d <= anc[ds] for f in ("LGBM", "XGB", "HGB", "RF", "GBR")]


def ours_gb():
    out = subprocess.run(["ps", "-u", "yingli", "-o", "rss="], capture_output=True, text=True).stdout.split()
    return sum(int(x) for x in out) / 1048576


def running_units(ours_only):
    """(tag, unit) of the running jobs; a job without --unit-ids runs all its remaining units."""
    ps = subprocess.run(["pgrep", "-af", "python scm_full.py"], capture_output=True, text=True).stdout.splitlines()
    out = set()
    for l in ps:
        if "--tag " not in l or "$0" in l or ("ours-only" in l) != ours_only:
            continue
        tag = l.split("--tag ")[1].split()[0]
        units = [int(l.split("--unit-ids ")[1].split()[0])] if "--unit-ids " in l else [0, 1, 2]
        out |= {(tag, u) for u in units}
    return out


def units_done(repo, tag):
    rows = []
    for f in glob.glob(str(W / repo / "results" / f"scm_full_{tag}.csv")) + glob.glob(str(W / repo / "results" / f"scm_full_{tag}_u*.csv")):
        rows.append(pd.read_csv(f))
    if not rows:
        return set()
    df = pd.concat(rows)
    return set(df.unit.astype(int))   # ok or infeasible: both are final


todo = []
sides = [("repo_dev2", True)] if "--ours" in sys.argv else [("repo", False)] if "--base" in sys.argv else [("repo", False), ("repo_dev2", True)]
for repo, ours_only in sides:
    run = running_units(ours_only)
    for ds, f, d in jobs:
        tag = f"{ds}_{f}_d{d}"
        for u in sorted({0, 1, 2} - units_done(repo, tag)):
            if (tag, u) not in run:
                todo.append((repo, ours_only, ds, f, d, tag, u))
print(f"{len(todo)} (job, unit) runs to launch", flush=True)
for t in todo:
    print("  ", t[0], t[5], "unit", t[6], flush=True)
if "--dry" in sys.argv:
    sys.exit()
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(W / "chirho_pkgs"))
for repo, ours_only, ds, f, d, tag, u in todo:
    while ours_gb() > START_GB or len(running_units(ours_only)) >= MAX_RUNNING[ours_only]:
        time.sleep(30)
    log = W / "logs" / ("scm_ours" if ours_only else "scm_full") / f"{tag}_u{u}.log"
    cmd = [str(W / "env/bin/python"), "scm_full.py", "--data", str(W / "data/causal"), "--dataset", ds, "--family", f,
           "--dims", str(d), "--tag", tag, "--unit-ids", str(u), "--build-threads", "16" if ds in BIG else "4",
           "--ours-only" if ours_only else "--baselines-only"]
    subprocess.Popen(cmd, cwd=W / repo / "experiments", env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    print(time.strftime("%F %T"), "started", repo, tag, "unit", u, flush=True)
    time.sleep(90 if ds in BIG else 3)   # let a big table load before measuring memory again
print("all launched", flush=True)
