"""Launch the missing units of unfinished jobs, one process per (job, unit), while our memory allows.

Base jobs run with --baselines-only (no index), ours jobs with --ours-only; each writes its own CSV
(scm_full_<tag>_u<unit>.csv).  A new process starts only while all our processes hold < START_GB."""
import glob, json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

W = Path("/mnt/data/yingli/counterprovenance")
SETS = "acs_income,jdk,concrete,light_tunnel,higgs,baf,postgres,methane,mslr,gov_census,house_sales,census1990,susy,airline,nyc_taxi".split(",")
START_GB = 400
anc = {ds: json.loads((W / "data/causal" / ds / "meta.json").read_text())["target_ancestors"] for ds in SETS}
jobs = [(ds, f, d) for ds in SETS for d in (1, 2, 4, 8) if d <= anc[ds] for f in ("LGBM", "XGB", "HGB", "RF", "GBR")]


def ours_gb():
    out = subprocess.run(["ps", "-u", "yingli", "-o", "rss="], capture_output=True, text=True).stdout.split()
    return sum(int(x) for x in out) / 1048576


def running_tags(ours_only):
    ps = subprocess.run(["pgrep", "-af", "python scm_full.py"], capture_output=True, text=True).stdout.splitlines()
    return {l.split("--tag ")[1].split()[0] for l in ps if "--tag " in l and "$0" not in l and ("ours-only" in l) == ours_only}


def units_done(repo, tag):
    rows = []
    for f in glob.glob(str(W / repo / "results" / f"scm_full_{tag}.csv")) + glob.glob(str(W / repo / "results" / f"scm_full_{tag}_u*.csv")):
        rows.append(pd.read_csv(f))
    if not rows:
        return set()
    df = pd.concat(rows)
    return set(df.unit.astype(int))   # ok or infeasible: both are final


todo = []
sides = [("repo_dev2", True)] if "--ours" in sys.argv else [("repo", False), ("repo_dev2", True)]
for repo, ours_only in sides:
    run = running_tags(ours_only)
    for ds, f, d in jobs:
        tag = f"{ds}_{f}_d{d}"
        if tag in run:
            continue
        for u in sorted({0, 1, 2} - units_done(repo, tag)):
            todo.append((repo, ours_only, ds, f, d, tag, u))
print(f"{len(todo)} (job, unit) runs to launch", flush=True)
for t in todo:
    print("  ", t[0], t[5], "unit", t[6], flush=True)
if "--dry" in sys.argv:
    sys.exit()
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(W / "chirho_pkgs"))
for repo, ours_only, ds, f, d, tag, u in todo:
    while ours_gb() > START_GB:
        time.sleep(60)
    log = W / "logs" / ("scm_ours" if ours_only else "scm_full") / f"{tag}_u{u}.log"
    cmd = [str(W / "env/bin/python"), "scm_full.py", "--data", str(W / "data/causal"), "--dataset", ds, "--family", f,
           "--dims", str(d), "--tag", tag, "--unit-ids", str(u), "--build-threads", "16",
           "--ours-only" if ours_only else "--baselines-only"]
    subprocess.Popen(cmd, cwd=W / repo / "experiments", env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    print(time.strftime("%F %T"), "started", repo, tag, "unit", u, flush=True)
    time.sleep(90)   # let it load its data before measuring memory again
print("all launched", flush=True)
