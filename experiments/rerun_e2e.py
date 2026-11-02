"""Launch the missing end-to-end jobs, one process per (dataset, family, d, unit), while our memory allows."""
import glob, json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

W = Path("/mnt/data/yingli/counterprovenance")
SETS = "concrete,light_tunnel,ailerons,pol,elevators,jdk,postgres,baf,mslr,census1990,susy,methane,higgs,airline,nyc_taxi".split(",")
BIG = {"nyc_taxi", "airline", "higgs", "methane", "susy", "census1990", "mslr", "baf", "postgres"}
START_GB, MAX_RUNNING = 600, 15   # with rerun_units.py: 75 processes on 64 cores with 128 threads
anc = {ds: json.loads((W / "data/causal" / ds / "meta.json").read_text())["target_ancestors"] for ds in SETS}
jobs = [(ds, f, d, u) for u in (0, 1, 2) for ds in SETS for d in (1, 2, 4, 8, 16) if d <= anc[ds]
        for f in ("LGBM", "XGB", "HGB", "RF", "GBR")]


# the applications on the tables of the papers they come from
GENETIC = ["recourse_genetic"]
RELEVANCE = ["dowhy_feature_relevance"]            # DoWhy-GCM's own API, on the tables of the recourse and attribution papers
MCCE = ["mcce_1000", "mcce_10000", "mcce_100000"]  # k = 1,000 by default, up to 100,000 in the paper
EXPLAIN = ["kernel_shap", "lime_tabular"]          # the defaults of the shap and lime libraries
OWN_APPS = dict(german_mint=["recourse_minimal"] + GENETIC + RELEVANCE, compas=GENETIC + RELEVANCE,
                adult=GENETIC + MCCE + RELEVANCE + EXPLAIN, german_full=GENETIC + MCCE + RELEVANCE, heloc=GENETIC + MCCE + RELEVANCE,
                give_credit=GENETIC + MCCE + RELEVANCE, bike=["shapley_attribution"] + RELEVANCE,
                boston=["dose_response"], white_wine=["dose_response"], pima=["dose_response"])


def ours_gb():
    # only the processes of this experiment: the account also runs other work on the machine
    out = subprocess.run(["ps", "-u", "yingli", "-o", "rss=,args="], capture_output=True, text=True).stdout.splitlines()
    return sum(int(l.split()[0]) for l in out if "counterprovenance/env/bin/python" in l) / 1048576


def running():
    ps = subprocess.run(["pgrep", "-af", "python e2e_apps.py"], capture_output=True, text=True).stdout.splitlines()
    return [l for l in ps if "--dataset " in l and "$" not in l]


def finished(ds, f, d, u):
    path = W / "repo_dev2/results" / f"e2e_{ds}_{f}_d{d}_u{u}.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    return all(((df.backend == "dowhy") & (df.app == a)).any() and ((df.backend == "online") & (df.app == a)).any() and
               (df.backend.astype(str).str.startswith("twin") & (df.app == a)).any() for a in OWN_APPS.get(ds, [df.app.iloc[0]]))


if "--workloads" in sys.argv:
    # the tables and the variables of the literature, with more units, as these tables are small
    jobs = []
    for ds in OWN_APPS:
        act = json.loads((W / "data/causal" / ds / "meta.json").read_text())["actionable"]
        jobs += [(ds, f, d, u) for u in range(10) for d in sorted({v for v in (1, 2, 4, 8) if v <= len(act)} | {len(act)})
                 for f in ("LGBM", "XGB", "HGB", "RF", "GBR")]

active = {tuple(l.split("--dataset ")[1].split()[0:1]) + (l.split("--family ")[1].split()[0], int(l.split("--d ")[1].split()[0]),
                                                           int(l.split("--unit ")[1].split()[0])) for l in running()}
todo = [j for j in jobs if j not in active and not finished(*j)]
print(len(todo), "end-to-end jobs to launch", flush=True)
if "--dry" in sys.argv:
    sys.exit()
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(W / "chirho_pkgs"))
(W / "logs/e2e").mkdir(exist_ok=True)
for ds, f, d, u in todo:
    while ours_gb() > START_GB or len(running()) >= MAX_RUNNING:
        time.sleep(30)
    log = W / "logs/e2e" / f"{ds}_{f}_d{d}_u{u}.log"
    cmd = [str(W / "env/bin/python"), "e2e_apps.py", "--data", str(W / "data/causal"), "--dataset", ds, "--family", f,
           "--d", str(d), "--unit", str(u), "--build-threads", "16" if ds in BIG else "4"]
    if "--workloads" in sys.argv:
        cmd += ["--apps", ",".join(OWN_APPS[ds])]
    subprocess.Popen(cmd, cwd=W / "repo_dev2/experiments", env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    print(time.strftime("%F %T"), "started", ds, f, d, u, flush=True)
    time.sleep(90 if ds in BIG else 3)
print("all launched", flush=True)
