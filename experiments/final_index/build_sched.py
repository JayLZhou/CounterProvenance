"""Builds the 75 final indexes with an estimate of the memory of each build, so that the builds running at once stay within
BUDGET GB: one group of an ensemble costs about 30 bytes per block and tree (measured: MET XGB MM264, 3.8e10 pairs, 1.13 TB),
otherwise the peak of the paper's run.  A build that fails is retried alone at the end."""
import glob, os, subprocess, sys, time
import pandas as pd
W = "/mnt/data/yingli/counterprovenance"; L = W + "/logs/final"; BUDGET = 900.0
fs = [f for f in glob.glob(W + '/repo_lean/results/rss/*.csv') if not f.endswith('_RF.csv')] + glob.glob(W + '/repo_lrf/results/rss/*_RF.csv')
r = pd.concat([pd.read_csv(f) for f in fs])
r['pairs'] = r.grid_all * r.trees
r['one'] = (r.grid_all <= 2e7) & (r.pairs <= 2e10)
est = {}
for (ds, fam), g in r.groupby(['dataset', 'family']):
    one = float((g.pairs * g.one).max()) * 30e-9
    est[(ds, fam)] = max(one, min(float(g.peak.max()) / 1e9, 400.0)) * 1.2 + 10
jobs = sorted(est, key=lambda k: -est[k])
def done(ds, fam):
    R = "repo_final_lrf" if fam == "RF" else "repo_final"
    return os.path.exists(f"{W}/{R}/results/main/{ds}_{fam}_build.csv")
def start(ds, fam):
    R = "repo_final_lrf" if fam == "RF" else "repo_final"
    env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", TMPDIR=W + "/tmp", CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{W}/{R}:{W}/chirho_pkgs")
    log = open(f"{L}/build_{ds}_{fam}.log", "a")
    return subprocess.Popen([W + "/env/bin/python", "build_index_cache.py", "--data", W + "/data/causal", "--dataset", ds, "--family", fam],
                            cwd=f"{W}/{R}/experiments", env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
def say(*a):
    with open(L + "/progress.txt", "a") as f: print(time.strftime("%F %T"), *a, file=f, flush=True)
for attempt, budget in ((1, BUDGET), (2, 1e9)):   # the retry runs one build at a time
    pending = [k for k in jobs if not done(*k)]
    running = {}
    while pending or running:
        for k, p in list(running.items()):
            if p.poll() is not None:
                say("build", *k, "exit=%s" % p.returncode, "est=%.0fGB" % est[k]); del running[k]
        used = sum(est[k] for k in running)
        for k in list(pending):
            if (not running) or (attempt == 1 and used + est[k] <= budget and len(running) < 6):
                running[k] = start(*k); pending.remove(k); used += est[k]; say("start", *k, "est=%.0fGB" % est[k], "attempt", attempt)
                if attempt == 2: break
        time.sleep(20)
missing = [k for k in jobs if not done(*k)]
say("builds missing:", missing)
