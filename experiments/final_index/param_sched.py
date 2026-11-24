"""Appendix parameter study with the final index: one-group limit (blocks for 1,000 trees: 1e6, 1e7, 5e7, 1e8; 2e7 is the main run)
and pre-merge cap (1, 2^14; 2^10 is the main run).  mech_main per model and value; the builds running at once stay within
BUDGET GB by an estimate (30 bytes per block and tree of an ensemble that is one group, else the peak of the paper's run);
cheap values first; a failed run is retried alone at the end; one that exceeds the memory limit is recorded."""
import glob, os, subprocess, time
import pandas as pd
W = "/mnt/data/yingli/counterprovenance"; L = W + "/logs/finalb"; BUDGET = 900.0
fs = [f for f in glob.glob(W + '/repo_lean/results/rss/*.csv') if not f.endswith('_RF.csv')] + glob.glob(W + '/repo_lrf/results/rss/*_RF.csv')
r = pd.concat([pd.read_csv(f) for f in fs]); r['pairs'] = r.grid_all * r.trees
def est(ds, fam, pairs_limit, one_max):
    g = r[(r.dataset == ds) & (r.family == fam)]
    one = g[(g.grid_all <= one_max) & (g.pairs <= pairs_limit)]
    return max(float(one.pairs.max() if len(one) else 0) * 30e-9, min(float(g.peak.max()) / 1e9, 400.0)) * 1.2 + 10
models = sorted({(d, f) for d, f in zip(r.dataset, r.family)})
SETS = [("start/1", dict(cap=1)), ("start/16384", dict(cap=16384)), ("limit/1e6", dict(pairs=1e9)), ("limit/1e7", dict(pairs=1e10)),
        ("limit/5e7", dict(pairs=5e10)), ("limit/1e8", dict(pairs=1e11, one_max=1e8))]
jobs = []
for out, p in SETS:
    for d, f in models:
        pairs, om = p.get('pairs', 2e10), p.get('one_max', 5e7)
        jobs.append(dict(out=out, ds=d, fam=f, pairs=pairs, one_max=om, cap=p.get('cap'), est=est(d, f, pairs, om)))
def done(j):
    R = "repo_final_lrf" if j['fam'] == "RF" else "repo_final"
    return bool(glob.glob(f"{W}/{R}/results/{j['out']}/{j['ds']}_{j['fam']}_d*_twin.csv"))
def start(j):
    R = "repo_final_lrf" if j['fam'] == "RF" else "repo_final"
    env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", TMPDIR=W + "/tmp", CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{W}/{R}:{W}/chirho_pkgs")
    if j['cap']: env["CF_START_CAP"] = str(j['cap'])
    log = open(f"{L}/param_{j['out'].replace('/', '_')}_{j['ds']}_{j['fam']}.log", "a")
    cmd = [W + "/env/bin/python", "mech_main.py", "--data", W + "/data/causal", "--dataset", j['ds'], "--family", j['fam'], "--backend", "twin",
           "--out", j['out'], "--one-pairs", f"{j['pairs']:g}", "--one-max", f"{j['one_max']:g}"]
    return subprocess.Popen(["timeout", "28800"] + cmd, cwd=f"{W}/{R}/experiments", env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
def say(*a):
    with open(L + "/progress.txt", "a") as f: print(time.strftime("%F %T"), *a, file=f, flush=True)
for attempt in (1, 2):
    pending = [j for j in jobs if not done(j)]
    running = []
    while pending or running:
        for x in list(running):
            p, j = x
            if p.poll() is not None:
                say("param", j['out'], j['ds'], j['fam'], "exit=%s" % p.returncode, "done=%s" % done(j)); running.remove(x)
        used = sum(j['est'] for _, j in running)
        for j in list(pending):
            if not running or (attempt == 1 and used + j['est'] <= BUDGET and len(running) < 6):
                running.append((start(j), j)); pending.remove(j); used += j['est']
                if attempt == 2: break
        time.sleep(20)
say("param missing:", [(j['out'], j['ds'], j['fam']) for j in jobs if not done(j)])
with open(L + "/done.txt", "a") as f: print("PARAM_DONE", file=f)
