"""CausalAI made exact on every application workload (18 of Section 7 and the 4 of MINT), in shards, one thread each.
At most P at once, 300 GB kept free, 5 days per shard. Results in repo_lean/results/shard (backend causalai)."""
import os, subprocess, sys, time
W = "/mnt/data/yingli/counterprovenance"; PY = W + "/env/bin/python"; LOG = W + "/logs/app6"
P = int(sys.argv[1]) if len(sys.argv) > 1 else 48
os.makedirs(LOG, exist_ok=True)
ROWS = {"adult": 45222, "german_mint": 1000, "german_full": 1000, "give_credit": 120269}
WL = []  # (dataset, family, app, flag, shard)
for app, flag, sh in (("what_if_mediated", "--actionable", 3000), ("fairness_cf", "--sensitive", 3000), ("recourse_mint", "--actionable", 1000)):
    for ds in ("adult", "german_mint"):
        for fam in ("HGB100", "RF100"):
            WL.append((ds, fam, app, flag, sh))
for ds, fam in (("german_full", "HGB100"), ("adult", "HGB100"), ("adult", "RF100"), ("give_credit", "HGB100")):
    WL.append((ds, fam, "dowhy_feature_relevance", "", 1000))
for ds in ("german_full", "adult"):
    WL += [(ds, "RF200", "mcce_1000", "", 2000), (ds, "RF200", "mcce_10000", "", 2000), (ds, "RF500D10", "geco_genetic", "", 2000)]
jobs = [(ds, fam, app, flag, a, min(a + sh, ROWS[ds])) for ds, fam, app, flag, sh in WL for a in range(0, ROWS[ds], sh)]
def log(s):
    with open(LOG + "/progress.txt", "a") as f: f.write(time.strftime("%F %T ") + s + "\n")
def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1]) / 1048576
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{W}/repo_lean:{W}/chirho_pkgs", TMPDIR=W + "/tmp",
           CAUSALAI_PATH=W + "/causalai_src/repo")
log(f"{len(jobs)} jobs"); run = {}
while jobs or run:
    for k, (p, t0) in list(run.items()):
        if p.poll() is not None: log(f"end {k} rc={p.returncode} {(time.time() - t0) / 60:.0f} min"); del run[k]
    while jobs and len(run) < P and avail() > 300:
        ds, fam, app, flag, a, z = jobs.pop(0); k = f"{ds} {fam} {app} causalai {a}"
        cmd = ["timeout", "432000", PY, "e2e_shard.py", "--data", W + "/data/causal", "--dataset", ds, "--family", fam, "--app", app,
               "--backend", "causalai", "--start", str(a), "--stop", str(z)] + ([flag] if flag else [])
        run[k] = (subprocess.Popen(cmd, cwd=W + "/repo_lean/experiments", env=env, stdout=open(f"{LOG}/{ds}_{fam}_{app}_{a}.log", "a"),
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True), time.time())
        log("start " + k); time.sleep(1)
    time.sleep(20)
log("APP6_DONE")
