"""Measured ablation and the remaining updates, after the timing sample: Exp.5/6 structure statistics on the final code; the index without the MST
(mech_table), without grouping (mech_rss --grouping none) and without both (mech_naive) really built, at most 1 TB of resident memory and
6 hours per model; the Give Me Credit RF100 application on the final index.  A grid whose map alone needs over 1 TB (4 B per grid cell) is not attempted."""
import os, resource, subprocess, sys, time
from pathlib import Path
import pandas as pd

W = Path("/mnt/data/yingli/counterprovenance")
ABL, LEAN = W / "repo_abl", W / "repo_lean"
PY, DATA = str(W / "env/bin/python"), str(W / "data/causal")
LOGS = W / "logs/abl"; LOGS.mkdir(parents=True, exist_ok=True)
BUDGET_GB, WEIGHT, FLOOR_GB, CAP_KB = 1700.0, 40, 250, 1 << 30
SETS = "concrete,light_tunnel,ailerons,pol,elevators,jdk,postgres,baf,mslr,census1990,susy,methane,higgs,airline,nyc_taxi".split(",")
FAMS = ("GBR", "HGB", "RF", "XGB", "LGBM")
H = 3600


def say(*a):
    with open(LOGS / "sched.txt", "a") as f:
        print(time.strftime("%F %T"), *a, file=f, flush=True)


def available_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


while "BASE2_DONE" not in (W / "logs/base2/progress.txt").read_text():
    time.sleep(60)
say("timing sample done, starting")
stat = {}
for ds in SETS:
    for f in FAMS:
        t = pd.read_csv(LEAN / "results/rss" / f"{ds}_{f}.csv")
        old = H and W / "repo_hybrid/results/rss_none" / f"{ds}_{f}.csv"
        n = pd.read_csv(old) if old.exists() else None
        log = W / "logs/none" / f"{ds}_{f}.log"
        ok = log.exists() and "done" in log.read_text(errors="ignore")
        stat[ds, f] = dict(grid=float(t.grid_all.max()), peak=t.peak.max() / 1e9, nomst=(t.formula - 8 * t.pft_nodes + 4 * t.all_leaves).sum() / 1e9,
                           none_peak=n.peak.max() / 1e9 if ok and n is not None else None)
env_of = lambda repo: dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{repo}:{W / 'chirho_pkgs'}", TMPDIR=str(W / "tmp"))
jobs, skipped = [], []   # name, argv, repo, est_gb, weight, seconds, rlimit
for ds, f in sorted(stat, key=lambda k: stat[k]["grid"]):
    jobs.append((f"struct_{ds}_{f}", ["mech_structure.py", "--dataset", ds, "--family", f], ABL, 30.0, 1, 3 * H, 100 << 30))
for ds, f in sorted(stat, key=lambda k: stat[k]["peak"]):
    s = stat[ds, f]
    jobs.append((f"table_{ds}_{f}", ["mech_table.py", "--dataset", ds, "--family", f], ABL, min(1050.0, s["peak"] + 1.3 * s["nomst"] + 5), 1, 6 * H, None))
for ds, f in sorted(stat, key=lambda k: stat[k]["grid"]):
    s = stat[ds, f]
    if s["grid"] > 2.5e11:
        skipped.append(f"{ds}_{f}")
        continue
    old_ok = (W / "repo_hybrid/results/naive" / f"{ds}_{f}.csv").exists()
    if not old_ok:
        jobs.append((f"naive_{ds}_{f}", ["mech_naive.py", "--dataset", ds, "--family", f], ABL, min(1050.0, 8 * s["grid"] / 1e9 + 20), 1, 6 * H, None))
    est = s["none_peak"] * 1.2 + 5 if s["none_peak"] else min(1050.0, s["grid"] * 1000 * 2.9e-8 + 5)
    th = 8 if s["grid"] > 1e9 else 1
    jobs.append((f"none_{ds}_{f}", ["mech_rss.py", "--dataset", ds, "--family", f, "--grouping", "none", "--threads", str(th)], ABL, est, th, 6 * H, None))
jobs.append(("app_give_credit_RF100", ["e2e_shard.py", "--data", DATA, "--dataset", "give_credit", "--family", "RF100", "--app", "dowhy_feature_relevance", "--backend", "twin"], LEAN, 80.0, 4, 30 * H, None))
say(len(jobs), "jobs;", len(skipped), "models whose map alone exceeds 1 TB, not attempted:", " ".join(skipped))
# restart: no naive CellTable (user, 2026-11-08); started jobs are not run again, the ones still running are adopted
log_lines = (LOGS / "sched.txt").read_text().splitlines()
started = {l.split()[3] for l in log_lines if len(l.split()) > 3 and l.split()[2] == "start"}
ended = {l.split()[3] for l in log_lines if len(l.split()) > 3 and l.split()[2] == "end"}
jobs = [j for j in jobs if not j[0].startswith("naive_")]
# methane XGB without the MST was stopped at 1 TB by CellForest's own build peak (1.13 TB), not by the table of the variant: once more, with 1.3 TB
started.discard("table_methane_XGB"); ended.discard("table_methane_XGB")
jobs = [j[:3] + (1191.0,) + j[4:] if j[0] == "table_methane_XGB" else j for j in jobs]


def external(job):
    sig = " ".join(job[1])
    out = subprocess.run(["ps", "-u", "yingli", "-o", "args="], capture_output=True, text=True).stdout.splitlines()
    return any(l.startswith(PY) and sig in l for l in out)


queue = [j for j in jobs if j[0] not in started]
running = [(None, j, time.time()) for j in jobs if j[0] in started and j[0] not in ended and external(j)]
say(f"restart without naive: {len(queue)} to run, adopted", " ".join(j[0] for _, j, _ in running))


def watch():
    out = subprocess.run(["ps", "-u", "yingli", "-o", "pid=,rss=,args="], capture_output=True, text=True).stdout.splitlines()
    for line in out:
        p = line.split(None, 2)
        if len(p) == 3 and p[2].startswith(PY) and int(p[1]) > (CAP_KB * 13 // 10 if "mech_table.py --dataset methane --family XGB" in p[2] else CAP_KB) and ("mech_table" in p[2] or "mech_naive" in p[2] or "grouping none" in p[2]):
            for _, job, _ in running:
                if " ".join(job[1]) in p[2]:
                    with open(LOGS / f"{job[0]}.log", "a") as log:
                        log.write(f"{time.strftime('%F %T')} resident memory over 1 TB ({int(p[1]) >> 20} GB): stopped\n")
                    say(f"over 1 TB {job[0]}")
            subprocess.run(["kill", p[0]])


while queue or running:
    watch()
    for r in running[:]:
        p, job, t0 = r
        if (p is None and external(job)) or (p is not None and p.poll() is None):
            continue
        running.remove(r)
        say(f"end {job[0]} rc={None if p is None else p.returncode} {(time.time() - t0) / 60:.0f} min")
    used, weight = sum(j[3] for _, j, _ in running), sum(j[4] for _, j, _ in running)
    for job in list(queue):
        if used + job[3] <= BUDGET_GB and weight + job[4] <= WEIGHT and available_gb() - job[3] >= FLOOR_GB:
            queue.remove(job)
            lim = job[6]
            pre = (lambda lim=lim: resource.setrlimit(resource.RLIMIT_AS, (lim, lim))) if lim else None
            p = subprocess.Popen(["timeout", str(job[5]), PY] + job[1], cwd=job[2] / "experiments", env=env_of(job[2]), stdout=open(LOGS / f"{job[0]}.log", "a"),
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, preexec_fn=pre)
            running.append((p, job, time.time()))
            used += job[3]; weight += job[4]
            say(f"start {job[0]} est {job[3]:.0f} GB (reserved {used:.0f}, weight {weight})")
            if job[3] > 100:
                break
    time.sleep(20)
say("all done")
