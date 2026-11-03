"""All the shards of the whole-table runs, cheapest first; a shard that was stopped continues from its checkpoint."""
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

W = Path("/mnt/data/yingli/counterprovenance")
ROWS = dict(adult=45222, give_credit=120269, heloc=10000, german_full=1000)
SIZE = dict(adult=2000, give_credit=5000, heloc=1000, german_full=1000)
R, M1, M2, G = "dowhy_feature_relevance", "mcce_1000", "mcce_10000", "geco_genetic"
PLAN = [("german_full", "HGB100", R, "dowhy", None), ("german_full", "HGB100", R, "three_step", None),
        ("adult", "RF200", M1, "dowhy", None), ("adult", "RF200", M2, "dowhy", None), ("adult", "RF500D10", G, "dowhy", None),
        ("adult", "RF500D10", G, "geco", None), ("heloc", "HGB100", R, "dowhy", None), ("adult", "HGB100", R, "dowhy", None),
        ("adult", "RF200", M1, "three_step", None), ("adult", "RF500D10", G, "three_step", None),
        ("give_credit", "HGB100", R, "dowhy", None), ("adult", "RF100", R, "dowhy", None), ("adult", "RF200", R, "dowhy", None),
        ("adult", "RF200", M2, "three_step", 2000), ("adult", "HGB100", R, "three_step", 1000), ("adult", "RF200", R, "three_step", 200)]
TWIN = [("german_full", "HGB100", R), ("adult", "RF200", M1), ("adult", "RF200", M2), ("adult", "RF500D10", G), ("heloc", "HGB100", R),
        ("adult", "HGB100", R), ("give_credit", "HGB100", R), ("adult", "RF100", R), ("adult", "RF200", R)]
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
if "gc" in sys.argv:    # Give Me Credit with forests: one index of hundreds of GB
    PLAN = [("give_credit", "RF100", R, "dowhy", None), ("give_credit", "RF200", R, "dowhy", None)]
    TWIN = [("give_credit", "RF100", R), ("give_credit", "RF200", R)]
if "ts" in sys.argv:    # the interventional TreeSHAP of the shap library, against the same background
    TWIN = []
    PLAN = [(d, f, R, "treeshap", None) for d, f in (("german_full", "HGB100"), ("heloc", "HGB100"), ("give_credit", "HGB100"),
                                                     ("adult", "HGB100"), ("adult", "RF100"), ("adult", "RF200"),
                                                     ("give_credit", "RF100"), ("give_credit", "RF200"))]
env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{W / 'shap_pkgs'}:{W / 'chirho_pkgs'}")
jobs = [(d, f, a, "twin", 0, -1) for d, f, a in TWIN]
for d, f, a, b, cap in PLAN:
    total = cap or ROWS[d]
    size = min(SIZE[d], max(1, math.ceil(total / 20)))
    jobs += [(d, f, a, b, s, min(s + size, total)) for s in range(0, total, size)]


def running():
    return subprocess.run(["pgrep", "-fc", "python e2e_shard[.]py"], capture_output=True, text=True).stdout.strip()


def finished(d, f, a, b, s, e):
    path = W / "repo_dev2/results/shard" / f"{d}_{f}_{a}_{b}_{s}.csv"
    if not path.exists():
        return False
    r = pd.read_csv(path).iloc[0]
    return int(r.rows_done) >= int(r.stop) - int(r.start)


print(len(jobs), "shards", flush=True)
(W / "logs/shard").mkdir(exist_ok=True)
for d, f, a, b, s, e in jobs:
    if finished(d, f, a, b, s, e):
        continue
    while int(running() or 0) >= LIMIT:
        time.sleep(30)
    log = W / "logs/shard" / f"{d}_{f}_{a}_{b}_{s}.log"
    subprocess.Popen([str(W / "env/bin/python"), "e2e_shard.py", "--data", str(W / "data/causal"), "--dataset", d, "--family", f, "--app", a,
                      "--backend", b, "--start", str(s), "--stop", str(e)], cwd=W / "repo_dev2/experiments", env=env,
                     stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(time.strftime("%F %T"), "started", d, f, a, b, s, flush=True)
    time.sleep(2)
print("all launched", flush=True)
