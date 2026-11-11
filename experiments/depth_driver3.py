#!/usr/bin/env python3
"""Driver of depth_exp3: 11 tree ensembles x 5 families x 6 depths; small depths first, P runs at once."""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
ROOT = "/mnt/data/yingli/counterprovenance"; W = ROOT + "/depth_exp3"
P = int(sys.argv[1]) if len(sys.argv) > 1 else 12
M = [("ailerons", "SeTime12"), ("elevators", "SaTime2"), ("pol", "f17"), ("mslr", "BM25-body"), ("elevators", "Goal"),
     ("ailerons", "goal"), ("mslr", "relevance"), ("pol", "foo"), ("census1990", "dIncome1"), ("baf", "fraud_bool"),
     ("jdk", "total_unit_test_time")]
T = [(d, n, f, h) for h in (3, 6, 10, 20, 50, 100) for f in ("GBR", "HGB", "RF", "XGB", "LGBM") for d, n in M]
def go(t):
    d, n, f, h = t
    if os.path.exists(f"{W}/{d}__{n}__{f}__{h}.csv"):
        return
    cmd = f"ulimit -v 250000000; timeout 10800 {ROOT}/env/bin/python {W}/depth_exp3.py '{d}' '{n}' {f} {h}"
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    print(d, n, f, h, "rc", p.returncode, (p.stdout or "").strip()[-400:], (p.stderr or "").strip()[-300:], flush=True)
with ThreadPoolExecutor(P) as ex:
    list(ex.map(go, T))
print("ALL DONE", flush=True)
