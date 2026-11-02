#!/usr/bin/env python3
"""C++ against Python compilation on real settings: identical boxes, and the speedup.

Python is given ``--py-limit`` seconds per setting, so that very slow settings still report the C++ time and
box count.  Output: ``results/native_check.csv``.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import write_csv
from counterprovenance.scm import Infeasible, ancestors, bind_scm, fit_scm, parents_of
from scm_bench import choose_treatments, zoo


class TimeUp(Exception):
    pass


def alarm(*_):
    raise TimeUp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--settings", required=True, help="comma-separated dataset:family:d")
    ap.add_argument("--py-limit", type=int, default=1800)
    ap.add_argument("--tag", default="native_check")
    ap.add_argument("--max-boxes", type=int, default=50_000_000)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    signal.signal(signal.SIGALRM, alarm)
    rows = []
    for spec in args.settings.split(","):
        ds, fam, d = spec.split(":")
        d = int(d)
        d0 = Path(args.data) / ds
        meta = json.loads((d0 / "meta.json").read_text())
        df = pd.read_parquet(d0 / "data.parquet")
        df.columns = [str(c) for c in df.columns]
        edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
        outcome = str(meta["target"])
        scm = fit_scm(df, edges, outcome, [sorted(ancestors(parents_of(edges), outcome))[0]], zoo(1000, 1)[fam],
                      max_rows=50_000, seed=0, n_jobs=4)
        unit_idx = np.random.default_rng(2028).integers(len(df), size=3)
        unit = df.iloc[[int(unit_idx[0])]].iloc[0].to_dict()
        T = choose_treatments(scm, outcome, d)
        row = dict(dataset=ds, family=fam, d=d)
        t0 = time.perf_counter()
        try:
            cc = bind_scm(scm, unit, T, outcome, max_boxes=args.max_boxes, native=True, threads=args.threads)
            row.update(cpp_s=time.perf_counter() - t0, boxes=len(cc.boxes), threads=args.threads)
        except Infeasible as e:
            row.update(cpp_s=time.perf_counter() - t0, boxes=-1, note=str(e))
            cc = None
        if cc is not None:
            signal.alarm(args.py_limit)
            t0 = time.perf_counter()
            try:
                py = bind_scm(scm, unit, T, outcome, max_boxes=args.max_boxes, native=False)
                signal.alarm(0)
                row["py_s"] = time.perf_counter() - t0
                same = len(py.boxes) == len(cc.boxes) and py.offset == cc.offset and all(
                    x.tree == y.tree and x.node == y.node and x.native_leaf == y.native_leaf and x.lo == y.lo
                    and x.hi == y.hi and x.value == y.value for x, y in zip(py.boxes, cc.boxes))
                row["identical"] = bool(same)
                row["speedup"] = row["py_s"] / max(row["cpp_s"], 1e-9)
                if not same:
                    raise AssertionError(f"{spec}: C++ and Python boxes differ")
            except TimeUp:
                row["py_s"] = float("nan")
                row["identical"] = "py over limit"
        rows.append(row)
        write_csv(f"{args.tag}.csv", rows)
        print(row, flush=True)


if __name__ == "__main__":
    main()
