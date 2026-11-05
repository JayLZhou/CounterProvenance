#!/usr/bin/env python3
"""Exp.7: the leaf boxes of the index for one unit, with and without the dominance test, on one unit per setting.

Without the test a box is cut at every boundary of a mediator inside it; the run stops at ``--max-boxes`` boxes."""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance.scm import Infeasible, _Mediator, bind_scm, relevant_nodes


def covered(bound, t):
    return float(sum(b.value for b in bound.boxes if all(b.lo[a] < t[a] <= b.hi[a] for a in range(bound.d))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--max-boxes", type=int, default=20_000_000)
    args = ap.parse_args()
    state = pickle.loads((RESULTS / "main" / "cache" / f"{args.dataset}_{args.family}.pkl").read_bytes())
    scm, y, rows, work = state["scm"], state["y"], state["rows"], state["work"]
    out = RESULTS / "structure" / f"{args.dataset}_{args.family}_dominance.csv"
    unit = rows.iloc[0].to_dict()
    got = []
    for d in (1, 2, 4):
        if d not in work:
            continue
        T = work[d]["T"]
        order = relevant_nodes(scm.parents, T, y)
        r = dict(dataset=args.dataset, family=args.family, d=d, mediators=len(order) - 1)
        t0 = time.perf_counter()
        on = bind_scm(scm, unit, T, y, max_boxes=10**15, native=True, threads=1)
        r.update(with_boxes=len(on.boxes), with_s=time.perf_counter() - t0)
        _Mediator.dominance = False
        t0 = time.perf_counter()
        try:
            off = bind_scm(scm, unit, T, y, max_boxes=args.max_boxes, native=False)
            r.update(without_boxes=len(off.boxes), status="ok")
            if len(off.boxes) <= 1_000_000:
                Q = np.random.default_rng([2028, d]).uniform(work[d]["lo"], work[d]["hi"], size=(10, d))
                r["same_answers"] = all(abs(covered(on, t) - covered(off, t)) < 1e-6 for t in Q)
        except (Infeasible, MemoryError) as e:
            r.update(without_boxes=args.max_boxes, status=f"over: {str(e)[:60]}")
        _Mediator.dominance = True
        r["without_s"] = time.perf_counter() - t0
        got.append(r)
        pd.DataFrame(got).to_csv(out, index=False)
        print(r, flush=True)


if __name__ == "__main__":
    main()
