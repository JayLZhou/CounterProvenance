#!/usr/bin/env python3
"""Extra experiment, batches: the same intervention for all 1,000 units of the main experiment at once.

For each setting, the 10 query points of the first unit of the main experiment are the interventions; each one is
applied to all units in one call, i.e., 10 batches of 1,000 queries:

* ``causalai``: CausalAI as designed (``causalai_baseline.ScmCausalAIBatch``): its data are the units, abduction is one
  ``predict`` per variable over all units, and one pass per intervention, one ``predict`` per variable;
* ``dowhy``: ``gcm.counterfactual_samples`` with all units as ``observed_data``.

TWIN has no batch mode: answering the same 10,000 queries is the work of the main experiment (results/main).
20 answers per setting are checked against ``ThreeStep``.  Output: ``results/batch/<dataset>_<family>_d<d>_<backend>.csv``."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance.scm_baselines import ScmThreeStepPy
from mech_main import prepare, queries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--backend", required=True, choices=["causalai", "dowhy"])
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--trees", type=int, default=1000)
    ap.add_argument("--units", type=int, default=1000)
    ap.add_argument("--per-unit", type=int, default=10)
    ap.add_argument("--out", default="batch")
    args = ap.parse_args()
    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    state = prepare(args)
    if args.d not in state["work"]:
        return
    scm, y, rows, T = state["scm"], state["y"], state["rows"], state["work"][args.d]["T"]
    points = next(queries(state, args.d, args.per_unit))
    got = []
    t0 = time.perf_counter()
    if args.backend == "causalai":
        from causalai_baseline import ScmCausalAIBatch
        ca = ScmCausalAIBatch(scm, rows, T, y)
        setup = time.perf_counter() - t0
        for t in points:
            got.append(ca.query_all(t))
    else:
        from scm_full import dowhy_model
        gcm, cg = dowhy_model(scm, state["edges"])
        setup = time.perf_counter() - t0
        for t in points:
            iv = {name: (lambda _v, value=float(t[k]): value) for k, name in enumerate(T)}
            got.append(gcm.counterfactual_samples(cg, iv, observed_data=rows)[y].to_numpy())
    total = time.perf_counter() - t0
    got = np.array(got)
    rng = np.random.default_rng(7)
    diffs = []
    for _ in range(20):
        i, u = int(rng.integers(len(points))), int(rng.integers(len(rows)))
        exact = ScmThreeStepPy(scm, rows.iloc[u].to_dict(), T, y).query(points[i])[0]
        diffs.append(abs(float(got[i, u]) - exact))
    n = got.size
    r = dict(dataset=args.dataset, family=args.family, d=args.d, backend=args.backend, units=len(rows), batches=len(points), queries=n,
             query_s=total, setup_s=setup, mean_us=total / n * 1e6, max_diff=max(diffs), diff_over_1e9=int(sum(x > 1e-9 for x in diffs)), status="ok")
    pd.DataFrame([r]).to_csv(out / f"{args.dataset}_{args.family}_d{args.d}_{args.backend}.csv", index=False)
    np.save(out / f"{args.dataset}_{args.family}_d{args.d}_{args.backend}.npy", got)
    print(r, flush=True)


if __name__ == "__main__":
    main()
