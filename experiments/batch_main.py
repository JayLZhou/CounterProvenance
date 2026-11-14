#!/usr/bin/env python3
"""Extra experiment, batches: the same intervention for all 1,000 units of the main experiment at once.

For each setting, the 10 query points of the first unit of the main experiment are the interventions; each one is
applied to all units in one call, i.e., 10 batches of 1,000 queries:

* ``causalai``: CausalAI as designed (``causalai_baseline.ScmCausalAIBatch``): its data are the units, abduction is one
  ``predict`` per variable over all units, and one pass per intervention, one ``predict`` per variable;
* ``dowhy``: ``gcm.counterfactual_samples`` with all units as ``observed_data``;
* ``twin``: the index of every mechanism (built once, offline, as in the main experiment) read with ``query_batch``,
  one vector pass per mechanism over all units: abduction reads the factual parents, then each intervention replays
  the path, ``u + f(pa') - f(pa)``.  ``--d 0`` runs every d on one build.
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
    ap.add_argument("--backend", required=True, choices=["causalai", "dowhy", "twin"])
    ap.add_argument("--d", type=int, required=True, help="0: every d (twin)")
    ap.add_argument("--trees", type=int, default=1000)
    ap.add_argument("--units", type=int, default=1000)
    ap.add_argument("--per-unit", type=int, default=10)
    ap.add_argument("--out", default="batch")
    ap.add_argument("--budget", type=float, default=2e7)
    ap.add_argument("--one-pairs", type=float, default=5e10)
    ap.add_argument("--one-max", type=float, default=5e7)
    ap.add_argument("--build-threads", type=int, default=4, help="threads of the offline build only")
    args = ap.parse_args()
    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    state = prepare(args)
    if args.backend == "twin":
        return run_twin(args, state, out)
    if args.d not in state["work"]:
        return
    run(args, state, out, args.d, args.backend)


def run_twin(args, state, out):
    from counterprovenance.scm import relevant_nodes
    from mech_proto import MechanismIndex
    scm, y, rows, work = state["scm"], state["y"], state["rows"], state["work"]
    dims = [args.d] if args.d else sorted(work)
    nodes = sorted({x for d in dims for x in relevant_nodes(scm.parents, work[d]["T"], y)})
    t0 = time.perf_counter()
    try:
        mech = MechanismIndex(scm, rows.iloc[0].to_dict(), nodes, args.budget, args.build_threads, args.one_pairs, args.one_max)
    except MemoryError as e:
        for d in dims:
            pd.DataFrame([dict(dataset=args.dataset, family=args.family, d=d, backend="twin", status=f"infeasible: {str(e)[:80]}")]).to_csv(
                out / f"{args.dataset}_{args.family}_d{d}_twin.csv", index=False)
        return
    build_s = time.perf_counter() - t0
    V = {c: rows[c].to_numpy(dtype=float) for c in rows.columns}
    n = len(rows)
    for d in dims:
        T = work[d]["T"]
        order = relevant_nodes(scm.parents, T, y)

        def solve(points):
            obs = {x: mech.indexes[x].query_batch(np.column_stack([V[p] for p in scm.parents[x]])) for x in order}
            res = []
            for t in points:
                W = dict(V)
                for k, name in enumerate(T):
                    W[name] = np.full(n, float(t[k]))
                for x in order:
                    W[x] = V[x] + (mech.indexes[x].query_batch(np.column_stack([W[p] for p in scm.parents[x]])) - obs[x])
                res.append(W[order[-1]])
            return res
        run(args, state, out, d, "twin", solve=solve, extra=dict(build_s=build_s))


def run(args, state, out, d, backend, solve=None, extra=None):
    scm, y, rows, T = state["scm"], state["y"], state["rows"], state["work"][d]["T"]
    args = argparse.Namespace(**{**vars(args), "d": d, "backend": backend})
    points = next(queries(state, args.d, args.per_unit))
    got = []
    t0 = time.perf_counter()
    setup = 0.0
    if solve is not None:
        got = solve(points)
    elif args.backend == "causalai":
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
             query_s=total, setup_s=setup, mean_us=total / n * 1e6, max_diff=max(diffs), diff_over_1e9=int(sum(x > 1e-9 for x in diffs)), status="ok", **(extra or {}))
    pd.DataFrame([r]).to_csv(out / f"{args.dataset}_{args.family}_d{args.d}_{args.backend}.csv", index=False)
    np.save(out / f"{args.dataset}_{args.family}_d{args.d}_{args.backend}.npy", got)
    print(r, flush=True)


if __name__ == "__main__":
    main()
