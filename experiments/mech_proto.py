#!/usr/bin/env python3
"""A prototype: one index per mechanism instead of one per unit.

The index of a mechanism f_X is built over all parents of X, so that it depends on no unit.  A query on any unit and any
intervened variables walks the mediators and the outcome in topological order and reads one index each:
X(t) = u[X] + f_X(pa') - f_X(u[pa]).  Compared with the per-unit index of the main experiment and checked against the
three-step evaluation.  Output: ``results/mech_<dataset>_<family>.csv``, one row per (d, unit)."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of, relevant_nodes
from counterprovenance.scm_baselines import ScmThreeStepPy
from scm_bench import choose_treatments, zoo


class MechanismIndex:
    def __init__(self, scm, unit0, nodes, budget, threads, one_pairs=5e10):
        self.scm, self.indexes, self.info = scm, {}, {}
        for x in nodes:
            pa = list(scm.parents[x])
            t0 = time.perf_counter()
            bound = bind_scm(scm, unit0, pa, x, threads=threads, native=True)
            index = ProvenanceIndex(bound, grouping="tables", budget=int(budget), build_threads=threads, one_pairs=float(one_pairs))
            st = index.stats
            self.indexes[x] = index
            self.info[x] = dict(parents=len(pa), boxes=len(bound.boxes), groups=len(index.groups), cells=st.grid_cells,
                                bytes=st.packed_bytes, build_s=time.perf_counter() - t0)

    def query(self, unit, T, t, order, observed):
        """One query, as the per-unit index is measured: one read per mechanism on the path."""
        values = dict(unit)
        for name, v in zip(T, t):
            values[name] = float(v)
        for x in order:
            y = self.indexes[x].query([values[p] for p in self.scm.parents[x]]).value
            values[x] = unit[x] + (y - observed[x])
        return values[order[-1]]

    def plan(self, T, order):
        """The partial reads of these intervened variables: a parent varies if it is intervened or a mediator."""
        vary = set(T) | set(order)
        return [(x, list(self.scm.parents[x]), self.indexes[x].partial([i for i, p in enumerate(self.scm.parents[x]) if p in vary]))
                for x in order]

    def bind(self, plan, unit, observed):
        """What stays the same for a unit: its noise, and the groups that no varying parent reaches."""
        return [(x, pa, partial, partial.bind([unit[p] for p in pa]), float(unit[x]), observed[x]) for x, pa, partial in plan]

    def query_partial(self, unit, T, t, bound):
        """One query that only reads the groups of the varying parents."""
        values = dict(unit)
        for name, v in zip(T, t):
            values[name] = float(v)
        x = None
        for x, pa, partial, b, ux, obs in bound:
            values[x] = ux + (partial.value([values[p] for p in pa], b) - obs)
        return values[x]

    def observed(self, unit, order):
        return {x: self.indexes[x].query([unit[p] for p in self.scm.parents[x]]).value for x in order}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--dims", default="2,4,8")
    ap.add_argument("--units", type=int, default=3)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--budget", type=float, default=2e7)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    y = str(meta["target"])
    scm = fit_scm(df, edges, y, [sorted(ancestors(parents_of(edges), y))[0]], zoo(1000, 1)[args.family], max_rows=50_000,
                  seed=0, n_jobs=4)
    unit_idx = np.random.default_rng(2028).integers(len(df), size=args.units)
    units = [df.iloc[[int(i)]].iloc[0].to_dict() for i in unit_idx]
    dims = [d for d in (int(v) for v in args.dims.split(",")) if choose_treatments(scm, y, d)]
    nodes = sorted({x for d in dims for x in relevant_nodes(scm.parents, choose_treatments(scm, y, d), y)})
    tag = RESULTS / "mech" / f"{args.dataset}_{args.family}_{args.budget:.0e}.csv"
    tag.parent.mkdir(parents=True, exist_ok=True)
    base = dict(dataset=args.dataset, family=args.family, budget=args.budget, mechanisms=len(nodes))
    t0 = time.perf_counter()
    try:
        mech = MechanismIndex(scm, units[0], nodes, args.budget, args.threads)
    except MemoryError as e:
        pd.DataFrame([dict(base, status=f"infeasible: {str(e)[:70]}", build_s=time.perf_counter() - t0)]).to_csv(tag, index=False)
        return
    build_s = time.perf_counter() - t0
    info = mech.info
    base.update(status="ok", build_s=build_s, bytes=sum(v["bytes"] for v in info.values()), cells=sum(v["cells"] for v in info.values()),
                max_parents=max(v["parents"] for v in info.values()), max_groups=max(v["groups"] for v in info.values()),
                sum_groups=sum(v["groups"] for v in info.values()),
                peak_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576)
    print(json.dumps({k: base[k] for k in ("mechanisms", "build_s", "bytes", "cells", "max_parents", "max_groups", "sum_groups", "peak_gb")}), flush=True)
    rows = []
    for d in dims:
        T = choose_treatments(scm, y, d)
        order = relevant_nodes(scm.parents, T, y)
        lo, hi = df[T].quantile(0.01).to_numpy(), df[T].quantile(0.99).to_numpy()
        for u, unit in enumerate(units):
            # the same queries as the main experiment: 3/4 uniform, 1/4 with one coordinate on a leaf-box boundary
            rng = np.random.default_rng([2028, u, d])
            per_unit = bind_scm(scm, unit, T, y, threads=args.threads, native=True)
            n_unif = args.queries * 3 // 4
            pts = list(rng.uniform(lo, hi, size=(n_unif, d)))
            th = [[b.hi[a] for b in per_unit.boxes if np.isfinite(b.hi[a])] for a in range(d)]
            for _ in range(args.queries - n_unif):
                p = rng.uniform(lo, hi)
                a = int(rng.integers(d))
                if th[a]:
                    p[a] = th[a][int(rng.integers(len(th[a])))]
                pts.append(p)
            Q = np.ascontiguousarray(pts, dtype=np.float64)
            t0 = time.perf_counter()
            observed = mech.observed(unit, order)
            noise_us = (time.perf_counter() - t0) * 1e6
            lat = []
            got = []
            for t in Q:
                t0 = time.perf_counter()
                got.append(mech.query(unit, T, t, order, observed))
                lat.append(time.perf_counter() - t0)
            lat = np.array(lat) * 1e6
            three = ScmThreeStepPy(scm, unit, T, y)
            k = len(Q)
            ref = np.array([three.query(t)[0] for t in Q[:k]])
            wrong = int((np.abs(np.array(got[:k]) - ref) > 1e-6 * (1 + np.abs(ref))).sum())
            r = dict(base, d=d, unit=u, path=len(order), noise_us=noise_us, mech_us=float(lat.mean()), mech_p50_us=float(np.median(lat)),
                     mech_p95_us=float(np.quantile(lat, .95)), checked=k, wrong=wrong)
            rows.append(r)
            pd.DataFrame(rows).to_csv(tag, index=False)
            print(json.dumps({k: r.get(k) for k in ("d", "unit", "path", "mech_us", "mech_p50_us", "wrong")}), flush=True)


if __name__ == "__main__":
    main()
