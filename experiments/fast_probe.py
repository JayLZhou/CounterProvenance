#!/usr/bin/env python3
"""Where the time of a query over many tables goes, and a vectorised read: all tables in one array, the cell of every
group from one row per axis and slab, one fancy index and one sum."""

from __future__ import annotations

import argparse
import json
import math
import time
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

from counterprovenance import ProvenanceIndex
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from counterprovenance.scm_baselines import ScmThreeStepPy
from scm_bench import choose_treatments, zoo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--budget", type=float, default=1e8)
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    y = str(meta["target"])
    scm = fit_scm(df, edges, y, [sorted(ancestors(parents_of(edges), y))[0]], zoo(1000, 1)[args.family],
                  max_rows=50_000, seed=0, n_jobs=4)
    unit = df.iloc[[int(np.random.default_rng(2028).integers(len(df), size=3)[0])]].iloc[0].to_dict()
    T = choose_treatments(scm, y, args.d)
    bound = bind_scm(scm, unit, T, y, threads=args.threads)
    ix = ProvenanceIndex(bound, grouping="greedy", budget=int(args.budget), grid_cap=int(args.budget),
                         max_leaf_cells=2 * 10**9, mask_ratio=1e18, lean_cells=True, build_threads=args.threads)
    rng = np.random.default_rng(5)
    lo, hi = df[T].quantile(0.01).to_numpy(), df[T].quantile(0.99).to_numpy()
    Q = rng.uniform(lo, hi, size=(args.queries, args.d))
    X = [ix._cast(t) for t in Q]

    def clock(fn, reps=1):
        t0 = time.perf_counter()
        out = [fn(x) for x in X]
        return (time.perf_counter() - t0) / len(X) * 1e6, out

    us_full, old = clock(lambda x: ix._query_fast(x).value)
    paxes = ix._paxes
    us_cast = (lambda t0: ([ix._cast(t) for t in Q], (time.perf_counter() - t0) / len(Q) * 1e6)[1])(time.perf_counter())
    us_bisect, _ = clock(lambda x: [bisect_left(th, x[a]) for a, th, _ in paxes])

    def packed(x):
        p = 0
        for a, th, rows in paxes:
            p += rows[bisect_left(th, x[a])]
        return memoryview(p.to_bytes(ix._nbytes, "little")).cast("I")
    us_packed, cells = clock(packed)

    # the vectorised read
    G = len(ix._tables)
    sizes = np.array([len(t) for t in ix._tables], dtype=np.int64)
    base = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    table = np.concatenate([np.frombuffer(t, dtype=np.float64) for t in ix._tables])
    by_axis: dict[int, list] = {}
    for g, a, contrib in ix._contrib:
        by_axis.setdefault(a, []).append((g, contrib))
    axes = []
    for a, lst in sorted(by_axis.items()):
        groups = np.array([g for g, _ in lst], dtype=np.int64)
        rows = np.stack([c for _, c in lst], axis=1).astype(np.int64)   # slabs x groups of the axis
        axes.append((a, ix.slabs.thresholds[a], groups, rows))
    const = ix._const
    th_lists = [list(map(float, t)) for _, t, _, _ in axes]

    def fast(x):
        c = base.copy()
        for (a, _, groups, rows), th in zip(axes, th_lists):
            c[groups] += rows[bisect_left(th, x[a])]
        return const + float(table[c].sum())
    us_np, new = clock(fast)
    err = float(np.max(np.abs(np.array(old) - np.array(new))))
    # order the reads by address: fewer cache misses for many tables?
    def fast_sorted(x):
        c = base.copy()
        for (a, _, groups, rows), th in zip(axes, th_lists):
            c[groups] += rows[bisect_left(th, x[a])]
        c.sort()
        return const + float(table[c].sum())
    us_sorted, _ = clock(fast_sorted)
    three = ScmThreeStepPy(scm, unit, T, y)
    t0 = time.perf_counter()
    for t in Q[:100]:
        three.query(t)
    base_us = (time.perf_counter() - t0) / 100 * 1e6
    print(f"{args.dataset} {args.family} d={args.d} groups={G} cells={len(table):.2e} | now {us_full:.1f}us (cast {us_cast:.1f} bisect {us_bisect:.1f} packed {us_packed:.1f}) | numpy {us_np:.1f}us sorted {us_sorted:.1f}us | err {err:.1e} | three-step {base_us:.0f}us -> x{base_us / min(us_full, us_np):.0f}", flush=True)


if __name__ == "__main__":
    main()
