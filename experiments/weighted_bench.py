#!/usr/bin/env python3
"""Queries with the trees' weights chosen at query time: the first N trees and arbitrary weights, CellForest against the
three steps, both in Python on the same queries, every answer compared.  Also the sizes: value tables, persistent
trees, leaf masks, and the size of storing the leaves per cell independently.  Output: ``results/weighted_<tag>.csv``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import write_csv
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from counterprovenance.scm_baselines import ScmThreeStepPy
from counterprovenance.weighted import WeightedIndex
from scm_bench import choose_treatments, zoo


def timed(fn, items):
    t0 = time.perf_counter()
    out = [fn(x) for x in items]
    return (time.perf_counter() - t0) / len(items) * 1e6, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--dims", default="1,2,4,8")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--base-queries", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    y = str(meta["target"])
    scm = fit_scm(df, edges, y, [sorted(ancestors(parents_of(edges), y))[0]], zoo(1000, 1)[args.family],
                  max_rows=50_000, seed=0, n_jobs=4)
    M = len(scm.ensembles[y].trees)
    unit = df.iloc[[int(np.random.default_rng(2028).integers(len(df), size=3)[0])]].iloc[0].to_dict()
    rows = []
    for d in (int(v) for v in args.dims.split(",")):
        T = choose_treatments(scm, y, d)
        if not T:
            continue
        rng = np.random.default_rng([2028, 0, d])
        t0 = time.perf_counter()
        bound = bind_scm(scm, unit, T, y, threads=args.threads)
        compile_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        index = ProvenanceIndex(bound, grouping="auto", native_cells=True, build_threads=args.threads)
        build_s = time.perf_counter() - t0
        wi = WeightedIndex(index, float(unit[y]), [float(unit[a]) for a in T])
        three = ScmThreeStepPy(scm, unit, T, y)
        lo, hi = df[T].quantile(0.01).to_numpy(), df[T].quantile(0.99).to_numpy()
        Q = rng.uniform(lo, hi, size=(args.queries, d))
        Qb = Q[: args.base_queries]
        st = index.stats
        table_b = sum(8 * len(t) for t in index._tables)
        pft_b = sum(16 * c.cells.stats.pft_nodes + 4 * c.cells.stats.faces for c in index._fast_leaves)
        iip_b = sum(4 * c.cells.stats.faces * c.cells.stats.slots for c in index._fast_leaves)
        row = dict(dataset=args.dataset, family=args.family, d=d, trees=M, boxes=len(bound.boxes), choice=index.choice,
                   groups=len(index.groups), masks_trees=st.bitmask_trees, compile_s=compile_s, build_s=build_s,
                   table_mb=table_b / 1e6, pft_mb=pft_b / 1e6, masks_mb=8 * st.bitmask_words / 1e6, iip_mb=iip_b / 1e6)
        row["fixed_us"], _ = timed(lambda t: index.query(t).value, Q)
        row["base_fixed_us"], _ = timed(lambda t: three.query(t)[0], Qb)
        worst = 0.0
        for n in (M // 10, M // 2):
            us, got = timed(lambda t, n=n: wi.prefix(t, n), Q)
            bus, ref = timed(lambda t, n=n: three.query_prefix(t, n), Qb)
            worst = max(worst, float(np.max(np.abs(np.array(got[: len(Qb)]) - np.array(ref)))))
            row[f"prefix{n}_us"], row[f"base_prefix{n}_us"] = us, bus
        W = rng.uniform(0, 2, size=(len(Q), M))
        pairs = list(zip(Q, W))
        row["weighted_us"], got = timed(lambda p: wi.weighted(p[0], p[1]), pairs)
        row["base_weighted_us"], ref = timed(lambda p: three.query_weighted(p[0], p[1]), pairs[: args.base_queries])
        worst = max(worst, float(np.max(np.abs(np.array(got[: len(Qb)]) - np.array(ref)))))
        row["max_abs_err"] = worst
        rows.append(row)
        write_csv(f"weighted_{args.tag}.csv", rows)
        print(row, flush=True)


if __name__ == "__main__":
    main()
