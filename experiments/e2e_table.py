#!/usr/bin/env python3
"""An application over a whole table: every row is explained or gets its recourse.

With all parents of the outcome intervened there is no mediator and g_u(t) = f_Y(t) + n_u: the units only differ by a
constant, so that one index serves the whole table.  CellForest runs on all rows (up to ``--hours``); the online evaluation
and DoWhy-GCM run on a random sample of the same rows, and their time is projected to the table.
Output: ``results/table_<dataset>_<family>_<app>.csv``, one row per backend."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import write_csv
from counterprovenance.scm import ancestors, fit_scm, parents_of
from e2e_apps import APPS, Ctx, DoWhy, Online, Oracle, CellForest, model_factory
from scm_full import dowhy_model


def at(base: Ctx, row: pd.DataFrame) -> Ctx:
    c = copy.copy(base)
    c.row = row
    c.u = np.clip(row[c.T].to_numpy(dtype=float)[0], c.lo, c.hi)
    return c


class Shifted:
    """The index of one unit for another unit: the same trees, the noise of the other unit."""

    def __init__(self, twin, shift):
        self.twin, self.shift = twin, shift

    def __call__(self, Q):
        return self.twin(Q) + self.shift


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--app", required=True)
    ap.add_argument("--sample", type=int, default=100, help="rows of the online evaluation and of DoWhy-GCM")
    ap.add_argument("--hours", type=float, default=6.0, help="limit of CellForest")
    ap.add_argument("--build-threads", type=int, default=8)
    args = ap.parse_args()
    d0 = Path(args.data) / args.dataset
    meta = json.loads((d0 / "meta.json").read_text())
    df = pd.read_parquet(d0 / "data.parquet")
    df.columns = [str(c) for c in df.columns]
    edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
    y = str(meta["target"])
    scm = fit_scm(df, edges, y, [sorted(ancestors(parents_of(edges), y))[0]], model_factory(args.family),
                  max_rows=50_000, seed=0, n_jobs=4)
    T = list(scm.parents[y])
    order = np.random.default_rng(2028).permutation(len(df))
    first = df.iloc[[int(order[0])]]
    base = Ctx(df, T, first, y, 2028, meta.get("increase_only", ()), meta.get("mcce_immutable", ()))
    base.lo = np.minimum(base.lo, df[T].quantile(0.01).to_numpy())   # the box does not depend on the unit
    twin = CellForest(scm, first.iloc[0].to_dict(), T, y, 2e7, args.build_threads)
    f0 = float(twin(first[T].to_numpy(dtype=float))[0])
    common = dict(dataset=args.dataset, family=args.family, app=args.app, d=len(T), table_rows=len(df), mechanisms=len(scm.models))
    app = APPS[args.app]
    tag = f"table_{args.dataset}_{args.family}_{args.app}.csv"
    rows = [dict(common, backend="twin", build_s=twin.build_s, **twin.info)]

    # the sample first, so that a limit on CellForest still leaves a comparison
    sample = order[: args.sample]
    outs = {}
    gcm, cg = dowhy_model(scm, edges)
    for name in ("online", "dowhy"):
        qs = total = 0.0
        n = 0
        got = []
        for i in sample:
            row = df.iloc[[int(i)]]
            backend = Online(scm, row.iloc[0].to_dict(), T, y) if name == "online" else DoWhy(gcm, cg, row, T, y)
            f = Oracle(backend)
            t0 = time.perf_counter()
            got.append(app(f, at(base, row)))
            total += time.perf_counter() - t0
            qs += f.query_s
            n += f.queries
        outs[name] = np.array(got)
        rows.append(dict(common, backend=name, rows_done=len(sample), queries=n, query_s=qs, run_s=total,
                         us_per_query=qs / n * 1e6, projected_days=qs / len(sample) * len(df) / 86400))
        write_csv(tag, rows)
        print(json.dumps({k: rows[-1].get(k) for k in ("backend", "rows_done", "queries", "query_s", "us_per_query", "projected_days")}), flush=True)

    r = rows[0]
    qs = total = 0.0
    n = done = 0
    got = []
    start = time.perf_counter()
    for i in order:
        row = df.iloc[[int(i)]]
        # g_u(t) - g_first(t) is the difference of the noises, read at the unit's own parents
        shift = float(row[y].iloc[0]) - float(first[y].iloc[0]) - (float(twin(row[T].to_numpy(dtype=float))[0]) - f0)
        f = Oracle(Shifted(twin, shift))
        t0 = time.perf_counter()
        out = app(f, at(base, row))
        total += time.perf_counter() - t0
        qs += f.query_s
        n += f.queries
        done += 1
        if done <= args.sample:
            got.append(out)
        if done % 100 == 0 or done == len(df) or time.perf_counter() - start > args.hours * 3600:
            r.update(rows_done=done, queries=n, query_s=qs, run_s=total, us_per_query=qs / n * 1e6,
                     projected_days=qs / done * len(df) / 86400, wall_s=time.perf_counter() - start)
            if len(got) == args.sample:
                for name, o in outs.items():
                    r[f"max_err_{name}"] = float(np.abs(np.array(got) - o).max())
            write_csv(tag, rows)
            if time.perf_counter() - start > args.hours * 3600:
                break
    print(json.dumps({k: r.get(k) for k in ("backend", "rows_done", "queries", "query_s", "us_per_query", "projected_days", "max_err_online", "max_err_dowhy")}), flush=True)


if __name__ == "__main__":
    main()
