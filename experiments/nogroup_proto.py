#!/usr/bin/env python3
"""No grouping: one cell graph for the whole forest when it fits, else every tree on its own.

``one``: all trees in a single cell graph (MCPST) when the slab grid of all axes has at most
``--budget`` cells; ``trees``: every varying tree located by its own leaf masks; ``greedy``:
the current greedy grouping.  Pure Python (numpy for the masks), compared with the three-step
baseline.  Output: ``results/nogroup_proto.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import per_call_seconds, write_csv
from counterprovenance import ProvenanceIndex
from counterprovenance.ensemble import Slabs
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from counterprovenance.scm_baselines import ScmThreeStepPy
from scm_bench import choose_treatments, zoo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=4_000_000)
    args = ap.parse_args()
    rows, fitted = [], {}
    for spec in args.settings.split(","):
        ds, fam, d = spec.split(":")
        d = int(d)
        if (ds, fam) not in fitted:
            d0 = Path(args.data) / ds
            meta = json.loads((d0 / "meta.json").read_text())
            df = pd.read_parquet(d0 / "data.parquet")
            df.columns = [str(c) for c in df.columns]
            edges = [tuple(r) for r in pd.read_csv(d0 / "graph.csv", dtype=str).itertuples(index=False)]
            outcome = str(meta["target"])
            scm = fit_scm(df, edges, outcome, [sorted(ancestors(parents_of(edges), outcome))[0]],
                          zoo(1000, 1)[fam], max_rows=50_000, seed=0, n_jobs=4)
            fitted[(ds, fam)] = (df, outcome, scm)
        df, outcome, scm = fitted[(ds, fam)]
        rng = np.random.default_rng(2028)
        unit = df.iloc[int(rng.integers(len(df)))].to_dict()
        T = choose_treatments(scm, outcome, d)
        if T is None:
            continue
        bound = bind_scm(scm, unit, T, outcome)
        grid = math.prod(Slabs(bound).counts)
        Q = rng.uniform(df[T].quantile(0.01), df[T].quantile(0.99), size=(args.queries, d))
        three = ScmThreeStepPy(scm, unit, T, outcome)
        base_us = per_call_seconds(lambda: [three.query(t) for t in Q[:500]], 500, repeats=1) * 1e6
        configs = {"greedy": dict(), "trees": dict(mask_ratio=0.0)}
        if grid <= args.budget:
            configs["one"] = dict(grouping="none", max_leaf_cells=args.budget)
        ref = None
        for name, kw in configs.items():
            start = time.perf_counter()
            index = ProvenanceIndex(bound, **kw)
            build_s = time.perf_counter() - start
            vals = np.array([index.query(t).value for t in Q[:500]])
            ref = vals if ref is None else ref
            err = float(np.max(np.abs(vals - ref) / np.maximum(1.0, np.abs(ref))))
            us = per_call_seconds(lambda: [index.query(t) for t in Q], len(Q)) * 1e6
            row = dict(dataset=ds, family=fam, d=d, config=name, grid=grid, groups=len(index.groups),
                       mask_trees=index.stats.bitmask_trees, mb=index.stats.packed_bytes / 1e6, build_s=build_s,
                       ours_us=us, three_step_us=base_us, err=err)
            rows.append(row)
            write_csv("nogroup_proto.csv", rows)
            print(f"{ds} {fam} d={d} {name:6s} grid={grid:>14,} groups={row['groups']:4d} masks={row['mask_trees']:4d} "
                  f"{row['mb']:7.1f}MB build={build_s:6.1f}s | {us:7.1f}us vs three-step {base_us:7.0f}us "
                  f"({base_us / us:6.0f}x) err {err:.0e}", flush=True)


if __name__ == "__main__":
    main()
