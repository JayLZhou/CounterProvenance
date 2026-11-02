#!/usr/bin/env python3
"""Effect of the table budget on TWIN: groups, build time, memory, query latency.  One setting per process.
Output: ``results/beta_probe_<tag>.csv``."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import write_csv
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
    ap.add_argument("--budget", type=float, default=2e6)
    ap.add_argument("--grid-cap", type=float, default=1 << 22)
    ap.add_argument("--one-cells", type=float, default=2e6)
    ap.add_argument("--max-leaf-cells", type=float, default=2e7)
    ap.add_argument("--mask-ratio", type=float, default=64.0)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--native-cells", action="store_true")
    ap.add_argument("--lean", action="store_true")
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
    unit_idx = np.random.default_rng(2028).integers(len(df), size=3)
    rows = []
    for u in range(1):
        unit = df.iloc[[int(unit_idx[u])]].iloc[0].to_dict()
        T = choose_treatments(scm, y, args.d)
        rng = np.random.default_rng([2028, u, args.d])
        t0 = time.perf_counter()
        bound = bind_scm(scm, unit, T, y, threads=args.threads)
        compile_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        index = ProvenanceIndex(bound, grouping="auto", budget=int(args.budget), grid_cap=int(args.grid_cap),
                                one_cells=int(args.one_cells), max_leaf_cells=int(args.max_leaf_cells),
                                mask_ratio=args.mask_ratio, max_mask_bytes=float("inf"),
                                native_cells=args.native_cells, lean_cells=args.lean, build_threads=args.threads)
        build_s = time.perf_counter() - t0
        lo, hi = df[T].quantile(0.01).to_numpy(), df[T].quantile(0.99).to_numpy()
        Q = rng.uniform(lo, hi, size=(args.queries, args.d))
        t0 = time.perf_counter()
        for t in Q:
            index.query(t)
        us = (time.perf_counter() - t0) / len(Q) * 1e6
        st = index.stats
        three = ScmThreeStepPy(scm, unit, T, y)
        err = max(abs(index.query(t).value - three.query(t)[0]) for t in Q[:100])
        t0 = time.perf_counter()
        for t in Q[:100]:
            three.query(t)
        base_us = (time.perf_counter() - t0) / 100 * 1e6
        rows.append(dict(max_abs_err=err, base_us=base_us, dataset=args.dataset, family=args.family, d=args.d, unit=u, budget=args.budget,
                         grid_cap=args.grid_cap, one_cells=args.one_cells, boxes=len(bound.boxes), choice=index.choice,
                         groups=len(index.groups), bitmask_trees=st.bitmask_trees, bytes=st.packed_bytes,
                         compile_s=compile_s, build_s=build_s, us=us,
                         peak_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576))
        write_csv(f"beta_probe_{args.tag}.csv", rows)
        print(rows[-1], flush=True)


if __name__ == "__main__":
    main()
