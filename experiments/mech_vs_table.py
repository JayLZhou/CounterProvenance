#!/usr/bin/env python3
"""The read of CellTable without groups against CellForest, on the outcome's ensemble, by the same code: CellTable is the index with one
group over all inputs, CellForest the index with our groups.  The two answer the same points in alternation.
Output: ``results/vs_table/<dataset>_<family>.csv``."""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import pandas as pd

from common import RESULTS
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import bind_scm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--points", type=int, default=5000)
    args = ap.parse_args()
    state = pickle.loads((RESULTS / "main" / "cache" / f"{args.dataset}_{args.family}.pkl").read_bytes())
    scm, y, rows = state["scm"], state["y"], state["rows"]
    pa = list(scm.parents[y])
    bound = bind_scm(scm, rows.iloc[0].to_dict(), pa, y, threads=1, native=True)
    t0 = time.perf_counter()
    twin = ProvenanceIndex(bound, grouping="tables", budget=int(2e7), build_threads=1)
    twin_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    table = ProvenanceIndex(bound, grouping="none", native_cells=True, max_leaf_cells=10**13, build_threads=1)
    table_s = time.perf_counter() - t0
    # points of the workload: observed rows, with the inputs drawn uniformly over their range in half of them
    P = rows[pa].to_numpy(dtype=float)
    rng = np.random.default_rng(2028)
    X = P[rng.integers(len(P), size=args.points)]
    U = rng.uniform(P.min(0), P.max(0), size=X.shape)
    X[1::2] = U[1::2]
    for x in X[:200]:  # warm up both
        twin.query(x), table.query(x)
    a_t, b_t, wrong = [], [], 0
    for k, x in enumerate(X):
        order = ((twin, a_t), (table, b_t)) if k % 2 == 0 else ((table, b_t), (twin, a_t))
        got = []
        for ix, out in order:
            t0 = time.perf_counter()
            v = ix.query(x).value
            out.append(time.perf_counter() - t0)
            got.append(v)
        wrong += abs(got[0] - got[1]) > 1e-6
    a, b = np.array(a_t) * 1e6, np.array(b_t) * 1e6
    r = dict(dataset=args.dataset, family=args.family, node=y, parents=len(pa), twin_groups=len(twin._fast_leaves), table_groups=len(table._fast_leaves),
             twin_bytes=twin.stats.packed_bytes, table_cells=table.stats.faces, table_grid=table.stats.grid_cells, table_index_bytes=table.stats.packed_bytes,
             twin_build_s=twin_s, table_build_s=table_s, twin_us=float(a.mean()), table_us=float(b.mean()), twin_p50=float(np.median(a)), table_p50=float(np.median(b)),
             twin_p95=float(np.quantile(a, .95)), table_p95=float(np.quantile(b, .95)), points=len(X), wrong=int(wrong))
    out = RESULTS / "vs_table"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r]).to_csv(out / f"{args.dataset}_{args.family}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
