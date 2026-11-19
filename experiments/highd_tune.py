#!/usr/bin/env python3
"""High d: dense group tables vs per-tree leaf masks.

Trees whose own grid table exceeds ``grid_cap`` are answered by leaf bitmasks in
rank space (one 64-bit field per 64 leaves, AND over the axes); the others by
grouped dense tables.  For each ``dataset:family:d`` this sweeps ``grid_cap``
and reports the groups, the bitmask trees, the index bytes, the time of one
numba call per query and the native time per query (numba loop), on 10,000
queries checked against the default index.  Output: ``results/highd_tune.csv``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import per_call_seconds, write_csv
from counterprovenance import ProvenanceIndex, accel
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from counterprovenance.scm_baselines import ScmOnlinePy
from scm_bench import choose_treatments, zoo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--settings", required=True, help="comma-separated dataset:family:d")
    ap.add_argument("--caps", default="4194304")
    ap.add_argument("--ratios", default="inf,256,64,16", help="mask_ratio values (inf = grid_cap only)")
    ap.add_argument("--queries", type=int, default=10_000)
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
        bound = bind_scm(scm, unit, T, outcome)
        Q = np.ascontiguousarray(rng.uniform(df[T].quantile(0.01), df[T].quantile(0.99), size=(args.queries, d)))
        online = ScmOnlinePy(scm, unit, T, outcome)
        online.query(Q[0])
        online_us = per_call_seconds(lambda: [online.query(t) for t in Q], len(Q)) * 1e6
        ref = None
        for cap, ratio in [(int(c), float(r)) for c in args.caps.split(",") for r in args.ratios.split(",")]:
            start = time.perf_counter()
            index = ProvenanceIndex(bound, grid_cap=cap, mask_ratio=ratio)
            build_s = time.perf_counter() - start
            nb = accel.NumbaIndex(index)
            got = np.array([nb.query(t) for t in Q])
            if ref is None:
                ref = got
            err = float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))))
            call_us = per_call_seconds(lambda: [nb.query(t) for t in Q], len(Q)) * 1e6
            nb.batch(Q)
            start = time.perf_counter()
            for _ in range(3):
                nb.batch(Q)
            native_us = (time.perf_counter() - start) / 3 / len(Q) * 1e6
            st = index.stats
            row = dict(dataset=ds, family=fam, d=d, grid_cap=cap, mask_ratio=ratio, boxes=len(bound.boxes), groups=len(index.groups),
                       bitmask_trees=st.bitmask_trees, bytes=st.packed_bytes, build_s=build_s, err=err,
                       call_us=call_us, native_us=native_us, online_nb_us=online_us)
            rows.append(row)
            write_csv("highd_tune.csv", rows)
            print(f"{ds} {fam} d={d} cap={cap:>8} ratio={ratio:>5} groups={row['groups']:5d} bitmask={row['bitmask_trees']:5d} "
                  f"MB={row['bytes'] / 1e6:8.1f} build={build_s:6.1f}s call={call_us:6.2f}us native={native_us:6.2f}us "
                  f"online={online_us:6.1f}us err={err:.0e}", flush=True)


if __name__ == "__main__":
    main()
