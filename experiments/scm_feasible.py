#!/usr/bin/env python3
"""Rerun the settings that exceeded the limits with the mediator dominance test.

For each ``dataset:family:d``: fit, compile through the mediators, build the index,
check 400 queries against ``ScmOnlinePy`` and time the numba index against it.
Output: ``results/scm_feasible.csv``.
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
    ap.add_argument("--units", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rows = []
    fitted = {}
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
        for u in range(args.units):
            unit = df.iloc[int(rng.integers(len(df)))].to_dict()
            T = choose_treatments(scm, outcome, d)
            start = time.perf_counter()
            bound = bind_scm(scm, unit, T, outcome)
            compile_s = time.perf_counter() - start
            index = ProvenanceIndex(bound)
            build_s = time.perf_counter() - start
            Q = rng.uniform(df[T].quantile(0.01).to_numpy(), df[T].quantile(0.99).to_numpy(), size=(400, d))
            py = ScmOnlinePy(scm, unit, T, outcome)
            truth = np.array([py.query(t)[0] for t in Q])
            nb_ours = accel.NumbaIndex(index)
            nb_ours.query(Q[0])
            err = max(float(np.max(np.abs(np.array(v) - truth) / np.maximum(1.0, np.abs(truth))))
                      for v in ([index.query(t).value for t in Q], [nb_ours.query(t) for t in Q]))
            row = dict(dataset=ds, family=fam, d=d, unit=u, mediators=len(py.mechs) - 1, boxes=len(bound.boxes),
                       groups=len(index.groups), compile_s=compile_s, build_s=build_s, max_rel_err=err,
                       ours_nb_us=per_call_seconds(lambda: [nb_ours.query(t) for t in Q], len(Q)) * 1e6,
                       online_py_us=per_call_seconds(lambda: [py.query(t) for t in Q], len(Q)) * 1e6)
            rows.append(row)
            write_csv(f"scm_feasible{args.tag}.csv", rows)
            print(f"{ds} {fam} d={d} unit {u} med={row['mediators']} boxes={row['boxes']:,} build={build_s:.1f}s "
                  f"err={err:.1e} nb {row['ours_nb_us']:.2f} vs {row['online_py_us']:.1f}us", flush=True)


if __name__ == "__main__":
    main()
