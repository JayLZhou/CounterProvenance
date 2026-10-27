#!/usr/bin/env python3
"""Pure Python at high d: the index (default leaf masks for deep trees) vs pure-Python propagation.

Output: ``results/py_highd.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import per_call_seconds, write_csv
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from counterprovenance.scm_baselines import ScmOnlinePy
from scm_bench import choose_treatments, zoo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--queries", type=int, default=2000)
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
        Q = rng.uniform(df[T].quantile(0.01), df[T].quantile(0.99), size=(args.queries, d))
        py = ScmOnlinePy(scm, unit, T, outcome)
        for ratio in (float("inf"), 64.0):
            index = ProvenanceIndex(bound, mask_ratio=ratio)
            err = max(abs(index.query(t).value - py.query(t)[0]) / max(1.0, abs(py.query(t)[0])) for t in Q[:300])
            row = dict(dataset=ds, family=fam, d=d, mask_ratio=ratio, groups=len(index.groups),
                       mask_trees=index.stats.bitmask_trees, err=err,
                       ours_us=per_call_seconds(lambda: [index.query(t) for t in Q], len(Q)) * 1e6,
                       online_us=per_call_seconds(lambda: [py.query(t) for t in Q], len(Q), repeats=1) * 1e6)
            rows.append(row)
            write_csv("py_highd.csv", rows)
            print(f"{ds} {fam} d={d} ratio={ratio} groups={row['groups']} masks={row['mask_trees']} | ours "
                  f"{row['ours_us']:.1f}us online {row['online_us']:.1f}us ({row['online_us'] / row['ours_us']:.1f}x) "
                  f"err {err:.0e}", flush=True)


if __name__ == "__main__":
    main()
