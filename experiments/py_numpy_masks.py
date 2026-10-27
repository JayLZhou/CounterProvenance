#!/usr/bin/env python3
"""Python index with the leaf-mask step vectorised by numpy, at high d.

Per axis, the mask rows are a ``(slabs, fields)`` uint64 array; a query ANDs one
row per axis, reads every field's leaf as ``frexp`` exponent of the remaining bit
(= ``bit_length``), gathers the leaf values and sums them.  Compared with the
index's pure-Python query and with pure-Python propagation.  Only settings whose
index uses leaf masks for every varying tree.  Output: ``results/py_numpy_masks.csv``.
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

from common import per_call_seconds, write_csv
from counterprovenance import ProvenanceIndex
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from counterprovenance.scm_baselines import ScmOnlinePy
from scm_bench import choose_treatments, zoo


class NumpyMasks:
    def __init__(self, index):
        self.cast = index._cast
        self.axes = []
        for a, th, _ in index._baxes:
            self.axes.append((a, th, index._bmasks[a]))  # (slabs, fields) uint64
        F = len(index._bvals)
        self.vals = np.zeros((F, 65))
        for j, v in enumerate(index._bvals):
            self.vals[j, :len(v)] = v
        self.rowsel = np.arange(F)
        self.const = index._const

    def query(self, t):
        x = self.cast(t)
        it = iter(self.axes)
        a, th, M = next(it)
        m = M[bisect_left(th, x[a])].copy()
        for a, th, M in it:
            m &= M[bisect_left(th, x[a])]
        leaf = np.frexp(m.astype(np.float64))[1]
        return self.const + float(self.vals[self.rowsel, leaf].sum())


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
        index = ProvenanceIndex(bound)
        if index._fast_leaves and any(l.axes for l in index._fast_leaves):
            print(f"{spec}: has table groups; skipped", flush=True)
            continue
        Q = rng.uniform(df[T].quantile(0.01), df[T].quantile(0.99), size=(args.queries, d))
        npm = NumpyMasks(index)
        py = ScmOnlinePy(scm, unit, T, outcome)
        err = max(abs(npm.query(t) - index.query(t).value) / max(1.0, abs(index.query(t).value)) for t in Q[:500])
        row = dict(dataset=ds, family=fam, d=d, fields=len(index._bvals), err=err,
                   numpy_us=per_call_seconds(lambda: [npm.query(t) for t in Q], len(Q)) * 1e6,
                   python_us=per_call_seconds(lambda: [index.query(t) for t in Q], len(Q)) * 1e6,
                   online_us=per_call_seconds(lambda: [py.query(t) for t in Q], len(Q), repeats=1) * 1e6)
        rows.append(row)
        write_csv("py_numpy_masks.csv", rows)
        print(f"{ds} {fam} d={d} fields={row['fields']} | numpy {row['numpy_us']:.1f}us python {row['python_us']:.1f}us "
              f"online {row['online_us']:.1f}us | numpy {row['online_us'] / row['numpy_us']:.1f}x python "
              f"{row['online_us'] / row['python_us']:.1f}x | err {err:.0e}", flush=True)


if __name__ == "__main__":
    main()
