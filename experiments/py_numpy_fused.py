#!/usr/bin/env python3
"""Python leaf masks at high d with fewer numpy calls and packed words.

* ``loop``  - one numpy AND per axis (``py_numpy_masks.NumpyMasks``);
* ``fused`` - all axes in one ``(d, slabs, words)`` array: slabs of all axes by one
  comparison, the ``d`` rows by one fancy index, one ``bitwise_and.reduce``;
* ``fused+pack`` - additionally two trees with at most 32 leaves share a word.

Only settings whose index uses leaf masks for every varying tree.  Checked against
the index.  Output: ``results/py_numpy_fused.csv``.
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
from py_numpy_masks import NumpyMasks
from scm_bench import choose_treatments, zoo


class Fused:
    def __init__(self, index, pack):
        self.cast = index._cast
        axes = [a for a, _, _ in index._baxes]
        ths = [np.asarray(th, dtype=np.float64) for _, th, _ in index._baxes]
        masks = [index._bmasks[a] for a in axes]  # (slabs, fields)
        F = len(index._bvals)
        nleaf = np.array([len(v) - 1 for v in index._bvals])
        vals = np.zeros((F, 65))
        for j, v in enumerate(index._bvals):
            vals[j, :len(v)] = v
        # word j holds field lo[j] in bits 0..31 or 0..63, and field hi[j] (or none) in bits 32..63
        lo, hi = [], []
        small = [j for j in range(F) if nleaf[j] <= 32] if pack else []
        big = [j for j in range(F) if j not in set(small)]
        for s in range(0, len(small), 2):
            lo.append(small[s])
            hi.append(small[s + 1] if s + 1 < len(small) else -1)
        self.n_pairs = len(lo)
        lo += big
        hi += [-1] * len(big)
        W = len(lo)
        C = max(m.shape[0] for m in masks)
        R = np.zeros((len(axes), C, W), dtype=np.uint64)
        for k, m in enumerate(masks):
            R[k, :m.shape[0]] = m[:, lo]
            pair = np.array(hi[:self.n_pairs])
            if self.n_pairs:
                has = pair >= 0
                R[k, :m.shape[0], :self.n_pairs][:, has] |= m[:, pair[has]] << np.uint64(32)
        self.R = R
        self.TH = np.full((len(axes), max(1, max(len(t) for t in ths))), np.inf)
        for k, t in enumerate(ths):
            self.TH[k, :len(t)] = t
        self.ax = np.asarray(axes)
        self.kk = np.arange(len(axes))
        self.vlo = vals[lo].ravel()
        self.vhi = np.zeros((self.n_pairs, 65))
        for i, j in enumerate(hi[:self.n_pairs]):
            if j >= 0:
                self.vhi[i] = vals[j]
        self.vhi = self.vhi.ravel()
        self.base = np.arange(W) * 65
        self.base_hi = np.arange(self.n_pairs) * 65
        self.const = index._const
        self.W = W

    def query(self, t):
        x = np.asarray(self.cast(t))[self.ax]
        s = (self.TH < x[:, None]).sum(1)
        m = np.bitwise_and.reduce(self.R[self.kk, s], axis=0)
        if self.n_pairs:
            p = self.n_pairs
            low = m.copy()
            low[:p] &= np.uint64(0xFFFFFFFF)
            total = self.vlo[self.base + np.frexp(low.astype(np.float64))[1]].sum()
            total += self.vhi[self.base_hi + np.frexp((m[:p] >> np.uint64(32)).astype(np.float64))[1]].sum()
        else:
            total = self.vlo[self.base + np.frexp(m.astype(np.float64))[1]].sum()
        return self.const + float(total)


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
        if any(l.axes for l in index._fast_leaves):
            print(f"{spec}: has table groups; skipped", flush=True)
            continue
        Q = rng.uniform(df[T].quantile(0.01), df[T].quantile(0.99), size=(args.queries, d))
        ref = np.array([index.query(t).value for t in Q[:500]])
        py = ScmOnlinePy(scm, unit, T, outcome)
        online = per_call_seconds(lambda: [py.query(t) for t in Q], len(Q), repeats=1) * 1e6
        variants = {"loop": NumpyMasks(index), "fused": Fused(index, False), "fused+pack": Fused(index, True)}
        for name, v in variants.items():
            got = np.array([v.query(t) for t in Q[:500]])
            err = float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))))
            us = per_call_seconds(lambda: [v.query(t) for t in Q], len(Q)) * 1e6
            row = dict(dataset=ds, family=fam, d=d, variant=name, words=getattr(v, "W", len(index._bvals)), err=err,
                       ours_us=us, online_us=online)
            rows.append(row)
            write_csv("py_numpy_fused.csv", rows)
            print(f"{ds} {fam} d={d} {name:10s} words={row['words']:5d} | {us:6.1f}us online {online:6.1f}us "
                  f"({online / us:5.1f}x) | err {err:.0e}", flush=True)


if __name__ == "__main__":
    main()
