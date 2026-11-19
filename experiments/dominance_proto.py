#!/usr/bin/env python3
"""Prototype: the counterfactual value as a dominance sum over signed box corners.

A leaf box ``(lo, hi]`` is ``prod_a ([t_a > lo_a] - [t_a > hi_a])`` over its bounded
axes; expanding the product turns it into at most ``2^k`` signed corners ``c`` with
``[c < t]`` (strict dominance), so

    g(t) = offset + sum over corners c of  weight(c) * [c < t].

Corners at the same point are merged.  Two dominance-sum queries are timed with
numba: a scan over all corners, and a k-d tree whose nodes keep the bounding box
and the weight sum of their corners (a node entirely dominated by ``t`` adds its
sum, a node whose minimum corner ``t`` does not dominate is skipped).  Compared
with the index (``NumbaIndex``) and propagation (``ScmOnlinePy``) on the same
10,000 queries.  Output: ``results/dominance_proto.csv``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from common import per_call_seconds, write_csv
from counterprovenance import ProvenanceIndex, accel
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from counterprovenance.scm_baselines import ScmOnlinePy
from scm_bench import choose_treatments, zoo


def corners(bound):
    """Merged signed corners (coordinates with -inf for 'always dominated', weights) and the constant."""
    d = bound.d
    acc: dict[tuple, float] = {}
    const = [bound.offset]
    for b in bound.boxes:
        options = []
        for a in range(d):
            lo, hi = b.lo[a], b.hi[a]
            if lo == -math.inf and hi == math.inf:
                options.append([(-math.inf, 1.0)])
            else:
                opt = [(lo, 1.0)]
                if hi != math.inf:
                    opt.append((hi, -1.0))
                options.append(opt)
        for combo in itertools.product(*options):
            c = tuple(x for x, _ in combo)
            s = math.prod(sg for _, sg in combo)
            if all(x == -math.inf for x in c):
                const.append(s * b.value)
            else:
                acc[c] = acc.get(c, 0.0) + s * b.value
    keys = [k for k, v in acc.items() if v != 0.0]
    pts = np.array(keys, dtype=np.float64).reshape(-1, d)
    w = np.array([acc[k] for k in keys], dtype=np.float64)
    return pts, w, math.fsum(const)


@njit(cache=True)
def _scan(t, pts, w):
    s = 0.0
    for i in range(pts.shape[0]):
        ok = True
        for a in range(pts.shape[1]):
            if not pts[i, a] < t[a]:
                ok = False
                break
        if ok:
            s += w[i]
    return s


def build_kd(pts, w, leaf=16):
    """Flat k-d tree: per node (lo corner, hi corner, sum, left, right, start, end) over a permutation."""
    n, d = pts.shape
    order = np.arange(n)
    lo_l, hi_l, sum_l, left_l, right_l, s_l, e_l = [], [], [], [], [], [], []

    def rec(idx, depth):
        me = len(lo_l)
        P = pts[idx]
        lo_l.append(P.min(0))
        hi_l.append(P.max(0))
        sum_l.append(math.fsum(w[idx].tolist()))
        left_l.append(-1)
        right_l.append(-1)
        s_l.append(0)
        e_l.append(0)
        if len(idx) <= leaf:
            s_l[me] = len(perm)
            perm.extend(idx.tolist())
            e_l[me] = len(perm)
            return me
        F = np.where(np.isfinite(P), P, np.nan)  # -inf means 'always dominated'; it still separates
        has_inf = (~np.isfinite(P)).any(0)
        fmin, fmax = np.nanmin(np.where(np.isnan(F), np.inf, F), 0), np.nanmax(np.where(np.isnan(F), -np.inf, F), 0)
        spread = np.where(np.isfinite(fmax - fmin), fmax - fmin, 0.0) + np.where(has_inf & np.isfinite(fmax), 1e300, 0.0)
        a = int(np.argmax(spread))
        if spread[a] == 0.0:
            s_l[me] = len(perm)
            perm.extend(idx.tolist())
            e_l[me] = len(perm)
            return me
        col = P[:, a]
        fin = col[np.isfinite(col)]
        cut = -np.inf if (~np.isfinite(col)).sum() * 2 >= len(col) or not len(fin) else np.median(col)
        m = col <= cut
        if m.all() or not m.any():
            m = np.zeros(len(idx), dtype=bool)
            m[: len(idx) // 2] = True
            idx = idx[np.argsort(P[:, a], kind="stable")]
        l = rec(idx[m], depth + 1)
        r = rec(idx[~m], depth + 1)
        left_l[me], right_l[me] = l, r
        return me

    perm: list[int] = []
    rec(order, 0)
    return (np.array(lo_l), np.array(hi_l), np.array(sum_l), np.array(left_l, dtype=np.int64),
            np.array(right_l, dtype=np.int64), np.array(s_l, dtype=np.int64), np.array(e_l, dtype=np.int64),
            np.array(perm, dtype=np.int64))


@njit(cache=True)
def _kd_query(t, lo, hi, sums, left, right, start, end, perm, pts, w, stack):
    d = t.shape[0]
    total = 0.0
    top = 0
    stack[0] = 0
    top = 1
    visited = 0
    while top > 0:
        top -= 1
        n = stack[top]
        visited += 1
        dom_all = True  # t dominates every point of the node: hi < t on every axis
        dom_none = False  # t dominates no point: some axis with lo >= t
        for a in range(d):
            if not hi[n, a] < t[a]:
                dom_all = False
            if lo[n, a] >= t[a]:
                dom_none = True
                break
        if dom_none:
            continue
        if dom_all:
            total += sums[n]
            continue
        if left[n] < 0:
            for j in range(start[n], end[n]):
                i = perm[j]
                ok = True
                for a in range(d):
                    if not pts[i, a] < t[a]:
                        ok = False
                        break
                if ok:
                    total += w[i]
            continue
        stack[top] = left[n]
        stack[top + 1] = right[n]
        top += 2
    return total, visited


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--settings", required=True, help="comma-separated dataset:family:d")
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
        X = np.ascontiguousarray(bound.ensemble.cast(Q))  # corners compare with cast inputs
        index = ProvenanceIndex(bound)
        nb = accel.NumbaIndex(index)
        ref = np.array([nb.query(t) for t in Q])
        pts, w, const = corners(bound)
        kd = build_kd(pts, w)
        stack = np.zeros(4096, dtype=np.int64)
        scan_v = np.array([const + _scan(x, pts, w) for x in X])
        kd_out = [_kd_query(x, *kd[:7], kd[7], pts, w, stack) for x in X]
        kd_v = np.array([const + v for v, _ in kd_out])
        err = lambda v: float(np.max(np.abs(v - ref) / np.maximum(1.0, np.abs(ref))))
        online = ScmOnlinePy(scm, unit, T, outcome)
        online.query(Q[0])
        row = dict(dataset=ds, family=fam, d=d, boxes=len(bound.boxes), corners=len(w),
                   corner_mb=(pts.nbytes + w.nbytes) / 1e6, kd_nodes=len(kd[2]),
                   index_mb=index.stats.packed_bytes / 1e6, scan_err=err(scan_v), kd_err=err(kd_v),
                   kd_visited=float(np.mean([v for _, v in kd_out])),
                   scan_us=per_call_seconds(lambda: [_scan(x, pts, w) for x in X[:1000]], 1000, repeats=1) * 1e6,
                   kd_us=per_call_seconds(lambda: [_kd_query(x, *kd[:7], kd[7], pts, w, stack) for x in X], len(X)) * 1e6,
                   index_us=per_call_seconds(lambda: [nb.query(t) for t in Q], len(Q)) * 1e6,
                   online_us=per_call_seconds(lambda: [online.query(t) for t in Q], len(Q)) * 1e6)
        rows.append(row)
        write_csv("dominance_proto.csv", rows)
        print(f"{ds} {fam} d={d} boxes={row['boxes']:,} corners={row['corners']:,} ({row['corner_mb']:.1f} MB, "
              f"index {row['index_mb']:.1f} MB) | scan {row['scan_us']:.0f}us kd {row['kd_us']:.1f}us "
              f"(visits {row['kd_visited']:.0f}) index {row['index_us']:.2f}us online {row['online_us']:.1f}us | "
              f"err scan {row['scan_err']:.0e} kd {row['kd_err']:.0e}", flush=True)


if __name__ == "__main__":
    main()
