#!/usr/bin/env python3
"""Prototype: faster leaf bitmasks at high d.

Variants of the per-axis leaf masks (one row per axis and global slab):

* ``plain``  - one 64-bit field per tree (per 64 leaves), branch on empty fields;
* ``nobranch`` - trees with at most 64 leaves always keep exactly one bit, so the
  leaf read needs no branch;
* ``pack``   - two trees with at most 32 leaves share one 64-bit field (low and
  high half), halving the words read and ANDed per axis;
* ``pack+nobranch``.

Trees with more than 64 leaves (RF) keep one field per 64 leaves and the branch.
Checked against the index on 10,000 queries.  Output: ``results/mask_fast_proto.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from numba.cpython.unsafe.numbers import trailing_zeros

from common import per_call_seconds, write_csv
from counterprovenance import ProvenanceIndex, accel
from counterprovenance.ensemble import Slabs
from counterprovenance.scm import ancestors, bind_scm, fit_scm, parents_of
from counterprovenance.scm_baselines import ScmNumba
from scm_bench import choose_treatments, zoo


def build(bound, pack):
    slabs = Slabs(bound)
    d = bound.d
    by_tree: dict[int, list[int]] = {}
    for i, b in enumerate(bound.boxes):
        by_tree.setdefault(b.tree, []).append(i)
    const = bound.offset
    small, big = [], []
    for m, items in sorted(by_tree.items()):
        if len(items) == 1:
            const += bound.boxes[items[0]].value
        elif len(items) <= (32 if pack else 64):
            small.append(items)
        else:
            big.append(items)
    # word layout: packed words (two small trees each, or one), then the fields of big trees
    words = []  # list of [(items, shift)]
    if pack:
        for s in range(0, len(small), 2):
            words.append([(small[s], 0)] + ([(small[s + 1], 32)] if s + 1 < len(small) else []))
    else:
        words += [[(it, 0)] for it in small]
    n_single = len(words)  # words whose every half holds exactly one set bit after the AND
    for items in big:
        for s in range(0, len(items), 64):
            words.append([(items[s:s + 64], 0)])
    W = len(words)
    vals = np.zeros((W, 64))
    halves = np.zeros(W, dtype=np.int64)  # 2 = two packed trees, 1 = one tree
    counts = slabs.counts
    rows = [np.zeros((counts[a], W), dtype=np.uint64) for a in range(d)]
    for j, parts in enumerate(words):
        halves[j] = len(parts)
        for items, shift in parts:
            for bit, i in enumerate(items):
                vals[j, shift + bit] = bound.boxes[i].value
                for a in range(d):
                    lo, hi = slabs.ranges[i][a]
                    rows[a][lo:hi, j] |= np.uint64(1 << (shift + bit))
    th = [np.asarray(t, dtype=np.float64) for t in slabs.thresholds]
    width = max(1, max(len(t) for t in th))
    tharr = np.full((d, width), np.inf)
    thlen = np.array([len(t) for t in th], dtype=np.int64)
    for a, t in enumerate(th):
        tharr[a, :len(t)] = t
    row_off = np.zeros(d + 1, dtype=np.int64)
    row_off[1:] = np.cumsum([r.size for r in rows])
    return dict(W=W, n_single=n_single, halves=halves, rows=np.concatenate([r.ravel() for r in rows]),
                row_off=row_off, vals=vals.ravel(), const=const, tharr=tharr, thlen=thlen,
                f32=int(bound.ensemble.input_dtype == np.float32),
                zt=bound.ensemble.zero_threshold if bound.ensemble.zero_threshold > 0 else -1.0,
                mb=sum(r.nbytes for r in rows) / 1e6)


@njit(cache=True)
def _query(t, W, n_single, halves, rows, row_off, vals, const, tharr, thlen, f32, zt, nobranch, work):
    d = tharr.shape[0]
    for a in range(d):
        v = t[a]
        if f32 == 1:
            v = np.float64(np.float32(v))
        if -zt <= v <= zt:
            v = 0.0
        lo, hi = 0, thlen[a]
        while lo < hi:
            mid = (lo + hi) >> 1
            if tharr[a, mid] < v:
                lo = mid + 1
            else:
                hi = mid
        base = row_off[a] + lo * W
        if a == 0:
            for j in range(W):
                work[j] = rows[base + j]
        else:
            for j in range(W):
                work[j] &= rows[base + j]
    total = const
    lo32 = np.uint64(0xFFFFFFFF)
    if nobranch:
        for j in range(n_single):
            m = work[j]
            total += vals[j * 64 + np.int64(trailing_zeros(m | np.uint64(1) << np.uint64(63)))] \
                if halves[j] == 1 else vals[j * 64 + np.int64(trailing_zeros(m & lo32))] + \
                vals[j * 64 + np.int64(trailing_zeros(m >> np.uint64(32))) + 32]
    else:
        for j in range(n_single):
            m = work[j]
            if halves[j] == 1:
                if m != 0:
                    total += vals[j * 64 + np.int64(trailing_zeros(m))]
            else:
                a_ = m & lo32
                b_ = m >> np.uint64(32)
                if a_ != 0:
                    total += vals[j * 64 + np.int64(trailing_zeros(a_))]
                if b_ != 0:
                    total += vals[j * 64 + 32 + np.int64(trailing_zeros(b_))]
    for j in range(n_single, W):
        m = work[j]
        if m != 0:
            total += vals[j * 64 + np.int64(trailing_zeros(m))]
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--settings", required=True, help="comma-separated dataset:family:d")
    ap.add_argument("--queries", type=int, default=10_000)
    args = ap.parse_args()
    out, fitted = [], {}
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
        Q = np.ascontiguousarray(rng.uniform(df[T].quantile(0.01), df[T].quantile(0.99), size=(args.queries, d)))
        nb = accel.NumbaIndex(ProvenanceIndex(bound))
        ref = np.array([nb.query(t) for t in Q])
        index_us = per_call_seconds(lambda: [nb.query(t) for t in Q], len(Q)) * 1e6
        online = ScmNumba(scm, unit, T, outcome)
        online.query(Q[0])
        online_us = per_call_seconds(lambda: [online.query(t) for t in Q], len(Q)) * 1e6
        for pack in (False, True):
            p = build(bound, pack)
            work = np.zeros(max(1, p["W"]), dtype=np.uint64)
            for nobranch in (False, True):
                argv = (p["W"], p["n_single"], p["halves"], p["rows"], p["row_off"], p["vals"], p["const"],
                        p["tharr"], p["thlen"], p["f32"], p["zt"], nobranch, work)
                got = np.array([_query(t, *argv) for t in Q])
                err = float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))))
                us = per_call_seconds(lambda: [_query(t, *argv) for t in Q], len(Q)) * 1e6
                name = ("pack" if pack else "plain") + ("+nobranch" if nobranch else "")
                row = dict(dataset=ds, family=fam, d=d, variant=name, words=p["W"], mask_mb=p["mb"], err=err,
                           mask_us=us, index_us=index_us, online_us=online_us)
                out.append(row)
                write_csv("mask_fast_proto.csv", out)
                print(f"{ds} {fam} d={d} {name:14s} words={p['W']:5d} {p['mb']:6.2f} MB | {us:6.2f}us "
                      f"(index {index_us:6.2f}, online {online_us:6.1f}) | err {err:.0e}", flush=True)


if __name__ == "__main__":
    main()
